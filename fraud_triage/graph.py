"""LangGraph fraud triage pipeline.

    ingest ──▶ rules screen ──▶ auto_clear ──────────────────▶ END
                     │                                          ▲
                     └──▶ investigate ◀──▶ tools                │
                              │                                 │
                              └──▶ verdict ──▶ human_review ────┘
                                        └──────────────────────-┘

Cheap deterministic rules gate the LLM: clean transactions never reach
Claude. Suspicious ones get an agentic investigation (tool calls against
the customer ledger) and a structured verdict with a SAR-style narrative.
"""

import json
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from . import config
from .data.store import get_store
from .rules import run_rules
from .schemas import FraudVerdict, RuleResult, Transaction
from .tools import INVESTIGATION_TOOLS

INVESTIGATOR_SYSTEM = """You are a senior fraud analyst at a card issuer.
A transaction was flagged by the deterministic rules engine and you must
investigate it before a verdict is issued.

Use the tools to gather evidence: establish the customer's baseline
(profile, recent transactions), then verify the specific signals the rules
engine raised (velocity, geo feasibility). Two or three well-chosen tool
calls are usually enough — stop investigating once the picture is clear.

Weigh evidence like an analyst: a large amount alone is weak; a large
amount at a new high-risk merchant minutes after four other card-not-present
hits is a stolen card. False declines hurt real customers, so do not treat
every flag as fraud."""

VERDICT_INSTRUCTION = """Based on your investigation, issue the final verdict.
- clear: evidence is consistent with the customer's normal behavior
- review: genuinely ambiguous — a human analyst should decide
- block: strong evidence of fraud; decline and lock the card
For review or block, write a concise SAR-style narrative (who, what, when,
where, and why it is suspicious) citing the specific evidence."""


class TriageState(TypedDict):
    transaction: dict
    rule_result: Optional[dict]
    messages: Annotated[list, add_messages]
    verdict: Optional[dict]
    escalated_to_llm: bool
    human_review_queued: bool


def _build_llm():
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.MODEL, max_tokens=config.MAX_TOKENS)

    from langchain_anthropic import ChatAnthropic

    # No temperature/top_p: Opus 4.8 rejects sampling params with a 400.
    return ChatAnthropic(model=config.MODEL, max_tokens=config.MAX_TOKENS)


def build_graph(llm=None):
    """Compile the triage graph. Pass a stub `llm` for offline tests."""
    if llm is None:
        llm = _build_llm()
    llm_with_tools = llm.bind_tools(INVESTIGATION_TOOLS)
    verdict_llm = llm.with_structured_output(FraudVerdict)

    def screen(state: TriageState) -> dict:
        txn = Transaction.model_validate(state["transaction"])
        history = get_store().get_history(txn.customer_id, limit=200)
        result: RuleResult = run_rules(txn, history)
        return {"rule_result": result.model_dump(),
                "escalated_to_llm": False, "human_review_queued": False}

    def route_after_screen(state: TriageState) -> str:
        if state["rule_result"]["score"] < config.AUTO_CLEAR_THRESHOLD:
            return "auto_clear"
        return "investigate"

    def auto_clear(state: TriageState) -> dict:
        verdict = FraudVerdict(
            verdict="clear",
            risk_score=int(state["rule_result"]["score"] * 100),
            key_signals=["no rule flags above auto-clear threshold"],
            rationale="Transaction matches the customer's established pattern; "
                      "cleared by the rules engine without escalation.",
        )
        return {"verdict": verdict.model_dump()}

    def investigate(state: TriageState) -> dict:
        if not state["messages"]:
            txn_json = json.dumps(state["transaction"], indent=2)
            rules_json = json.dumps(state["rule_result"], indent=2)
            seed = [
                SystemMessage(content=INVESTIGATOR_SYSTEM),
                HumanMessage(content=(
                    f"Transaction under review:\n{txn_json}\n\n"
                    f"Rules engine output:\n{rules_json}\n\n"
                    "Investigate and then issue your verdict."
                )),
            ]
            return {"messages": seed + [llm_with_tools.invoke(seed)],
                    "escalated_to_llm": True}
        return {"messages": [llm_with_tools.invoke(state["messages"])],
                "escalated_to_llm": True}

    def verdict(state: TriageState) -> dict:
        result = verdict_llm.invoke(
            state["messages"] + [HumanMessage(content=VERDICT_INSTRUCTION)]
        )
        return {"verdict": result.model_dump()}

    def route_after_verdict(state: TriageState) -> str:
        v = state["verdict"]
        low, high = config.REVIEW_BAND
        if v["verdict"] == "review" or low <= v["risk_score"] <= high:
            return "human_review"
        return END

    def human_review(state: TriageState) -> dict:
        # In production this would push to a case-management queue.
        return {"human_review_queued": True}

    graph = StateGraph(TriageState)
    graph.add_node("screen", screen)
    graph.add_node("auto_clear", auto_clear)
    graph.add_node("investigate", investigate)
    graph.add_node("tools", ToolNode(INVESTIGATION_TOOLS))
    graph.add_node("verdict", verdict)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "screen")
    graph.add_conditional_edges("screen", route_after_screen,
                                {"auto_clear": "auto_clear", "investigate": "investigate"})
    graph.add_edge("auto_clear", END)
    graph.add_conditional_edges("investigate", tools_condition,
                                {"tools": "tools", END: "verdict"})
    graph.add_edge("tools", "investigate")
    graph.add_conditional_edges("verdict", route_after_verdict,
                                {"human_review": "human_review", END: END})
    graph.add_edge("human_review", END)
    return graph.compile()


def triage(txn: Transaction, graph=None) -> dict:
    """Run one transaction through the pipeline and return a flat summary."""
    if graph is None:
        graph = build_graph()
    state = graph.invoke({
        "transaction": txn.model_dump(),
        "rule_result": None, "messages": [], "verdict": None,
        "escalated_to_llm": False, "human_review_queued": False,
    })
    return {
        "txn_id": txn.txn_id,
        "rule_flags": state["rule_result"]["flags"],
        "rule_score": state["rule_result"]["score"],
        "verdict": state["verdict"],
        "escalated_to_llm": state["escalated_to_llm"],
        "human_review_queued": state["human_review_queued"],
    }
