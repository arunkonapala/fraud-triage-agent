"""Run the four labeled scenarios through the triage agent.

Requires ANTHROPIC_API_KEY (three of the four scenarios escalate to Claude).

    python demo.py
"""

import json

from dotenv import load_dotenv

load_dotenv()

from fraud_triage.data.store import reset_store  # noqa: E402
from fraud_triage.graph import build_graph, triage  # noqa: E402


def _setup_tracing():
    """Optional observability via agentobs (github.com/arunkonapala/agent-observability).
    Activates when AGENTOBS_EXPORTER is set (console | otlp | memory)."""
    import os

    if not os.getenv("AGENTOBS_EXPORTER"):
        return None, None, None
    try:
        from agentobs import agent_turn, init_tracing
        from agentobs.integrations.langchain import OTelCallbackHandler
        from agentobs.tracing import flush
    except ImportError:
        print("AGENTOBS_EXPORTER set but agentobs not installed — tracing off.")
        return None, None, None
    init_tracing("fraud-triage-agent")
    return agent_turn, [OTelCallbackHandler()], flush


def main() -> None:
    store = reset_store()
    graph = build_graph()
    agent_turn, callbacks, flush = _setup_tracing()

    for scenario in store.scenarios:
        print(f"\n{'=' * 70}\nSCENARIO: {scenario['label']}  "
              f"(customer {scenario['customer_id']})\n{'=' * 70}")
        txn = scenario["txn"]
        print(f"  {txn.timestamp}  ${txn.amount:>8.2f}  {txn.merchant}  "
              f"[{txn.channel}]")

        if agent_turn is not None:
            with agent_turn(scenario["label"], customer_id=scenario["customer_id"]):
                result = triage(txn, graph=graph, callbacks=callbacks)
        else:
            result = triage(txn, graph=graph)
        verdict = result["verdict"]

        print(f"\n  rule flags:      {result['rule_flags'] or 'none'}")
        print(f"  rule score:      {result['rule_score']:.2f}")
        print(f"  escalated:       {result['escalated_to_llm']}")
        print(f"  VERDICT:         {verdict['verdict'].upper()}  "
              f"(risk {verdict['risk_score']}/100)")
        print(f"  human review:    {result['human_review_queued']}")
        print(f"  rationale:       {verdict['rationale']}")
        if verdict.get("sar_narrative"):
            print(f"\n  --- draft SAR narrative ---\n  {verdict['sar_narrative']}")

    if flush is not None:
        flush()
    print(f"\n{'=' * 70}\nDone.")


if __name__ == "__main__":
    main()
