from fraud_triage.data.store import get_store
from fraud_triage.graph import build_graph, triage


def _scenario(label: str) -> dict:
    return next(s for s in get_store().scenarios if s["label"] == label)


def test_legit_transaction_auto_clears_without_llm(stub_llm):
    graph = build_graph(llm=stub_llm)
    result = triage(_scenario("legit")["txn"], graph=graph)
    assert result["verdict"]["verdict"] == "clear"
    assert result["escalated_to_llm"] is False
    assert stub_llm.calls == 0  # the LLM was never invoked


def test_velocity_burst_escalates_to_llm(stub_llm):
    graph = build_graph(llm=stub_llm)
    result = triage(_scenario("velocity_burst")["txn"], graph=graph)
    assert result["escalated_to_llm"] is True
    assert stub_llm.calls >= 1
    assert result["verdict"]["verdict"] == "block"
    assert result["verdict"]["sar_narrative"]


def test_impossible_travel_escalates_to_llm(stub_llm):
    graph = build_graph(llm=stub_llm)
    result = triage(_scenario("impossible_travel")["txn"], graph=graph)
    assert result["escalated_to_llm"] is True
    assert "impossible_travel" in result["rule_flags"]


def test_review_verdict_routes_to_human_queue(stub_llm):
    stub_llm.verdict = stub_llm.verdict.model_copy(
        update={"verdict": "review", "risk_score": 55}
    )
    graph = build_graph(llm=stub_llm)
    result = triage(_scenario("amount_spike")["txn"], graph=graph)
    assert result["human_review_queued"] is True
