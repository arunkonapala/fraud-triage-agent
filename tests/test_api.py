from fastapi.testclient import TestClient

from fraud_triage import api
from fraud_triage.data.store import get_store
from fraud_triage.graph import build_graph


def test_health():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_triage_endpoint_with_stub_graph(stub_llm, monkeypatch):
    stub_graph = build_graph(llm=stub_llm)
    monkeypatch.setattr(api, "get_graph", lambda: stub_graph)
    client = TestClient(api.app)

    scenario = next(s for s in get_store().scenarios if s["label"] == "velocity_burst")
    response = client.post("/triage", json=scenario["txn"].model_dump())

    assert response.status_code == 200
    body = response.json()
    assert body["escalated_to_llm"] is True
    assert body["verdict"]["verdict"] == "block"
    assert 0 <= body["verdict"]["risk_score"] <= 100
