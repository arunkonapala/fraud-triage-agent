"""Shared fixtures. StubLLM lets the whole graph run offline: it mimics the
two interfaces the graph uses (bind_tools -> chat model, with_structured_output
-> FraudVerdict producer) without any network calls."""

import pytest
from langchain_core.messages import AIMessage

from fraud_triage.data.store import reset_store
from fraud_triage.schemas import FraudVerdict


class _StructuredStub:
    def __init__(self, verdict: FraudVerdict):
        self._verdict = verdict

    def invoke(self, _messages):
        return self._verdict


class StubLLM:
    """Answers immediately with no tool calls, and returns a canned verdict."""

    def __init__(self, verdict: FraudVerdict | None = None):
        self.verdict = verdict or FraudVerdict(
            verdict="block",
            risk_score=92,
            key_signals=["velocity burst", "high-risk merchant"],
            rationale="Stub verdict for offline tests.",
            sar_narrative="Stub SAR narrative.",
        )
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return _StructuredStub(self.verdict)

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content="Investigation complete; issuing verdict.")


@pytest.fixture(autouse=True)
def fresh_store():
    """Each test gets a deterministic, un-mutated dataset."""
    return reset_store(seed=42)


@pytest.fixture
def stub_llm():
    return StubLLM()
