from fraud_triage.data.store import get_store
from fraud_triage.rules import run_rules


def _scenario(label: str) -> dict:
    store = get_store()
    return next(s for s in store.scenarios if s["label"] == label)


def _run(label: str):
    store = get_store()
    scenario = _scenario(label)
    history = store.get_history(scenario["customer_id"], limit=200)
    return run_rules(scenario["txn"], history)


def test_legit_transaction_scores_low():
    result = _run("legit")
    assert result.score < 0.25
    assert "impossible_travel" not in result.flags


def test_velocity_burst_is_flagged():
    result = _run("velocity_burst")
    assert "velocity" in result.flags
    assert result.score >= 0.25


def test_impossible_travel_is_flagged():
    result = _run("impossible_travel")
    assert "impossible_travel" in result.flags
    assert result.details["impossible_travel"]["implied_speed_kmh"] > 900


def test_amount_spike_is_flagged():
    result = _run("amount_spike")
    assert "amount_anomaly" in result.flags
    assert result.details["amount_anomaly"]["zscore"] >= 3.0


def test_score_is_bounded():
    for label in ("legit", "velocity_burst", "impossible_travel", "amount_spike"):
        result = _run(label)
        assert 0.0 <= result.score <= 1.0
