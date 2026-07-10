from fraud_triage.data.generator import build_dataset


def test_dataset_is_deterministic():
    a, b = build_dataset(seed=42), build_dataset(seed=42)
    txn_a = a["scenarios"][0]["txn"]
    txn_b = b["scenarios"][0]["txn"]
    assert txn_a.amount == txn_b.amount
    assert txn_a.merchant == txn_b.merchant


def test_all_scenarios_present():
    data = build_dataset()
    labels = {s["label"] for s in data["scenarios"]}
    assert labels == {"legit", "velocity_burst", "impossible_travel", "amount_spike"}


def test_customers_have_history():
    data = build_dataset()
    for customer_id, history in data["histories"].items():
        assert len(history) > 20, f"{customer_id} has too little history"
        # sorted ascending by timestamp
        timestamps = [t.timestamp for t in history]
        assert timestamps == sorted(timestamps)
