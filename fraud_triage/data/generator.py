"""Seeded synthetic bank dataset: customers, transaction history, and
injected fraud scenarios with ground-truth labels for eval."""

import random
import uuid
from datetime import datetime, timedelta, timezone

from ..schemas import Transaction

CITIES = {
    "austin": (30.2672, -97.7431),
    "nyc": (40.7128, -74.0060),
    "chicago": (41.8781, -87.6298),
    "seattle": (47.6062, -122.3321),
    "miami": (25.7617, -80.1918),
    "lagos": (6.5244, 3.3792),
    "bucharest": (44.4268, 26.1025),
}

NORMAL_MERCHANTS = [
    ("HEB Grocery", "5411"),
    ("Shell Oil", "5541"),
    ("Chipotle", "5814"),
    ("Amazon.com", "5942"),
    ("Netflix", "4899"),
    ("CVS Pharmacy", "5912"),
    ("Home Depot", "5211"),
]

FRAUD_MERCHANTS = [
    ("CoinFlash Exchange", "6051"),
    ("LuxWatch Online", "5944"),
    ("QuickWire Intl", "4829"),
    ("BetRoyale Casino", "7995"),
]


def _txn(customer_id: str, ts: datetime, amount: float, merchant: str,
         mcc: str, lat: float, lon: float, channel: str = "card_present") -> Transaction:
    return Transaction(
        txn_id=f"txn_{uuid.uuid4().hex[:12]}",
        customer_id=customer_id,
        timestamp=ts.replace(microsecond=0).isoformat(),
        amount=round(amount, 2),
        merchant=merchant,
        mcc=mcc,
        lat=round(lat, 4),
        lon=round(lon, 4),
        channel=channel,  # type: ignore[arg-type]
    )


def make_customer(rng: random.Random, customer_id: str, home_city: str) -> dict:
    lat, lon = CITIES[home_city]
    return {
        "customer_id": customer_id,
        "name": f"Customer {customer_id[-4:].upper()}",
        "home_city": home_city,
        "home_lat": lat,
        "home_lon": lon,
        "avg_amount": rng.uniform(30, 120),
        "account_age_days": rng.randint(200, 3000),
    }


def make_history(rng: random.Random, customer: dict, days: int = 60) -> list[Transaction]:
    """Benign spending pattern near the customer's home city."""
    txns = []
    now = datetime.now(timezone.utc)
    for day in range(days, 0, -1):
        for _ in range(rng.randint(0, 3)):
            merchant, mcc = rng.choice(NORMAL_MERCHANTS)
            ts = now - timedelta(days=day, hours=rng.uniform(8, 21))
            amount = max(3.0, rng.gauss(customer["avg_amount"], customer["avg_amount"] * 0.4))
            jitter = 0.05
            txns.append(_txn(
                customer["customer_id"], ts, amount, merchant, mcc,
                customer["home_lat"] + rng.uniform(-jitter, jitter),
                customer["home_lon"] + rng.uniform(-jitter, jitter),
            ))
    txns.sort(key=lambda t: t.timestamp)
    return txns


def scenario_legit(rng: random.Random, customer: dict) -> Transaction:
    """An ordinary purchase consistent with the customer's pattern."""
    merchant, mcc = rng.choice(NORMAL_MERCHANTS)
    return _txn(
        customer["customer_id"], datetime.now(timezone.utc),
        max(3.0, rng.gauss(customer["avg_amount"], 10)),
        merchant, mcc, customer["home_lat"], customer["home_lon"],
    )


def scenario_velocity_burst(rng: random.Random, customer: dict) -> list[Transaction]:
    """Stolen-card pattern: rapid card-not-present hits at risky merchants.

    Returns the burst; the last element is the transaction under triage.
    """
    now = datetime.now(timezone.utc)
    burst = []
    for i in range(5):
        merchant, mcc = rng.choice(FRAUD_MERCHANTS)
        burst.append(_txn(
            customer["customer_id"], now - timedelta(minutes=45 - i * 9),
            rng.uniform(180, 950), merchant, mcc,
            customer["home_lat"], customer["home_lon"],
            channel="card_not_present",
        ))
    return burst


def scenario_impossible_travel(rng: random.Random, customer: dict) -> list[Transaction]:
    """Card-present purchase at home, then another on a different continent
    two hours later. Returns [home_txn, foreign_txn]; the second is under triage.
    """
    now = datetime.now(timezone.utc)
    home = scenario_legit(rng, customer)
    home = home.model_copy(update={
        "timestamp": (now - timedelta(hours=2)).replace(microsecond=0).isoformat()
    })
    foreign_city = rng.choice(["lagos", "bucharest"])
    lat, lon = CITIES[foreign_city]
    merchant, mcc = rng.choice(FRAUD_MERCHANTS)
    foreign = _txn(
        customer["customer_id"], now, rng.uniform(400, 1500),
        merchant, mcc, lat, lon,
    )
    return [home, foreign]


def scenario_amount_spike(rng: random.Random, customer: dict) -> Transaction:
    """Single purchase an order of magnitude above the customer's baseline
    at a high-risk merchant."""
    merchant, mcc = rng.choice(FRAUD_MERCHANTS)
    return _txn(
        customer["customer_id"], datetime.now(timezone.utc),
        customer["avg_amount"] * rng.uniform(15, 30),
        merchant, mcc, customer["home_lat"], customer["home_lon"],
        channel="card_not_present",
    )


def build_dataset(seed: int = 42) -> dict:
    """Deterministic dataset: 4 customers with history plus labeled scenarios."""
    rng = random.Random(seed)
    customers, histories, scenarios = {}, {}, []

    specs = [
        ("cust_alice01", "austin"), ("cust_bob0002", "nyc"),
        ("cust_carol03", "chicago"), ("cust_dave004", "seattle"),
    ]
    for cid, city in specs:
        customer = make_customer(rng, cid, city)
        customers[cid] = customer
        histories[cid] = make_history(rng, customer)

    # (label, customer, extra history to prepend, transaction under triage)
    legit = scenario_legit(rng, customers["cust_alice01"])
    scenarios.append({"label": "legit", "customer_id": "cust_alice01",
                      "extra_history": [], "txn": legit})

    burst = scenario_velocity_burst(rng, customers["cust_bob0002"])
    scenarios.append({"label": "velocity_burst", "customer_id": "cust_bob0002",
                      "extra_history": burst[:-1], "txn": burst[-1]})

    travel = scenario_impossible_travel(rng, customers["cust_carol03"])
    scenarios.append({"label": "impossible_travel", "customer_id": "cust_carol03",
                      "extra_history": travel[:-1], "txn": travel[-1]})

    spike = scenario_amount_spike(rng, customers["cust_dave004"])
    scenarios.append({"label": "amount_spike", "customer_id": "cust_dave004",
                      "extra_history": [], "txn": spike})

    return {"customers": customers, "histories": histories, "scenarios": scenarios}
