"""Tools the investigation agent can call. Each returns compact JSON so
tool results stay cheap in the context window."""

import json

from langchain_core.tools import tool

from .data.store import get_store
from .rules import (
    check_impossible_travel,
    check_velocity,
    haversine_km,
)
from .schemas import Transaction


@tool
def get_customer_profile(customer_id: str) -> str:
    """Look up the customer's profile: home city, account age, and typical
    spend. Call this first to establish the baseline."""
    customer = get_store().get_customer(customer_id)
    if not customer:
        return json.dumps({"error": f"unknown customer {customer_id}"})
    return json.dumps({
        "customer_id": customer["customer_id"],
        "home_city": customer["home_city"],
        "account_age_days": customer["account_age_days"],
        "typical_amount_usd": round(customer["avg_amount"], 2),
    })


@tool
def get_recent_transactions(customer_id: str, limit: int = 10) -> str:
    """Fetch the customer's most recent transactions (newest last). Use this
    to compare the flagged transaction against actual recent behavior."""
    history = get_store().get_history(customer_id, limit=limit)
    return json.dumps([
        {
            "txn_id": t.txn_id, "timestamp": t.timestamp,
            "amount": t.amount, "merchant": t.merchant,
            "mcc": t.mcc, "channel": t.channel,
        }
        for t in history
    ])


@tool
def check_transaction_velocity(customer_id: str, txn_json: str) -> str:
    """Count how many transactions the customer made in the last hour
    relative to the transaction under review (passed as JSON). Bursts of
    5+ suggest a stolen card being drained."""
    txn = Transaction.model_validate_json(txn_json)
    history = get_store().get_history(txn.customer_id, limit=200)
    hit, info = check_velocity(txn, history)
    return json.dumps({"velocity_flag": hit, **info})


@tool
def check_geo_feasibility(customer_id: str, txn_json: str) -> str:
    """Check whether the transaction location is physically reachable from
    the customer's previous transaction (implied travel speed), and its
    distance from their home city."""
    txn = Transaction.model_validate_json(txn_json)
    store = get_store()
    history = store.get_history(txn.customer_id, limit=200)
    hit, info = check_impossible_travel(txn, history)
    customer = store.get_customer(txn.customer_id)
    if customer:
        info["km_from_home"] = round(
            haversine_km(customer["home_lat"], customer["home_lon"], txn.lat, txn.lon), 1
        )
    return json.dumps({"impossible_travel_flag": hit, **info})


INVESTIGATION_TOOLS = [
    get_customer_profile,
    get_recent_transactions,
    check_transaction_velocity,
    check_geo_feasibility,
]
