"""Deterministic rules engine. GenAI is not the classifier here — these
cheap, auditable checks produce the signals; Claude reasons over them only
when a transaction actually looks suspicious."""

import math
import statistics
from datetime import datetime

from . import config
from .schemas import RuleResult, Transaction


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _ts(txn: Transaction) -> datetime:
    return datetime.fromisoformat(txn.timestamp)


def check_amount_anomaly(txn: Transaction, history: list[Transaction]) -> tuple[bool, dict]:
    amounts = [t.amount for t in history]
    if len(amounts) < 5:
        return False, {"reason": "insufficient history"}
    mean = statistics.mean(amounts)
    stdev = statistics.pstdev(amounts) or 1.0
    z = (txn.amount - mean) / stdev
    return z >= config.AMOUNT_ZSCORE_FLAG, {
        "zscore": round(z, 2), "baseline_mean": round(mean, 2),
    }


def check_velocity(txn: Transaction, history: list[Transaction]) -> tuple[bool, dict]:
    window_start = _ts(txn).timestamp() - config.VELOCITY_WINDOW_MINUTES * 60
    recent = [
        t for t in history
        if _ts(t).timestamp() >= window_start and t.txn_id != txn.txn_id
    ]
    count = len(recent) + 1  # include the transaction under review
    return count > config.VELOCITY_MAX_TXNS, {
        "txns_in_window": count,
        "window_minutes": config.VELOCITY_WINDOW_MINUTES,
    }


def check_impossible_travel(txn: Transaction, history: list[Transaction]) -> tuple[bool, dict]:
    prior = [t for t in history if t.timestamp < txn.timestamp and t.txn_id != txn.txn_id]
    if not prior:
        return False, {"reason": "no prior transaction"}
    last = prior[-1]
    km = haversine_km(last.lat, last.lon, txn.lat, txn.lon)
    hours = max((_ts(txn) - _ts(last)).total_seconds() / 3600, 1 / 60)
    speed = km / hours
    return speed > config.IMPOSSIBLE_TRAVEL_KMH, {
        "distance_km": round(km, 1),
        "hours_since_last": round(hours, 2),
        "implied_speed_kmh": round(speed, 1),
        "previous_txn": last.txn_id,
    }


def check_merchant_risk(txn: Transaction, history: list[Transaction]) -> tuple[bool, dict]:
    high_risk_mcc = txn.mcc in config.HIGH_RISK_MCCS
    seen_before = any(t.merchant == txn.merchant for t in history if t.txn_id != txn.txn_id)
    return high_risk_mcc and not seen_before, {
        "high_risk_mcc": high_risk_mcc, "new_merchant": not seen_before,
    }


# (name, check function, weight toward the composite score)
CHECKS = [
    ("amount_anomaly", check_amount_anomaly, 0.30),
    ("velocity", check_velocity, 0.30),
    ("impossible_travel", check_impossible_travel, 0.40),
    ("high_risk_new_merchant", check_merchant_risk, 0.20),
]


def run_rules(txn: Transaction, history: list[Transaction]) -> RuleResult:
    flags, details, score = [], {}, 0.0
    for name, check, weight in CHECKS:
        hit, info = check(txn, history)
        details[name] = info
        if hit:
            flags.append(name)
            score += weight
    if txn.channel == "card_not_present" and flags:
        score += 0.10  # CNP amplifies any other signal
        details["card_not_present_amplifier"] = True
    return RuleResult(flags=flags, score=min(score, 1.0), details=details)
