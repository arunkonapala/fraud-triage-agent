"""Central configuration for the fraud triage agent."""

import os

# Provider for the investigation and verdict nodes: "anthropic" or "groq".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "groq": "llama-3.3-70b-versatile",
}

# Opus 4.8 rejects temperature/top_p/top_k with a 400 — never set them.
MODEL = os.getenv("FRAUD_TRIAGE_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "claude-opus-4-8"))
MAX_TOKENS = int(os.getenv("FRAUD_TRIAGE_MAX_TOKENS", "4096"))

# Transactions scoring below this on the deterministic rules engine are
# auto-cleared without an LLM call (cost control: the model only sees
# transactions with at least one meaningful signal).
AUTO_CLEAR_THRESHOLD = 0.25

# Verdict risk scores in this band are routed to the human review queue.
REVIEW_BAND = (40, 75)

# Rules engine knobs.
VELOCITY_WINDOW_MINUTES = 60
VELOCITY_MAX_TXNS = 4
AMOUNT_ZSCORE_FLAG = 3.0
IMPOSSIBLE_TRAVEL_KMH = 900.0
HIGH_RISK_MCCS = {
    "4829",  # wire transfer / money orders
    "6051",  # quasi-cash: crypto, foreign currency
    "5967",  # direct marketing - inbound teleservices
    "7995",  # gambling
    "6211",  # securities brokers
}
