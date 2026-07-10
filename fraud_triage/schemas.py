"""Pydantic schemas shared by the rules engine, graph, and API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    txn_id: str
    customer_id: str
    timestamp: str  # ISO 8601
    amount: float
    currency: str = "USD"
    merchant: str
    mcc: str  # merchant category code
    lat: float
    lon: float
    channel: Literal["card_present", "card_not_present"] = "card_present"


class RuleResult(BaseModel):
    flags: list[str] = Field(default_factory=list)
    score: float = 0.0  # 0.0 (benign) .. 1.0 (maximally suspicious)
    details: dict = Field(default_factory=dict)


class FraudVerdict(BaseModel):
    """Structured verdict produced by the agent after investigation."""

    verdict: Literal["clear", "review", "block"] = Field(
        description=(
            "clear = release the transaction; review = queue for a human "
            "analyst; block = decline and lock the card pending contact"
        )
    )
    risk_score: int = Field(ge=0, le=100, description="Overall fraud risk, 0-100")
    key_signals: list[str] = Field(
        description="The specific evidence items that drove the verdict"
    )
    rationale: str = Field(
        description="Plain-English explanation an analyst can act on"
    )
    sar_narrative: Optional[str] = Field(
        default=None,
        description=(
            "For review/block verdicts: a draft Suspicious Activity Report "
            "narrative (who, what, when, where, why suspicious)"
        ),
    )


class TriageResponse(BaseModel):
    txn_id: str
    rule_flags: list[str]
    rule_score: float
    verdict: FraudVerdict
    escalated_to_llm: bool
    human_review_queued: bool
