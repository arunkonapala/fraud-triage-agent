"""FastAPI wrapper: POST a transaction, get a triage verdict."""

from functools import lru_cache

from fastapi import FastAPI

from . import config
from .graph import build_graph, triage
from .schemas import Transaction, TriageResponse

app = FastAPI(
    title="Fraud Triage Agent",
    description="LangGraph + Claude agent that screens card transactions, "
                "investigates suspicious ones, and drafts SAR narratives.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": config.LLM_PROVIDER, "model": config.MODEL}


@app.post("/triage", response_model=TriageResponse)
def triage_transaction(txn: Transaction) -> TriageResponse:
    result = triage(txn, graph=get_graph())
    return TriageResponse(**result)
