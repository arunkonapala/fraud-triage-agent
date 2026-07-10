"""In-memory data store the agent tools query. In production this would be
a customer DB / transaction warehouse; here it wraps the synthetic dataset."""

from ..schemas import Transaction
from .generator import build_dataset


class BankStore:
    def __init__(self, seed: int = 42):
        data = build_dataset(seed)
        self._customers: dict = data["customers"]
        self._histories: dict[str, list[Transaction]] = data["histories"]
        self.scenarios: list[dict] = data["scenarios"]
        # Scenario "extra history" (e.g. the earlier hits of a velocity burst)
        # becomes part of the queryable ledger.
        for scenario in self.scenarios:
            self._histories[scenario["customer_id"]].extend(scenario["extra_history"])
        for history in self._histories.values():
            history.sort(key=lambda t: t.timestamp)

    def get_customer(self, customer_id: str) -> dict | None:
        return self._customers.get(customer_id)

    def get_history(self, customer_id: str, limit: int = 50) -> list[Transaction]:
        return self._histories.get(customer_id, [])[-limit:]

    def add_transaction(self, txn: Transaction) -> None:
        self._histories.setdefault(txn.customer_id, []).append(txn)


_store: BankStore | None = None


def get_store() -> BankStore:
    global _store
    if _store is None:
        _store = BankStore()
    return _store


def reset_store(seed: int = 42) -> BankStore:
    global _store
    _store = BankStore(seed)
    return _store
