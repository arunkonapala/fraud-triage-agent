"""Run the four labeled scenarios through the triage agent.

Requires ANTHROPIC_API_KEY (three of the four scenarios escalate to Claude).

    python demo.py
"""

import json

from dotenv import load_dotenv

load_dotenv()

from fraud_triage.data.store import reset_store  # noqa: E402
from fraud_triage.graph import build_graph, triage  # noqa: E402


def main() -> None:
    store = reset_store()
    graph = build_graph()

    for scenario in store.scenarios:
        print(f"\n{'=' * 70}\nSCENARIO: {scenario['label']}  "
              f"(customer {scenario['customer_id']})\n{'=' * 70}")
        txn = scenario["txn"]
        print(f"  {txn.timestamp}  ${txn.amount:>8.2f}  {txn.merchant}  "
              f"[{txn.channel}]")

        result = triage(txn, graph=graph)
        verdict = result["verdict"]

        print(f"\n  rule flags:      {result['rule_flags'] or 'none'}")
        print(f"  rule score:      {result['rule_score']:.2f}")
        print(f"  escalated:       {result['escalated_to_llm']}")
        print(f"  VERDICT:         {verdict['verdict'].upper()}  "
              f"(risk {verdict['risk_score']}/100)")
        print(f"  human review:    {result['human_review_queued']}")
        print(f"  rationale:       {verdict['rationale']}")
        if verdict.get("sar_narrative"):
            print(f"\n  --- draft SAR narrative ---\n  {verdict['sar_narrative']}")

    print(f"\n{'=' * 70}\nDone.")


if __name__ == "__main__":
    main()
