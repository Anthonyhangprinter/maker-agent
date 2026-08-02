#!/usr/bin/env python3
"""M6' campaign spend tracker — ledger (list-price estimate) vs Console credit anchors.

The local ledger prices at LIST rate; Sonnet 5 actually bills at the intro rate (~2/3 of
list) so the ledger OVERSTATES Sonnet sync calls ~1.5x. Batch rows are already halved.
Console credit anchors (user-reported balances) are the ground truth; this script keeps
both in one place.

    python3 scripts/campaign_spend.py                       # status
    python3 scripts/campaign_spend.py --anchor 51.22        # record a Console balance
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path.home() / ".openclaw" / "cad-cloud-spend.jsonl"
STATE = Path.home() / ".openclaw" / "cad-campaign-spend.json"
CAMPAIGN_START = "2026-07-31T12:00:00+00:00"   # UTC — Console top-up to ~$54.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=float, help="record current Console credit balance")
    a = ap.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "campaign_start_utc": CAMPAIGN_START, "baseline_credits": 54.32, "anchors": []}
    if a.anchor is not None:
        state["anchors"].append({"ts": datetime.now(timezone.utc).isoformat(),
                                 "credits": a.anchor})
        STATE.write_text(json.dumps(state, indent=1))
        print(f"anchored: ${a.anchor:.2f}")

    rows = [json.loads(l) for l in LEDGER.read_text().splitlines()]
    camp = [r for r in rows if r["ts"] >= CAMPAIGN_START]
    tot = sum(r["usd"] for r in camp)
    batch = sum(r["usd"] for r in camp if r.get("batch"))
    print(f"campaign ledger (list-priced): ${tot:.2f} over {len(camp)} calls "
          f"(${batch:.2f} batch-rate, ${tot - batch:.2f} sync)")
    if state["anchors"]:
        last = state["anchors"][-1]
        used = state["baseline_credits"] - last["credits"]
        print(f"Console ground truth: ${used:.2f} used of the ~$50 top-up "
              f"(balance ${last['credits']:.2f} at {last['ts'][:16]})")
    print(f"budget remaining vs $50 plan: ~${50 - (state['baseline_credits'] - state['anchors'][-1]['credits']) if state['anchors'] else 50:.2f}"
          if state["anchors"] else "no Console anchor yet — pass --anchor <balance>")


if __name__ == "__main__":
    main()
