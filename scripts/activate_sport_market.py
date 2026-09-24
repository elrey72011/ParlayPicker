"""Owner-confirmed activation for one reviewed sport/market; never places wagers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core.prospective_evidence import deployment_state
from core.exposure_ledger import snapshot
from core.market_policy import SPORT_MARKET_FAMILIES
from core.sport_market_activation import activate_market
from core.sport_policy import SportPolicy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", required=True, choices=SPORT_MARKET_FAMILIES)
    parser.add_argument("--market-family", required=True)
    parser.add_argument("--evidence-db", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True,
                        help="Owner-reviewed JSON of SportPolicy fields for this exact market")
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--destination", type=Path,
                        help="Defaults to the exact market file read by the live gate")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    if args.market_family not in SPORT_MARKET_FAMILIES[args.sport]:
        parser.error("Market family does not belong to this sport")
    if not args.confirm:
        parser.error("Owner --confirm is required")
    destination = args.destination or Path(os.environ.get(
        "PARLAYPICKER_MARKET_ACTIVATIONS_DIR", "data/policies/active_markets")) / (
            f"{args.sport}-{args.market_family}.json")
    if destination.name != f"{args.sport}-{args.market_family}.json":
        parser.error("Destination filename must match the exact sport and market")
    state = deployment_state(args.evidence_db, args.sport, args.market_family)
    if any(token in str(state.get(key, "")).upper() for key in
           ("validation_id", "artifact_id", "model_id", "calibration_id")
           for token in ("TEST", "SYNTHETIC")):
        parser.error("Test artifacts cannot activate production markets")
    policy = SportPolicy(**json.loads(args.policy.read_text(encoding="utf-8")))
    record = activate_market(state, policy, snapshot(args.ledger),
                             owner_id=args.owner_id, expires_at=args.expires_at,
                             confirm=True)
    if destination.exists():
        parser.error("Activation record already exists; create a new reviewed record")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps({"sport": args.sport, "market_family": args.market_family,
                      "activation_hash": record["activation_hash"], "destination": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
