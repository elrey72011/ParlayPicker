"""Audit, then record, the independent deployment review of a validated market (roadmap D11 option B).

    python scripts/deployment_review.py audit  --directory EMPTY_DIR --output OUT.json --plan PLAN_ID [--cache-dir DIR]
    python scripts/deployment_review.py record --directory EMPTY_DIR --output OUT.json --plan PLAN_ID
                                               --audit-hash SHA256 --reviewer NAME [--cache-dir DIR]

``audit`` restores the canonical store with every Drive upload refused and writes the
audit (``app_core.deployment_audit``) and its hash. It changes nothing.

``record`` is the owner's step after reading an audit. It restores the store, recomputes
the audit, and continues only if it hashes to ``--audit-hash`` and has no FAIL. It then
creates the validation artifact and records the deployment review with the reviewer bound
to that audit hash, and backs the new rows up to Drive with read-back. A review grants no
stake: funding still needs the owner's market activation (``core.sport_market_activation``).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_core import deployment_audit  # noqa: E402
from app_core import prospective_evidence as evidence  # noqa: E402
from bound_plan import _drive, _empty  # noqa: E402


def record(path, plan_id, audit_hash, reviewer):
    """Recompute, verify the reviewed hash, then write the artifact and the review."""
    result = deployment_audit.audit(path, plan_id)
    if result["audit_hash"] != audit_hash:
        raise SystemExit("the audit changed since it was reviewed; run audit again and review the new report")
    if result["status"] == "FAIL":
        raise SystemExit("the audit has a FAIL check; a review cannot be recorded")
    plan = evidence.load_record(path, "prospective_validation_plan", plan_id)
    criteria = json.loads(plan["payload"]) if isinstance(plan.get("payload"), str) else plan
    target = (criteria.get("deployment_criteria") or {}).get("target_state")
    suffix = audit_hash[:16]
    artifact_id = evidence.create_validation_artifact(path, plan_id, f"artifact:{plan_id}:{suffix}")
    review_id = evidence.record_deployment_review(path, {
        "deployment_id": f"review:{plan_id}:{suffix}", "artifact_id": artifact_id,
        "sport": plan["sport"], "market_family": plan["market_family"],
        "validation_id": f"validation:{plan_id}:{suffix}", "deployment_state": target,
        "reviewer_id": deployment_audit.reviewer_label(reviewer, audit_hash)})
    return dict(result, artifact_id=artifact_id, deployment_id=review_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("audit", "record"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", required=True, help="validation plan ID")
    parser.add_argument("--audit-hash")
    parser.add_argument("--reviewer")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    path = _empty(args.directory)
    client, folder = _drive(args.cache_dir)
    from app_core.prospective_remote import sync
    if args.action == "audit":
        from market_readiness_report import ReadOnlyDrive
        guarded = ReadOnlyDrive(client, args.cache_dir)
        restore = sync(path, guarded, folder, {})
        result = dict(deployment_audit.audit(path, args.plan), mode="AUDIT_READ_ONLY",
                      restore=dict(restore, writes_refused=guarded.writes_refused))
    else:
        if not (args.audit_hash and args.reviewer):
            raise SystemExit("record needs --audit-hash from a reviewed audit and --reviewer")
        session = {}
        sync(path, client, folder, session)
        result = dict(record(path, args.plan, args.audit_hash.strip(), args.reviewer), mode="RECORDED_AFTER_REVIEW")
        result["backup"] = sync(path, client, folder, session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = deployment_audit.markdown(result)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
