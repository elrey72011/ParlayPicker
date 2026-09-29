"""Build private current-wager diagnostics from saved local evidence only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app_core.current_wagers_trace import build_private_candidate_trace
from app_core.release_preflight import evaluate_release


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    check=True, capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def write_artifacts(candidates_path: Path | None, package_path: Path, output: Path, *,
                    evaluated_at: str | None = None, current_at: str | None = None,
                    published_at: str | None = None,
                    publication_version_path: Path | None = None) -> dict:
    package = json.loads(package_path.read_text(encoding="utf-8-sig"))
    candidates = pd.read_csv(candidates_path) if candidates_path else None
    report = build_private_candidate_trace(
        candidates, package, evaluated_at=evaluated_at, current_at=current_at,
        selection_options={"college_fallback": True, "nfl_fallback": True,
                           "research_fallback": True},
    )
    preflight = evaluate_release(package, at=current_at, published_at=published_at)
    sha, dirty = _revision()
    publication_version = (json.loads(publication_version_path.read_text(encoding="utf-8-sig"))
                           if publication_version_path else None)
    selected = [row for row in package.get("games", {}).get("overall", [])
                if isinstance(row, dict)]
    saved_status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in selected)
    saved_qualification_counts = Counter(
        str(row.get("qualification_reason") or "UNKNOWN") for row in selected
    )
    negative_ev_count = sum(
        1 for row in selected
        if isinstance(row.get("ev"), (int, float)) and not isinstance(row.get("ev"), bool)
        and row["ev"] < 0
    )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "source-runtime-publication-manifest.json": {
            "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_git_sha": sha, "source_git_dirty": dirty,
            "package_reference": package_path.name, "package_sha256": _digest(package_path),
            "candidate_reference": candidates_path.name if candidates_path else None,
            "candidate_sha256": _digest(candidates_path) if candidates_path else None,
            "publication_version": publication_version,
            "publication_version_sha256": _digest(publication_version_path) if publication_version_path else None,
            "historical_reconstruction_complete": candidates_path is not None,
        },
        "current-wagers-candidate-funnel.json": report["funnel"],
        "current-wagers-original-vs-current-blockers.json": {
            "primary_blocker_counts": report["primary_blocker_counts"],
            "overlapping_blocker_counts": report["overlapping_blocker_counts"],
            "selected_overall_denominator": len(selected),
            "selected_overall_saved_status_counts": dict(sorted(saved_status_counts.items())),
            "selected_overall_saved_qualification_counts": dict(sorted(saved_qualification_counts.items())),
            "selected_overall_negative_ev_count": negative_ev_count,
            "release_preflight": preflight,
        },
        "current-wagers-release-timeline.json": {
            "evaluated_at": preflight["evaluated_at"], "published_at": preflight["published_at"],
            "package_built_at": preflight["package_built_at"],
            "rows": [{key: row.get(key) for key in (
                "row_id", "section", "saved_status", "quote_age_seconds",
                "analysis_age_seconds", "quote_age_at_publication_seconds",
                "analysis_age_at_publication_seconds", "expires_at",
                "remaining_validity_seconds",
            )} for row in preflight["rows"]],
        },
        "current-wagers-preflight-report.json": preflight,
        "current-wagers-output-parity.json": {
            "candidate_count": report["all_market_candidate_count"],
            "packaged_candidate_ids": report.get("funnel", {}).get("packaged_output", {}).get("candidate_ids", []),
            "currently_usable_candidate_ids": report.get("funnel", {}).get("currently_usable_wager", {}).get("candidate_ids", []),
        },
        "current-wagers-readiness-dependencies.json": {
            "model_qualification": "UNKNOWN_FROM_LOCAL_REPLAY",
            "authenticated_market_census": "NOT_EXECUTED_BY_THIS_LOCAL_REPLAY",
            "staging_hosted_verification": "NOT_EXECUTED_BY_THIS_LOCAL_REPLAY",
            "provider_refresh": "NOT_EXECUTED_BY_THIS_LOCAL_REPLAY",
            "production_authority_created": False,
        },
    }
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "current-wagers-candidate-trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in report["candidates"]),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluated-at")
    parser.add_argument("--current-at")
    parser.add_argument("--published-at")
    parser.add_argument("--publication-version", type=Path)
    args = parser.parse_args(argv)
    report = write_artifacts(args.candidates, args.package, args.output,
                             evaluated_at=args.evaluated_at, current_at=args.current_at,
                             published_at=args.published_at,
                             publication_version_path=args.publication_version)
    print(json.dumps({
        "status": report["trace_status"],
        "candidate_count": report["all_market_candidate_count"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
