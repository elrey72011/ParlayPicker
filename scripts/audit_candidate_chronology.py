"""Reclassify a saved candidate audit without creating current picks or wagers.

The original CSV is input evidence.  This tool writes a new diagnostic CSV and
summary; it never edits a saved candidate or production store in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_core import candidate_chronology as chronology


def export_time(frame: pd.DataFrame) -> pd.Timestamp:
    ids = frame.get("export_run_id", pd.Series(dtype="string")).dropna().astype(str).unique()
    if len(ids) != 1:
        raise ValueError("ONE_EXPORT_RUN_ID_REQUIRED")
    observed = pd.to_datetime(ids[0], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    if pd.isna(observed):
        raise ValueError("INVALID_EXPORT_RUN_TIME")
    return observed


def reclassify(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    as_of = export_time(frame)
    out = chronology.annotate(frame, as_of=as_of)
    current = out["candidate_context"].eq("CURRENT_PREGAME") & out["pregame_quote_valid"]
    for field in ("best_available_selected", "best_available_finalist", "final_pick_valid", "wager_approved"):
        if field not in out:
            out[field] = False
        out.loc[~current, field] = False
    if "final_pick_valid_reason" not in out:
        out["final_pick_valid_reason"] = "not_selected"
    if "best_available_rejection_reason" not in out:
        out["best_available_rejection_reason"] = "not_selected"
    post = out["candidate_context"].eq("POST_START_DIAGNOSTIC")
    out.loc[post, ["final_pick_valid_reason", "best_available_rejection_reason"]] = "post_start_quote"
    historical = out["candidate_context"].eq("HISTORICAL_BACKTEST")
    out.loc[historical, ["final_pick_valid_reason", "best_available_rejection_reason"]] = "historical_backtest"
    # A researched or unvalidated model never gains authority from positive EV.
    unvalidated = ~out["production_model_eligible"] | ~out["market_validation_status"].eq("VALIDATED")
    out.loc[unvalidated, "wager_approved"] = False
    chronology.assert_integrity(out)
    contradictions = chronology.contradictions(out)
    ev = pd.to_numeric(out.get("expected_value"), errors="coerce")
    summary = {
        "source_run_id": str(frame["export_run_id"].iloc[0]),
        "as_of_utc": as_of.isoformat(),
        "total_rows": len(out),
        "current_pregame_rows": int(current.sum()),
        "historical_backtest_rows": int(historical.sum()),
        "post_start_diagnostic_rows": int(post.sum()),
        "positive_ev_research_rows": int((ev.gt(0) & unvalidated).sum()),
        "wager_approved_rows": int(out["wager_approved"].map(chronology._bool).sum()),
        "chronology_contradictions": contradictions["chronology"],
        "semantic_contradictions": contradictions["semantics"],
    }
    return out, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    corrected, summary = reclassify(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(args.output, index=False)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
