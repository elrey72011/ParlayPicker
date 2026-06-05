#!/usr/bin/env python3
"""
Empty-card frequency tracker.

Scans a folder of ``best_picks_export*.csv`` files and reports, per slate, whether
the production card was empty (zero Actionable picks) and why, plus the aggregate
empty-card rate and a breakdown of the dominant block reasons.

Why this exists
---------------
The 0.65 MLB-over actionable gate (and the other anti-overconfidence guards) can
downgrade a whole slate to High Variance / Below Threshold / No Play, producing an
empty production card. That is by design for an overconfident-Over slate — but if it
happens *often*, the gate may be too strict. Per the tuning protocol, that decision
should be made on several slates of evidence, not a single reactive look. This script
turns a folder of exports into that evidence.

It does NOT grade outcomes (use scripts/grade_slate.py + backtest_*.py for W/L). It
only measures how frequently the card comes back empty and which guard is responsible.

Usage
-----
    python3 scripts/empty_card_frequency.py <dir_of_exports>

Drop your saved ``best_picks_export - *.csv`` files into one folder and point at it.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def _first(series: pd.Series, default=None):
    """First non-null value of a column, or default if absent/empty."""
    if series is None or len(series) == 0:
        return default
    nonnull = series.dropna()
    return nonnull.iloc[0] if len(nonnull) else default


def _as_bool(v: object) -> bool:
    return str(v).strip().lower() in {"true", "1", "1.0", "yes"}


def _slate_id(df: pd.DataFrame, path: Path) -> str:
    for col in ("export_run_id", "Local Date", "Commence (Local)"):
        if col in df.columns:
            val = _first(df[col])
            if val is not None and str(val).strip():
                return str(val).strip()
    return path.stem


def analyze_export(path: Path) -> dict:
    df = pd.read_csv(path)
    n = len(df)

    status = df["Pick_Status"].astype(str).str.strip() if "Pick_Status" in df.columns else pd.Series([], dtype=str)
    actionable = int((status == "Actionable").sum())

    # Prefer the pipeline's own count when present; fall back to the row tally.
    final_actionable = _first(df.get("final_actionable_count"))
    if final_actionable is not None and not pd.isna(final_actionable):
        actionable = int(float(final_actionable))

    flag = _first(df.get("production_card_empty_flag"))
    is_empty = _as_bool(flag) if flag is not None else (actionable == 0)

    # Why empty: the most common blocker stage among the non-Actionable rows.
    reason = ""
    if "status_blocker_stage" in df.columns:
        stages = [
            str(s).strip()
            for s, st in zip(df["status_blocker_stage"], status)
            if st != "Actionable" and str(s).strip()
        ]
        if stages:
            reason = Counter(stages).most_common(1)[0][0]
    if not reason:
        reason = str(_first(df.get("production_card_empty_reason")) or "").strip()

    return {
        "slate": _slate_id(df, path),
        "file": path.name,
        "picks": n,
        "actionable": actionable,
        "empty": is_empty,
        "build": str(_first(df.get("pipeline_build")) or "").strip(),
        "dominant_block": reason or "(none)",
    }


def run(directory: Path) -> int:
    files = sorted(p for p in directory.glob("*.csv") if "best_picks_export" in p.name.lower())
    if not files:
        # Fall back to every CSV in the folder so this still works on ad-hoc dirs.
        files = sorted(directory.glob("*.csv"))
    if not files:
        print(f"No CSV exports found in {directory}")
        return 1

    rows = []
    for path in files:
        try:
            rows.append(analyze_export(path))
        except Exception as exc:  # one bad file shouldn't kill the report
            print(f"  ! skipped {path.name}: {exc}")

    if not rows:
        return 1

    print("\n" + "=" * 96)
    print("Empty-card frequency")
    print("=" * 96)
    print(f"{'slate':28} {'picks':>5} {'actionable':>10} {'empty':>6}  dominant block (if empty)")
    print("-" * 96)
    for r in rows:
        block = r["dominant_block"] if r["empty"] else ""
        print(f"{r['slate'][:28]:28} {r['picks']:>5} {r['actionable']:>10} {str(r['empty']):>6}  {block[:40]}")

    n = len(rows)
    empties = [r for r in rows if r["empty"]]
    e = len(empties)
    print("-" * 96)
    print(f"Slates: {n}  |  empty cards: {e}  ({(100.0 * e / n):.0f}%)  |  with actionable picks: {n - e}")

    if empties:
        print("\n-- empty-card cause breakdown --")
        for stage, cnt in Counter(r["dominant_block"] for r in empties).most_common():
            print(f"  {cnt:>3}  {stage}")

    builds = {r["build"] for r in rows if r["build"]}
    if len(builds) > 1:
        print(f"\nNOTE: exports span multiple pipeline builds {sorted(builds)} — empty-card")
        print("  rate mixes code versions; filter to one build before drawing tuning conclusions.")

    print("\nCAVEAT: an empty card is the correct, conservative outcome on an overconfident-Over")
    print("  slate. Only treat a persistently high empty-card rate as a reason to revisit the")
    print("  0.65 gate, and pair it with graded W/L (scripts/grade_slate.py) before changing it.")
    return 0


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/backtest_exports")
    if not path.exists():
        print(f"Path not found: {path}")
        return 1
    return run(path)


if __name__ == "__main__":
    raise SystemExit(main())
