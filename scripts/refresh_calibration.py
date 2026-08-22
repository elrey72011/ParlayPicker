#!/usr/bin/env python3
"""Bulk-refresh the calibration table + bucket stats from raw (export, recap) pairs.

The feedback loop is otherwise per-day and manual — scripts/grade_slate.py joins ONE
export to ONE recap, then fit_calibration.py / fit_bucket_stats.py refit. Because nobody
ran it, the calibration went stale at 2026-06-11 while a week of graded recaps piled up
unused. This wrapper closes that gap and makes the loop one command:

  1. Scan RAW_DIR for exports (carry ``export_run_id`` + ``best_pick``) and recaps
     (carry ``Pick Taken`` + ``Outcome``).
  2. Pair each recap to the export sharing its ``export_run_id`` (the only correct
     join — same pipeline run that produced the graded card). When several recaps map
     to one game date, keep the latest run.
  3. grade() each pair into data/backtest_exports/<game-date>.csv.
  4. Refit data/calibration/effective_prob_calibration.json and bucket_stats.json.

Usage:
    python3 scripts/refresh_calibration.py <raw_dir>
    python3 scripts/refresh_calibration.py <raw_dir> --dry-run   # grade only, no refit

Pair recaps + exports into one folder (e.g. from Drive) and point RAW_DIR at it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_slate import grade  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = ROOT / "data" / "backtest_exports"
CAL_JSON = ROOT / "data" / "calibration" / "effective_prob_calibration.json"
BUCKET_JSON = ROOT / "data" / "calibration" / "bucket_stats.json"

_SCOPED_EXPORT_PREFIXES = (
    "all_games_play_card",
    "best_picks_candidate_audit",
    "mlb_player_props_all_export",
    "player_props_all_export",
    "precision_game_card",
    "production_game_bets",
)
_REQUIRED_BEST_PICKS_COLUMNS = {"league", "Home", "Away", "best_pick"}


def _run_id(df: pd.DataFrame) -> str | None:
    if "export_run_id" not in df.columns:
        return None
    ids = df["export_run_id"].dropna().astype(str).str.strip().unique()
    return ids[0] if len(ids) == 1 else (ids[0] if len(ids) else None)


def _game_date(run_id: str) -> str | None:
    # run id 20260618T152710Z -> game/slate date 2026-06-18
    if run_id and len(run_id) >= 8 and run_id[:8].isdigit():
        return f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    return None


def _classify(raw_dir: Path):
    """Return full-card exports and recaps found under ``raw_dir``.

    Several downloadable artifacts preserve ``best_pick`` and ``export_run_id``
    for traceability.  They are not interchangeable calibration inputs: a
    production-only or precision-only export contains a selected subset, while
    a player-prop export describes a different prediction target entirely.  The
    old last-filename-wins rule silently replaced ``best_picks_export`` with
    those scoped files when they shared a run id.

    Prefer the named full-card export, reject known scoped artifacts, and use
    schema/coverage only as deterministic tie-breakers for renamed full cards.
    """
    exports: dict[str, Path] = {}
    export_ranks: dict[str, tuple[int, int, int]] = {}
    recaps: list[tuple[Path, str]] = []
    for f in sorted(raw_dir.glob("*.csv")):
        try:
            head = pd.read_csv(f, nrows=5)
        except Exception:
            continue
        cols = set(head.columns)
        rid = _run_id(head)
        if {"Pick Taken", "Outcome"} <= cols:
            recaps.append((f, rid))
            continue
        if not rid or not _REQUIRED_BEST_PICKS_COLUMNS <= cols:
            continue
        if not ({"effective_win_probability", "WinProbability"} & cols):
            continue

        normalized_name = f.stem.casefold().replace(" ", "_")
        if normalized_name.startswith(_SCOPED_EXPORT_PREFIXES):
            continue
        named_full_card = int(normalized_name.startswith("best_picks_export"))
        unique_games = int(
            head[["Home", "Away"]].fillna("").drop_duplicates().shape[0]
        )
        rank = (named_full_card, unique_games, int(len(head)))
        if rid not in exports or rank > export_ranks[rid]:
            exports[rid] = f
            export_ranks[rid] = rank
    return exports, recaps


def _load_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text()).get("meta", {})
    except Exception:
        return {}


def _run_refits(exports_dir: Path = EXPORTS_DIR) -> bool:
    """Refresh both artifacts; return whether global calibration promoted.

    ``fit_calibration.py`` uses exit code 2 for a statistically valid refusal to
    promote. That is not a pipeline failure and must not prevent bucket history
    from incorporating the newly graded slate. Any other non-zero status remains
    a real error.
    """
    print("\n=== fit_calibration.py ===")
    calibration = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fit_calibration.py"),
            str(exports_dir),
            str(CAL_JSON),
        ],
        check=False,
    )
    if calibration.returncode not in (0, 2):
        calibration.check_returncode()
    calibration_promoted = calibration.returncode == 0
    if not calibration_promoted:
        print(
            "Global calibration promotion was rejected by chronological validation; "
            "retaining the prior artifact and continuing with bucket statistics."
        )

    print("\n=== fit_bucket_stats.py ===")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fit_bucket_stats.py"),
            str(exports_dir),
            str(BUCKET_JSON),
        ],
        check=True,
    )
    return calibration_promoted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", help="folder holding export + recap CSV pairs")
    ap.add_argument("--dry-run", action="store_true", help="grade only; skip the refit")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise SystemExit(f"not a directory: {raw_dir}")

    exports, recaps = _classify(raw_dir)
    print(f"found {len(exports)} export(s), {len(recaps)} recap(s) in {raw_dir}\n")

    # Pair by run id; when several recaps share a game date keep the latest run id.
    by_date: dict[str, tuple[str, Path, Path]] = {}
    for recap_path, rid in recaps:
        if not rid or rid not in exports:
            print(f"  [SKIP] {recap_path.name}: no export with run id {rid!r} in raw_dir")
            continue
        date = _game_date(rid)
        if not date:
            print(f"  [SKIP] {recap_path.name}: cannot derive date from run id {rid!r}")
            continue
        if date not in by_date or rid > by_date[date][0]:
            by_date[date] = (rid, exports[rid], recap_path)

    if not by_date:
        raise SystemExit("no (export, recap) pairs matched by run id — nothing to grade")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    graded = 0
    for date in sorted(by_date):
        rid, exp_path, rec_path = by_date[date]
        out = EXPORTS_DIR / f"{date}.csv"
        print(f"grading {date}  (run {rid})  {exp_path.name}  x  {rec_path.name}")
        grade(exp_path, rec_path, out)
        try:
            graded += len(pd.read_csv(out))
        except Exception:
            pass

    print(f"\ngraded {len(by_date)} slate(s), {graded} rows into {EXPORTS_DIR}")

    if args.dry_run:
        print("--dry-run: skipping refit")
        return 0

    before_cal, before_bkt = _load_meta(CAL_JSON), _load_meta(BUCKET_JSON)
    calibration_promoted = _run_refits(EXPORTS_DIR)

    after_cal, after_bkt = _load_meta(CAL_JSON), _load_meta(BUCKET_JSON)
    print("\n=== refit summary ===")
    print(f"calibration  n_graded: {before_cal.get('n_graded')} -> {after_cal.get('n_graded')}"
          f"   fitted_on: {before_cal.get('fitted_on')} -> {after_cal.get('fitted_on')}")
    print(f"bucket_stats fitted_on: {before_bkt.get('fitted_on')} -> {after_bkt.get('fitted_on')}")
    if not calibration_promoted:
        print("calibration  promotion: BLOCKED (production falls back to upstream probabilities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
