#!/usr/bin/env python3
"""
Grade a slate: merge a best-picks export with its performance recap into a
backtest_exports CSV that scripts/backtest_*.py can read.

Replaces the error-prone hand-merge. Given:
  * the best-picks export (consensus_agreement, kalshi_probability, Kelly_Bet_Size,
    odds_american, market_type, WinProbability/effective_*, Home/Away/best_pick), and
  * the performance recap (Home, Away, Pick Taken, Outcome),
it matches rows on (Home, Away, best_pick), grades W/L from Outcome, computes
Win Amount = Kelly_Bet_Size * decimal_odds for wins, and writes the graded CSV.

Usage
-----
    python3 scripts/grade_slate.py <export.csv> <recap.csv> data/backtest_exports/2026-06-03.csv

Then re-run:  python3 scripts/backtest_theover_direction.py data/backtest_exports
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _norm(s: object) -> str:
    return " ".join(str(s).strip().lower().split())


def _decimal(odds: float) -> float:
    o = float(odds)
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def _norm_result(v: object) -> str:
    s = str(v).strip().upper()
    if s.startswith("W"):
        return "WIN"
    if s.startswith("L"):
        return "LOSS"
    if s.startswith("P"):
        return "PUSH"
    return ""


OUT_COLS = [
    "league", "Home", "Away", "best_pick", "WinProbability",
    "effective_win_probability", "effective_edge", "effective_expected_value",
    "consensus_agreement", "kalshi_probability", "Kelly_Bet_Size", "Win Amount", "W/L",
]


def _pick_family(pick: object) -> str:
    """over / under / side — the comparable part of a pick string."""
    p = _norm(pick)
    if p.startswith("over"):
        return "over"
    if p.startswith("under"):
        return "under"
    return "side"


def find_line_drift(exp: pd.DataFrame, rec: pd.DataFrame) -> list[dict]:
    """Export rows whose matchup+direction matches a recap row but whose PICK
    (line) differs — i.e. the recap was built from an earlier card than the
    final export. The recap's graded outcome belongs to a DIFFERENT bet, so
    these must be excluded, not silently dropped: on 11 Jun the recap graded
    NYM/STL 'Under 5.5' LOSS while the final card's Under 10.5 WON."""
    rec_by_game: dict[tuple, list] = {}
    for _, r in rec.iterrows():
        key = (_norm(r.get("Home")), _norm(r.get("Away")), _pick_family(r.get("Pick Taken")))
        rec_by_game.setdefault(key, []).append(_norm(r.get("Pick Taken")))

    drift = []
    for _, e in exp.iterrows():
        pick = _norm(e.get("best_pick"))
        if "unresolved" in pick:
            continue
        key = (_norm(e.get("Home")), _norm(e.get("Away")), _pick_family(e.get("best_pick")))
        recap_picks = rec_by_game.get(key, [])
        if recap_picks and pick not in recap_picks:
            drift.append({
                "Home": e.get("Home"), "Away": e.get("Away"),
                "export_pick": e.get("best_pick"), "recap_pick": recap_picks[0],
            })
    return drift


def _run_id_mismatch(exp: pd.DataFrame, rec: pd.DataFrame) -> tuple[str, str] | None:
    """(export_run_id, recap_run_id) when both frames carry run ids and they
    differ — the recap was built from a different pipeline run than the export."""
    if "export_run_id" not in exp.columns or "export_run_id" not in rec.columns:
        return None
    exp_ids = exp["export_run_id"].dropna().astype(str).unique()
    rec_ids = rec["export_run_id"].dropna().astype(str).unique()
    if len(exp_ids) != 1 or len(rec_ids) != 1:
        return None
    return None if exp_ids[0] == rec_ids[0] else (exp_ids[0], rec_ids[0])


def _clean_line_keys(exp: pd.DataFrame) -> set[tuple[str, str, str]]:
    """(Home, Away, best_pick) keys whose EXPORT line passed every provenance/consistency
    guard. A pick can be "No Play" for two very different reasons: (a) its edge was below
    the stake threshold on a perfectly good line, or (b) its line was rejected/recovered/
    suspicious. Case (a) is a valid directional (prediction, outcome) datapoint and belongs
    in the calibration — excluding it biases the fit toward only the bets we were confident
    enough to stake. Case (b) must stay excluded: the graded outcome may describe a different
    bet than the one we'd have placed. This set marks the case-(a) rows so a clean below-stake
    pick still grades, while corrupt-feed No Play rows (6-19/20/21-style) stay out even if a
    corrupt slate is ever fed in."""
    keys: set[tuple[str, str, str]] = set()
    for _, e in exp.iterrows():
        pick = _norm(e.get("best_pick"))
        if not pick or "unresolved" in pick:
            continue
        warn = e.get("line_provenance_warning")
        if pd.notna(warn) and str(warn).strip():
            continue
        consistent = e.get("line_consistency_flag")
        if "line_consistency_flag" in exp.columns and consistent is not None \
                and pd.notna(consistent) and not bool(consistent):
            continue
        keys.add((_norm(e.get("Home")), _norm(e.get("Away")), pick))
    return keys


def grade(export_csv: Path, recap_csv: Path, out_csv: Path) -> None:
    exp = pd.read_csv(export_csv)
    rec = pd.read_csv(recap_csv)
    clean_keys = _clean_line_keys(exp)

    mismatch = _run_id_mismatch(exp, rec)
    if mismatch:
        print(
            f"  [WARN] RUN ID MISMATCH: recap was built from run {mismatch[1]} but the "
            f"export is run {mismatch[0]} — recap outcomes may describe a stale card. "
            f"Drifted lines below are excluded; re-grade from the matching export.",
            file=sys.stderr,
        )

    drift = find_line_drift(exp, rec)
    for d in drift:
        print(
            f"  [WARN] LINE DRIFT (excluded): {d['Home']} v {d['Away']} — export pick "
            f"'{d['export_pick']}' vs recap pick '{d['recap_pick']}'. The recap graded a "
            f"different line; grade this game from final scores instead.",
            file=sys.stderr,
        )

    # Recap pick column is "Pick Taken"; outcome is "Outcome".
    # "Missing Line" rows and unresolved-line placeholders are always excluded: they were
    # never resolvable picks, and the 10 Jun recap showed an unresolved line graded LOSS —
    # phantom losses that poison the calibration fit. A "No Play" row is excluded ONLY when
    # its line is not clean (rejected/recovered/suspicious); a No Play that was merely below
    # the stake threshold on a clean line is a valid directional datapoint and is kept, so
    # the calibration learns from the full slate, not just the bets we staked.
    def _gradeable(r) -> bool:
        status = str(r.get("Status", "")).strip().lower()
        if status == "missing line":
            return False
        if "unresolved" in _norm(r.get("Pick Taken")):
            return False
        # "no edge" is the user-facing export label for "no play" (4 Jul rename);
        # recaps built from those exports carry it, so treat them as synonyms.
        if status in ("no play", "no edge"):
            key = (_norm(r.get("Home")), _norm(r.get("Away")), _norm(r.get("Pick Taken")))
            if key not in clean_keys:
                return False
        # 0-0 finals don't exist in MLB/NBA/NHL — postponed game, void it.
        h = pd.to_numeric(r.get("actual_home_score"), errors="coerce")
        a = pd.to_numeric(r.get("actual_away_score"), errors="coerce")
        if pd.notna(h) and pd.notna(a) and float(h) == 0 and float(a) == 0:
            return False
        return True

    rec_key = {
        (_norm(r.get("Home")), _norm(r.get("Away")), _norm(r.get("Pick Taken"))): _norm_result(r.get("Outcome"))
        for _, r in rec.iterrows()
        if _gradeable(r)
    }

    drifted_games = {(_norm(d["Home"]), _norm(d["Away"])) for d in drift}
    rows, unmatched = [], []
    for _, e in exp.iterrows():
        key = (_norm(e.get("Home")), _norm(e.get("Away")), _norm(e.get("best_pick")))
        result = rec_key.get(key)
        if result is None:
            if (key[0], key[1]) not in drifted_games:  # drift already reported above
                unmatched.append(key)
            continue
        kelly = pd.to_numeric(e.get("Kelly_Bet_Size"), errors="coerce")
        odds = pd.to_numeric(e.get("odds_american"), errors="coerce")
        win_amt = ""
        if result == "WIN" and pd.notna(kelly) and pd.notna(odds):
            win_amt = round(float(kelly) * _decimal(odds), 2)
        rows.append({
            "league": e.get("league"), "Home": e.get("Home"), "Away": e.get("Away"),
            "best_pick": e.get("best_pick"), "WinProbability": e.get("WinProbability"),
            "effective_win_probability": e.get("effective_win_probability"),
            "effective_edge": e.get("effective_edge"),
            "effective_expected_value": e.get("effective_expected_value"),
            "consensus_agreement": e.get("consensus_agreement"),
            "kalshi_probability": e.get("kalshi_probability"),
            "Kelly_Bet_Size": kelly, "Win Amount": win_amt, "W/L": result,
        })

    out = pd.DataFrame(rows, columns=OUT_COLS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    w = sum(1 for r in rows if r["W/L"] == "WIN")
    l = sum(1 for r in rows if r["W/L"] == "LOSS")
    print(f"wrote {out_csv}: {len(out)} graded picks, {w}-{l}")
    if unmatched:
        print(f"  [WARN] {len(unmatched)} export rows had no recap match: {unmatched[:5]}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    grade(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
