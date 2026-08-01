#!/usr/bin/env python3
"""
Fit an isotonic (PAV) calibration table mapping effective_win_probability to the
REALIZED win rate, from the graded slates in data/backtest_exports/.

Motivation (Jun 5-10 recaps): effective_win_probability is overconfident in the
0.55-0.65 band — the band that feeds Actionable/HV promotion and parlay legs.
Dozens of hand-tuned shrink/penalty knobs in weights_config.py have been patching
this symptom one slate at a time; this fits the mapping once, from all graded data,
and writes it to data/calibration/effective_prob_calibration.json where
generate_parlays (and later the gating pipeline) can apply it.

Usage
-----
    python3 scripts/fit_calibration.py [exports_dir] [out_json]

Defaults: data/backtest_exports -> data/calibration/effective_prob_calibration.json
Re-run after each graded slate is added (scripts/grade_slate.py).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.probability_calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    apply_calibration,
    fit_isotonic_calibration,
    save_calibration,
)
from core.walk_forward import chronological_split, probability_metrics  # noqa: E402

PROB_COLS = ["effective_win_probability", "WinProbability"]
ROOT = Path(__file__).resolve().parents[1]


def _source_label(path: Path) -> str:
    """Store portable repo-relative provenance instead of a user's absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _to_prob(v: object) -> float:
    """Accept 0.61, '0.61', or '61.0%'."""
    s = str(v).strip().rstrip("%")
    try:
        p = float(s)
    except ValueError:
        return float("nan")
    if "%" in str(v) or p > 1.0:
        p /= 100.0
    return p


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.strip(): c for c in df.columns}
    prob_col = next((cols[c] for c in PROB_COLS if c in cols), None)
    wl_col = cols.get("W/L")
    if prob_col is None or wl_col is None:
        return pd.DataFrame(columns=["prob", "win"])
    out = pd.DataFrame({
        "prob": df[prob_col].map(_to_prob),
        "wl": df[wl_col].astype(str).str.strip().str.upper(),
    })
    date_col = next((c for c in ("game_date", "event_date", "date") if c in df.columns), None)
    if date_col:
        out["slate_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    out = out[out["wl"].isin(["WIN", "LOSS", "W", "L"])]
    out["win"] = out["wl"].isin(["WIN", "W"]).astype(int)
    keep = ["prob", "win"] + (["slate_date"] if "slate_date" in out.columns else [])
    return out[keep].dropna(subset=["prob", "win"])


# The May .txt slates are hand-pasted spreadsheet dumps: all rows on ONE line, with
# three column-order variants. Two invariants hold across all of them:
#   * effective_win_probability is the field immediately before the consensus token
#     (effective_expected_value, effective_edge, effective_win_probability, consensus)
#   * the graded W/L cell (" W " / " L ") appears after that consensus token and
#     before the NEXT row's consensus token
# so we pair each consensus match with the first W/L cell that follows it. Summary
# rows ("9-8") and ungraded rows pair nothing and drop out naturally.
_CONSENSUS_RE = re.compile(r"([0-9]*\.?[0-9]+%?),(?:Agrees|Neutral|Disagrees|No Kalshi),")
_WL_CELL_RE = re.compile(r",\s([WL])\s,")


def _extract_txt(text: str) -> pd.DataFrame:
    rows = []
    matches = list(_CONSENSUS_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        wl = _WL_CELL_RE.search(text, m.end(), end)
        if wl:
            rows.append({"prob": _to_prob(m.group(1)), "win": int(wl.group(1) == "W")})
    return pd.DataFrame(rows, columns=["prob", "win"]).dropna()


def load_graded(exports_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(exports_dir.glob("*.csv")) + sorted(exports_dir.glob("*.txt")):
        try:
            if f.suffix == ".txt":
                got = _extract_txt(f.read_text())
            else:
                got = _extract(pd.read_csv(f))
        except Exception as e:  # hand-pasted slates; skip unparseable ones
            print(f"  [skip] {f.name}: {e}", file=sys.stderr)
            continue
        if not got.empty:
            file_date = pd.to_datetime(f.stem[:10], errors="coerce", utc=True)
            if "slate_date" not in got.columns:
                got["slate_date"] = file_date
            else:
                got["slate_date"] = got["slate_date"].fillna(file_date)
            frames.append(got)
            print(f"  {f.name}: {len(got)} graded picks")
    if not frames:
        raise SystemExit(f"no graded picks found under {exports_dir}")
    return pd.concat(frames, ignore_index=True)


def report(graded: pd.DataFrame) -> None:
    bins = pd.cut(graded["prob"], [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0])
    by_bin = graded.groupby(bins, observed=True).agg(n=("win", "size"), predicted=("prob", "mean"), realized=("win", "mean"))
    print("\nPredicted vs realized by bin:")
    print(by_bin.to_string(float_format=lambda v: f"{v:.3f}"))


def validate_calibration_promotion(
    graded: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
    min_train_rows: int = 100,
) -> dict:
    """Evaluate a train-only isotonic map on a strictly future holdout."""
    required = {"prob", "win", "slate_date"}
    if not required.issubset(graded.columns) or graded["slate_date"].isna().any():
        return {"promotable": False, "reason": "dated graded rows are required"}
    if len(graded) <= int(min_train_rows):
        return {
            "promotable": False,
            "reason": (
                f"not enough training rows: need more than {int(min_train_rows)}, "
                f"got {len(graded)}"
            ),
        }
    try:
        train, test = chronological_split(
            graded,
            "slate_date",
            test_fraction=test_fraction,
            min_train_rows=min_train_rows,
        )
    except ValueError as exc:
        return {"promotable": False, "reason": str(exc)}
    knots = fit_isotonic_calibration(train["prob"].tolist(), train["win"].tolist())
    calibrated = apply_calibration(test["prob"], knots)
    base_rate = pd.Series(float(train["win"].mean()), index=test.index)
    metrics = {
        "calibrated": probability_metrics(calibrated, test["win"]),
        "raw": probability_metrics(test["prob"], test["win"]),
        "train_base_rate": probability_metrics(base_rate, test["win"]),
    }
    promotable = all(
        metrics["calibrated"][metric]
        < min(metrics["raw"][metric], metrics["train_base_rate"][metric])
        for metric in ("brier", "log_loss")
    )
    return {
        "promotable": bool(promotable),
        "reason": (
            "beats raw probability and train base rate on the future holdout"
            if promotable
            else "does not beat raw probability and train base rate on the future holdout"
        ),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(pd.to_datetime(train["slate_date"], utc=True).max()),
        "test_start": str(pd.to_datetime(test["slate_date"], utc=True).min()),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports_dir", nargs="?", default="data/backtest_exports")
    parser.add_argument("out_json", nargs="?", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write even when chronological validation fails (emergency/manual use only).",
    )
    args = parser.parse_args(argv)
    exports_dir = Path(args.exports_dir)
    out_json = Path(args.out_json)

    graded = load_graded(exports_dir)
    validation = validate_calibration_promotion(graded)
    print("\nChronological promotion validation:")
    for name, metrics in validation.get("metrics", {}).items():
        print(
            f"  {name:<16} log_loss={metrics['log_loss']:.4f} "
            f"brier={metrics['brier']:.4f} n={metrics['n']}"
        )
    if not validation["promotable"] and not args.force:
        print(f"\nPROMOTION BLOCKED: {validation['reason']}", file=sys.stderr)
        return 2

    knots = fit_isotonic_calibration(graded["prob"].tolist(), graded["win"].tolist())
    save_calibration(knots, out_json, meta={
        "n_graded": int(len(graded)),
        "source": _source_label(exports_dir),
        "fitted_on": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "prob_col": "effective_win_probability (fallback WinProbability)",
        "validation": validation,
    })
    print(f"\nfit on {len(graded)} graded picks -> {len(knots)} knots -> {out_json}")
    report(graded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
