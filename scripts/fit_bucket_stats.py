#!/usr/bin/env python3
"""
Fit realized win rates by (league, market family, consensus) bucket from the
graded slates in data/backtest_exports/, for the empirical tier overlay
(core/empirical_tiers.py).

Motivation (Jun 5-10 recaps): tiers assigned from model-vs-market EV/edge were
INVERTED vs reality — Actionable went 1-4, HV 3-11 (~21%) while Below Threshold
went 29-20 (59%). Realized bucket performance is the signal the tiers should
express. Buckets are deliberately coarse so each carries a usable sample; the
overlay shrinks each bucket's delta toward the overall mean by n/(n+shrink).

Usage
-----
    python3 scripts/fit_bucket_stats.py [exports_dir] [out_json]

Defaults: data/backtest_exports -> data/calibration/bucket_stats.json
Re-run alongside scripts/fit_calibration.py after each graded slate.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.empirical_tiers import DEFAULT_BUCKET_STATS_PATH, bucket_key  # noqa: E402
from scripts.fit_calibration import _WL_CELL_RE  # noqa: E402

_CONSENSUS_CAPTURE_RE = re.compile(
    r"[0-9]*\.?[0-9]+%?,(Agrees|Neutral|Disagrees|No Kalshi),"
)
_LEAGUE_RE = re.compile(r"\b(MLB|NBA|NHL)\b")
_DIRECTION_RE = re.compile(r"\b(Over|Under)\b")


def _rows_from_txt(text: str) -> list[dict]:
    """Single-line hand-pasted slates: pair each consensus token with the row
    fields that follow it (league, best_pick direction, W/L) before the next
    consensus token. Same invariants as scripts/fit_calibration._extract_txt."""
    rows = []
    matches = list(_CONSENSUS_CAPTURE_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        span = text[m.end():end]
        wl = _WL_CELL_RE.search(span)
        league = _LEAGUE_RE.search(span[: wl.start()] if wl else span)
        if not (wl and league):
            continue
        direction = _DIRECTION_RE.search(span[: wl.start()])
        market = f"total_{direction.group(1).lower()}" if direction else "side"
        rows.append({
            "league": league.group(1),
            "market_type": market,
            "consensus": m.group(1),
            "win": int(wl.group(1) == "W"),
        })
    return rows


def _rows_from_csv(df: pd.DataFrame) -> list[dict]:
    if not {"league", "best_pick", "consensus_agreement", "W/L"}.issubset(df.columns):
        return []
    rows = []
    for _, r in df.iterrows():
        wl = str(r["W/L"]).strip().upper()
        if wl not in ("WIN", "LOSS", "W", "L"):
            continue
        pick = str(r["best_pick"]).lower()
        market = "total_over" if "over" in pick else ("total_under" if "under" in pick else "side")
        rows.append({
            "league": str(r["league"]),
            "market_type": market,
            "consensus": str(r["consensus_agreement"]),
            "win": int(wl in ("WIN", "W")),
        })
    return rows


def main() -> int:
    exports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/backtest_exports")
    out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BUCKET_STATS_PATH

    rows: list[dict] = []
    for f in sorted(exports_dir.glob("*.csv")):
        rows += _rows_from_csv(pd.read_csv(f))
    for f in sorted(exports_dir.glob("*.txt")):
        rows += _rows_from_txt(f.read_text())
    if not rows:
        raise SystemExit(f"no graded picks found under {exports_dir}")

    graded = pd.DataFrame(rows)
    graded["bucket"] = [
        bucket_key(l, m, c) for l, m, c in zip(graded["league"], graded["market_type"], graded["consensus"])
    ]
    by_bucket = graded.groupby("bucket").agg(n=("win", "size"), wins=("win", "sum"))
    overall = float(graded["win"].mean())

    payload = {
        "overall": {"n": int(len(graded)), "win_rate": overall},
        "buckets": {
            b: {"n": int(r["n"]), "wins": int(r["wins"]), "win_rate": float(r["wins"] / r["n"])}
            for b, r in by_bucket.iterrows()
        },
        "meta": {"source": str(exports_dir), "fitted_on": pd.Timestamp.now().strftime("%Y-%m-%d")},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"fit on {len(graded)} graded picks -> {out_json}")
    print(f"overall win rate: {overall:.3f}\n")
    rpt = by_bucket.assign(rate=by_bucket["wins"] / by_bucket["n"]).sort_values("rate", ascending=False)
    print(rpt.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
