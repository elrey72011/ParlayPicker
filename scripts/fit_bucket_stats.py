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

Recency weighting (1 Jul): the fit was equal-weight over every pick since 20 May,
so a bucket that decayed mid-season stayed "proven" on stale volume — through
30 Jun the cumulative table read over:Agrees 51.9% / over:Disagrees 46.4% while
the last 21 days realized 44.4% and 29.4%, and the overall card ran 44.9% over
its final 127 graded picks with no tier reaction. Each pick is now weighted
0.5 ** (age_days / HALF_LIFE_DAYS), aged against the NEWEST slate in the exports
(deterministic: rerunning on a later wall-clock day with no new slates yields
the same fit). Bucket ``n`` becomes the Kish effective sample size
((sum w)^2 / sum w^2) so the overlay's sample-size floors and n/(n+shrink)
confidence honestly reflect the weighted evidence, and ``wins`` is scaled to
keep wins/n == win_rate for the overlay's Laplace smoothing. Raw counts ride
along as raw_n/raw_wins for transparency. overall.n stays the RAW graded count
("how much history exists"); overall.win_rate is recency-weighted (it is the
baseline the overlay tilts deltas against, so it must live in the same space
as the bucket rates).

Usage
-----
    python3 scripts/fit_bucket_stats.py [exports_dir] [out_json] [half_life_days]

Defaults: data/backtest_exports -> data/calibration/bucket_stats.json, 21-day
half-life. Re-run alongside scripts/fit_calibration.py after each graded slate.
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

# Exponential-decay half-life for graded-pick weights. 21 days keeps effective
# sample sizes usable (every major MLB bucket held eff-n >= 20 on the 30 Jun
# data) while reacting to a regime shift in ~2-3 weeks instead of never. The
# fitted table is not very sensitive to the exact value (21/28/35 all flag the
# same decayed buckets and keep under:Agrees proven).
DEFAULT_HALF_LIFE_DAYS = 21.0

_FILE_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

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


def _file_date(path: Path) -> str | None:
    m = _FILE_DATE_RE.search(path.name)
    return m.group(1) if m else None


def fit_bucket_stats(exports_dir: Path, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> dict:
    """Fit the recency-weighted bucket table from an exports directory."""
    rows: list[dict] = []
    for f in sorted(exports_dir.glob("*.csv")):
        got = _rows_from_csv(pd.read_csv(f))
        for r in got:
            r["date"] = _file_date(f)
        rows += got
    for f in sorted(exports_dir.glob("*.txt")):
        got = _rows_from_txt(f.read_text())
        for r in got:
            r["date"] = _file_date(f)
        rows += got
    if not rows:
        raise SystemExit(f"no graded picks found under {exports_dir}")

    graded = pd.DataFrame(rows)
    graded["bucket"] = [
        bucket_key(l, m, c) for l, m, c in zip(graded["league"], graded["market_type"], graded["consensus"])
    ]

    # Age each pick against the newest dated slate (deterministic anchor). An
    # undated file cannot be aged, so its rows get full weight — with a warning,
    # since a stale undated dump would otherwise masquerade as fresh evidence.
    dates = pd.to_datetime(graded["date"], errors="coerce")
    anchor = dates.max()
    if dates.isna().any():
        print(f"WARNING: {int(dates.isna().sum())} rows from undated files get full weight", file=sys.stderr)
    age_days = (anchor - dates).dt.days.fillna(0).clip(lower=0)
    graded["weight"] = 0.5 ** (age_days / float(half_life_days))

    overall_rate = float((graded["win"] * graded["weight"]).sum() / graded["weight"].sum())

    buckets: dict[str, dict] = {}
    for b, sub in graded.groupby("bucket"):
        w = sub["weight"]
        rate = float((sub["win"] * w).sum() / w.sum())
        eff_n = float(w.sum() ** 2 / (w**2).sum())  # Kish effective sample size
        n = max(1, int(round(eff_n)))
        buckets[b] = {
            "n": n,
            "wins": int(round(rate * n)),
            "win_rate": rate,
            "raw_n": int(len(sub)),
            "raw_wins": int(sub["win"].sum()),
        }

    return {
        "overall": {"n": int(len(graded)), "win_rate": overall_rate},
        "buckets": buckets,
        "meta": {
            "source": str(exports_dir),
            "fitted_on": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "half_life_days": float(half_life_days),
            "recency_anchor": anchor.strftime("%Y-%m-%d") if pd.notna(anchor) else None,
        },
    }


def main() -> int:
    exports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/backtest_exports")
    out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BUCKET_STATS_PATH
    half_life = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_HALF_LIFE_DAYS

    payload = fit_bucket_stats(exports_dir, half_life)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"fit on {payload['overall']['n']} graded picks (half-life {half_life:g}d, "
          f"anchor {payload['meta']['recency_anchor']}) -> {out_json}")
    print(f"overall win rate (recency-weighted): {payload['overall']['win_rate']:.3f}\n")
    rpt = pd.DataFrame(payload["buckets"]).T.sort_values("win_rate", ascending=False)
    print(rpt.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
