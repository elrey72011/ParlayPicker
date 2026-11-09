"""Offline edge screens on graded data already in the repository (roadmap M5, item 2). No network, no credits.

    python3 scripts/offline_edge_screens.py            # prints the report as Markdown

Two questions the audit left unmeasured (``docs/roadmap/VALIDATION-ROADMAP-2026-11-09.md``, section 5):

* **MLB props against their logged prices** (``data/prop_results/prop_results_log.csv``). Each graded
  pick has the American odds it was logged at, so its break-even win rate is known.
* **Graded best picks against Kalshi** (``data/backtest_exports/*.csv``). Each pick has the Kalshi
  probability for its side at export time, a market price to beat.

For every segment the report gives picks, wins, the win rate, the mean break-even (or market)
probability, the excess (win rate minus that), and a z score. The standard error is clustered by
slate date: picks on one night share weather, umpires and line moves, so they are not independent.
A segment is noise unless |z| is about 2 or more **and** it was named before looking; every row here
is in-sample. Nothing is fitted and nothing here grants authority.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_PICKS = 30


def decimal_odds(american):
    return 1 + (american / 100 if american > 0 else 100 / -american)


def segment(frame, label, *, price="breakeven", day="day"):
    """One report row: wins against the summed ``price`` probabilities, standard error clustered by ``day``."""
    n = len(frame)
    if n < MIN_PICKS:
        return None
    residual = frame["won"] - frame[price]
    clustered = math.sqrt(float((residual.groupby(frame[day]).sum() ** 2).sum()))
    row = {"segment": label, "picks": n, "days": int(frame[day].nunique()), "win_rate": float(frame["won"].mean()),
           "price": float(frame[price].mean()), "excess": float(residual.mean()),
           "z": float(residual.sum() / clustered) if clustered else None}
    if "decimal" in frame:
        returns = frame["won"] * (frame["decimal"] - 1) - (1 - frame["won"])
        row["roi"] = float(returns.mean())
    return row


def props(path=ROOT / "data/prop_results/prop_results_log.csv"):
    frame = pd.read_csv(path)
    frame = frame[frame["result"].isin(["WIN", "LOSS"])].copy()
    frame["won"] = (frame["result"] == "WIN").astype(int)
    frame["decimal"] = frame["odds_american"].map(decimal_odds)
    frame["breakeven"] = 1 / frame["decimal"]
    frame["day"] = frame["game_date"]
    rows = [segment(frame, "All graded props")]
    for column, name in (("book", "Book"), ("side", "Side"), ("participant_type", "Type"), ("market_type", "Market")):
        rows += [segment(part, f"{name}: {value}") for value, part in sorted(frame.groupby(column))]
    rows += [segment(frame[frame["edge"] > 0], "Model edge above 0"),
             segment(frame[frame["edge"] >= 0.03], "Model edge 3% or more"),
             segment(frame[frame["expected_value"] > 0], "Model EV above 0"),
             segment(frame[frame["source_stake_status"].str.startswith("Qualified")], "Status: Qualified"),
             segment(frame[frame["odds_american"] <= -200], "Price -200 or shorter"),
             segment(frame[frame["odds_american"] > -200], "Price longer than -200")]
    return frame, [r for r in rows if r]


def best_picks(pattern=str(ROOT / "data/backtest_exports/*.csv")):
    frame = pd.concat([pd.read_csv(name).assign(day=Path(name).stem) for name in sorted(glob.glob(pattern))],
                      ignore_index=True)
    frame = frame[frame["W/L"].isin(["WIN", "LOSS"])].copy()
    frame["won"] = (frame["W/L"] == "WIN").astype(int)
    frame["flat"] = 110 / 210                                   # break-even at -110: the exports carry no price
    frame["decimal"] = 1 + 100 / 110
    flat = [segment(frame, "All graded picks (-110 assumed)", price="flat")]
    flat += [segment(part, f"League: {value}", price="flat") for value, part in sorted(frame.groupby("league"))]
    flat += [segment(part, f"Kalshi: {value}", price="flat")
             for value, part in sorted(frame.groupby("consensus_agreement"))]
    flat += [segment(frame[frame["Kelly_Bet_Size"] > 0], "Staked (Kelly above 0)", price="flat"),
             segment(frame[frame["effective_edge"] >= 0.03], "Model edge 3% or more", price="flat"),
             segment(frame[frame["effective_edge"] >= 0.06], "Model edge 6% or more", price="flat")]
    # Against Kalshi's own price for the pick's side. 0 means no market; values at the extremes are
    # unit or matching errors, so both are left out.
    priced = frame[(frame["kalshi_probability"] > 0.05) & (frame["kalshi_probability"] < 0.95)].drop(columns="decimal")
    kalshi = [segment(priced, "All picks with a Kalshi price", price="kalshi_probability")]
    kalshi += [segment(part, f"Kalshi: {value}", price="kalshi_probability")
               for value, part in sorted(priced.groupby("consensus_agreement"))]
    kalshi += [segment(priced[priced["effective_win_probability"] - priced["kalshi_probability"] >= gap],
                       f"Model at least {gap:.0%} above Kalshi", price="kalshi_probability") for gap in (0.03, 0.06)]
    return frame, [r for r in flat if r], [r for r in kalshi if r]


def table(rows, price_label):
    roi = any("roi" in r for r in rows)
    lines = [f"| Segment | Picks | Days | Win rate | {price_label} | Excess | z |" + (" Flat ROI |" if roi else ""),
             "|---|---|---|---|---|---|---|" + ("---|" if roi else "")]
    for r in rows:
        z = "" if r["z"] is None else f"{r['z']:+.1f}"
        lines.append(f"| {r['segment']} | {r['picks']} | {r['days']} | {r['win_rate']:.1%} | {r['price']:.1%} | "
                     f"{r['excess']:+.1%} | {z} |" + (f" {r['roi']:+.1%} |" if roi else ""))
    return lines


def report():
    prop_frame, prop_rows = props()
    pick_frame, flat_rows, kalshi_rows = best_picks()
    lines = ["## MLB props against their logged prices", "",
             f"{len(prop_frame)} graded picks, {prop_frame['day'].min()} to {prop_frame['day'].max()}, "
             f"{prop_frame['day'].nunique()} slates. Break-even is 1 / decimal odds at the logged price.", ""]
    lines += table(prop_rows, "Break-even")
    lines += ["", "## Graded best picks at an assumed -110", "",
              f"{len(pick_frame)} graded picks, {pick_frame['day'].min()} to {pick_frame['day'].max()}, "
              f"{pick_frame['day'].nunique()} slates. The exports carry no price, so break-even is 52.4%.", ""]
    lines += table(flat_rows, "Break-even")
    lines += ["", "## Graded best picks against Kalshi's price for the same side", ""]
    lines += table(kalshi_rows, "Kalshi")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(report())
