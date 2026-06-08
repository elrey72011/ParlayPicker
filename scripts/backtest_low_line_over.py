#!/usr/bin/env python3
"""
Backtest: MLB sub-8.0 total_over guardrail (low_line_over_guardrail).

Question this answers
---------------------
core/streamlit_pipeline.py demotes EVERY MLB total_over whose line is below
MLB_OVER_MIN_TOTAL_LINE (8.0) to "High Variance/Speculative", with no quality
escape hatch (see streamlit_pipeline.py:2832 and weights_config.py:200). This
script measures, from graded historical best-picks exports, how sub-8.0 MLB
overs actually performed -- overall, bucketed by effective win probability, and
under a proposed "strong-pick" escape hatch -- so the thresholds for any escape
hatch can be chosen from data rather than a hunch.

Inputs
------
A directory of graded exports. Two formats are supported:
  * .txt  -- the natural-language rendering of a Strategy Lab xlsx as produced
             by the Drive "read file content" tool (header-driven; see _parse_txt).
  * .csv / .xlsx -- a normal best-picks export with the same column names
             (parsed with pandas; requires openpyxl for xlsx).

Each row needs at least: league, best_pick, W/L (or Result), effective_win_probability
(or WinProbability), edge, consensus_agreement, Kelly_Bet_Size, Win Amount.

Usage
-----
    python3 scripts/backtest_low_line_over.py data/backtest_exports
    python3 scripts/backtest_low_line_over.py data/backtest_exports --line-cap 8.0

Every .txt file embeds a "Totals" W-L row; the loader validates that the parsed
win/loss counts match it and prints a warning on mismatch, so transcription or
parsing errors are caught loudly.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# The guardrail constant under test (mirrors weights_config.MLB_OVER_MIN_TOTAL_LINE).
DEFAULT_LINE_CAP = 8.0

STATUS_KEYWORDS = ["Actionable", "High Variance/Speculative", "Below Threshold", "No Play"]
_STATUS_RE = re.compile("(" + "|".join(re.escape(s) for s in STATUS_KEYWORDS) + ")")


@dataclass
class Pick:
    date: str
    league: str
    home: str
    away: str
    best_pick: str
    market_type: str          # total_over | total_under | spread | moneyline | unknown
    line: float | None
    eff_win_prob: float | None
    win_prob: float | None
    edge: float | None
    consensus: str
    kelly: float
    win_amount: float | None
    result: str               # WIN | LOSS | PUSH | ""
    kalshi_prob: float | None = None   # P(pick side) per Kalshi; < 0.50 => Kalshi opposes direction
    ev: float | None = None            # effective_expected_value (fraction; may be negative)

    @property
    def graded(self) -> bool:
        return self.result in ("WIN", "LOSS")

    @property
    def kalshi_opposes(self) -> bool:
        """Kalshi prices the OTHER side as more likely (a true market override)."""
        return self.kalshi_prob is not None and self.kalshi_prob < 0.50

    @property
    def profit(self) -> float:
        """Net profit on a flat-staked bet using the exported Win Amount (gross return)."""
        if self.result == "WIN" and self.win_amount is not None:
            return self.win_amount - self.kelly
        if self.result == "LOSS":
            return -self.kelly
        return 0.0


# --------------------------------------------------------------------------- #
# Field parsing helpers
# --------------------------------------------------------------------------- #
def _to_prob(raw: str | float | None) -> float | None:
    """Parse '65.6%' or '0.656' (or a float) into a 0..1 probability."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in {"-", "#REF!"}:
        return None
    pct = s.endswith("%")
    s = s.rstrip("%").replace(",", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if pct or v > 1.0:
        v /= 100.0
    return v


def _to_money(raw: str | float | None) -> float | None:
    """Parse ' 1,143 ', '776', '(800)', '-' into a float (parens => negative)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in {"-", "#REF!"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _to_num(raw: str | float | None) -> float | None:
    """Parse a plain numeric (EV/edge fraction). Like _to_money but no currency intent;
    handles parens-as-negative and a trailing %, and does NOT auto-divide by 100."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in {"-", "#REF!"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _classify_pick(best_pick: str) -> tuple[str, float | None]:
    """Return (market_type, line) parsed from a best_pick string like 'Over 7.5'."""
    s = (best_pick or "").strip()
    low = s.lower()
    num = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    line = float(num.group()) if num else None
    if low.startswith("over"):
        return "total_over", abs(line) if line is not None else None
    if low.startswith("under"):
        return "total_under", abs(line) if line is not None else None
    if re.search(r"[+-]\d", s):
        return "spread", line
    if low.endswith(" ml") or " ml " in f" {low} ":
        return "moneyline", None
    return "unknown", line


def _norm_result(raw: str | None) -> str:
    s = (raw or "").strip().upper()
    if s.startswith("W"):
        return "WIN"
    if s.startswith("L"):
        return "LOSS"
    if s.startswith("P"):
        return "PUSH"
    return ""


# --------------------------------------------------------------------------- #
# .txt (Drive rendering) parser
# --------------------------------------------------------------------------- #
def _deescape(text: str) -> str:
    # The Drive rendering escapes _, ", #, ! with a backslash.
    return text.replace("\\_", "_").replace('\\"', '"').replace("\\#", "#").replace("\\!", "!")


def _split_boundary(tok: str, first_col: str) -> tuple[str, str]:
    """A row's last cell (Total Amount) is space-joined to the next row's first
    cell (no comma). Split that merged token back into (last_cell, next_first_cell)."""
    if first_col == "Pick_Status":
        m = _STATUS_RE.search(tok)
        if m:
            return tok[: m.start()].strip(), tok[m.start():].strip()
    # rank-first layout: next row begins with an integer rank
    m = re.search(r"^(.*?)(\d+)\s*$", tok)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    last, _, first = tok.rpartition(" ")
    return last.strip(), first.strip()


def _expected_wl(raw: str) -> tuple[int, int] | None:
    """Pull the grand 'Totals' W-L record for validation. It is the final
    \\d+-\\d+ token in the sheet (dates contain dashes but appear earlier; the
    summary's net-total cell after the W-L has no dash)."""
    matches = re.findall(r"(\d+)-(\d+)", raw)
    if not matches:
        return None
    w, l = matches[-1]
    return (int(w), int(l))


def _parse_txt(raw: str, source: str) -> list[Pick]:
    text = _deescape(raw)
    text = re.sub(r"^\s*best_picks_export\s*-\s*[\dT:/ -]+?T\s+", "", text)  # drop title prefix
    if "Total Amount" not in text:
        raise ValueError(f"{source}: no 'Total Amount' header found")
    head_str, data_str = text.split("Total Amount", 1)
    headers = [h.strip() for h in (head_str + "Total Amount").split(",")]
    n = len(headers)
    first_col = headers[0]

    # Truncate the trailing summary block (long run of empty cells then 'Totals').
    cut = re.search(r",{6,}", data_str)
    if cut:
        data_str = data_str[: cut.start()]

    fields = next(csv.reader(io.StringIO(data_str.strip())))

    rows: list[list[str]] = []
    cur: list[str] = []
    for i, tok in enumerate(fields):
        if len(cur) == n - 1:
            is_last = i == len(fields) - 1
            if is_last:
                cur.append(tok.strip())
                rows.append(cur)
                cur = []
            else:
                last, nxt = _split_boundary(tok, first_col)
                cur.append(last)
                rows.append(cur)
                cur = [nxt]
        else:
            cur.append(tok.strip())
    if cur and len(cur) == n:
        rows.append(cur)

    date = source
    picks: list[Pick] = []
    for r in rows:
        if len(r) != n:
            continue
        d = dict(zip(headers, r))
        best = d.get("best_pick", "").strip()
        mt, line = _classify_pick(best)
        picks.append(
            Pick(
                date=date,
                league=d.get("league", "").strip().upper(),
                home=d.get("Home", "").strip(),
                away=d.get("Away", "").strip(),
                best_pick=best,
                market_type=mt,
                line=line,
                eff_win_prob=_to_prob(d.get("effective_win_probability")),
                win_prob=_to_prob(d.get("WinProbability")),
                edge=_to_prob(d.get("effective_edge") or d.get("edge")),
                consensus=d.get("consensus_agreement", "").strip(),
                kelly=_to_money(d.get("Kelly_Bet_Size")) or 0.0,
                win_amount=_to_money(d.get("Win Amount")),
                result=_norm_result(d.get("W/L")),
                kalshi_prob=_to_prob(d.get("kalshi_probability")),
                ev=_to_num(d.get("effective_expected_value") or d.get("expected_value")),
            )
        )

    exp = _expected_wl(_deescape(raw))
    if exp:
        w = sum(1 for p in picks if p.result == "WIN")
        l = sum(1 for p in picks if p.result == "LOSS")
        if (w, l) != exp:
            print(f"  [WARN] {source}: parsed W-L {w}-{l} != embedded {exp[0]}-{exp[1]}", file=sys.stderr)
    return picks


def _parse_tabular(path: Path) -> list[Pick]:
    import pandas as pd

    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def g(row, *names):
        for nm in names:
            c = cols.get(nm.lower())
            if c is not None and c in row and str(row[c]).strip() not in ("", "nan"):
                return row[c]
        return None

    picks: list[Pick] = []
    for _, row in df.iterrows():
        best = str(g(row, "best_pick") or "").strip()
        mt, line = _classify_pick(best)
        picks.append(
            Pick(
                date=path.stem,
                league=str(g(row, "league") or "").strip().upper(),
                home=str(g(row, "Home", "home_team") or "").strip(),
                away=str(g(row, "Away", "away_team") or "").strip(),
                best_pick=best,
                market_type=mt,
                line=line,
                eff_win_prob=_to_prob(g(row, "effective_win_probability")),
                win_prob=_to_prob(g(row, "WinProbability")),
                edge=_to_prob(g(row, "effective_edge", "edge")),
                consensus=str(g(row, "consensus_agreement") or "").strip(),
                kelly=_to_money(g(row, "Kelly_Bet_Size")) or 0.0,
                win_amount=_to_money(g(row, "Win Amount")),
                result=_norm_result(g(row, "W/L", "Result")),
                kalshi_prob=_to_prob(g(row, "kalshi_probability")),
                ev=_to_num(g(row, "effective_expected_value", "expected_value")),
            )
        )
    return picks


def load_dir(path: Path) -> list[Pick]:
    picks: list[Pick] = []
    files = sorted(p for p in path.iterdir() if p.suffix in (".txt", ".csv", ".xlsx"))
    for f in files:
        if f.suffix == ".txt":
            picks.extend(_parse_txt(f.read_text(), f.stem))
        else:
            picks.extend(_parse_tabular(f))
    print(f"Loaded {len(picks)} picks from {len(files)} file(s) in {path}")
    return picks


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _summary(picks: list[Pick]) -> dict:
    graded = [p for p in picks if p.graded]
    w = sum(1 for p in graded if p.result == "WIN")
    n = len(graded)
    stake = sum(p.kelly for p in graded)
    profit = sum(p.profit for p in graded)
    return {
        "n": n,
        "wins": w,
        "losses": n - w,
        "hit_rate": (w / n) if n else 0.0,
        "stake": stake,
        "profit": profit,
        "roi": (profit / stake) if stake else 0.0,
    }


def _print_block(title: str, picks: list[Pick]) -> None:
    s = _summary(picks)
    if s["n"] == 0:
        print(f"{title:<46}  (no graded picks)")
        return
    print(
        f"{title:<46}  n={s['n']:>3}  {s['wins']}-{s['losses']}  "
        f"hit={s['hit_rate']:>5.1%}  ROI={s['roi']:>7.1%}  "
        f"(stake {s['stake']:.0f} -> profit {s['profit']:+.0f})"
    )


def run(picks: list[Pick], line_cap: float) -> None:
    sub = [p for p in picks if p.league == "MLB" and p.market_type == "total_over"
           and p.line is not None and p.line < line_cap]
    hi = [p for p in picks if p.league == "MLB" and p.market_type == "total_over"
          and p.line is not None and p.line >= line_cap]

    print("\n" + "=" * 78)
    print(f"MLB total_over guardrail backtest   (line cap = {line_cap})")
    print("=" * 78)
    _print_block("All MLB overs", [p for p in picks if p.league == "MLB" and p.market_type == "total_over"])
    _print_block(f"  >= {line_cap} (currently can be Actionable)", hi)
    _print_block(f"  <  {line_cap} (currently vetoed -> High Var)", sub)

    print(f"\nSub-{line_cap} MLB overs by effective win probability bucket")
    print("-" * 78)
    edges = [0.0, 0.55, 0.60, 0.65, 0.70, 1.01]
    for lo, hib in zip(edges, edges[1:]):
        bucket = [p for p in sub if p.eff_win_prob is not None and lo <= p.eff_win_prob < hib]
        _print_block(f"  eff_win_prob [{lo:.2f}, {hib:.2f})", bucket)

    print("\nSub-{} MLB overs by consensus".format(line_cap))
    print("-" * 78)
    for c in ["Agrees", "Neutral", "Disagrees", "No Kalshi"]:
        _print_block(f"  consensus = {c}", [p for p in sub if p.consensus == c])

    print("\nProposed escape-hatch grids (sub-{} overs that WOULD stay Actionable)".format(line_cap))
    print("-" * 78)
    for min_p in (0.62, 0.64, 0.66):
        for min_e in (0.08, 0.10, 0.12):
            hatch = [
                p for p in sub
                if p.eff_win_prob is not None and p.eff_win_prob >= min_p
                and p.edge is not None and p.edge >= min_e
                and p.consensus == "Agrees"
            ]
            _print_block(f"  eff_win>={min_p:.2f} & edge>={min_e:.2f} & Agrees", hatch)

    # Show the individual sub-cap overs so the sample is fully auditable.
    print(f"\nAll graded sub-{line_cap} MLB overs ({sum(1 for p in sub if p.graded)} picks)")
    print("-" * 78)
    for p in sorted((p for p in sub if p.graded), key=lambda x: (x.eff_win_prob or 0), reverse=True):
        print(
            f"  {p.date:<12} {p.away[:14]:<14} @ {p.home[:14]:<14} {p.best_pick:<10} "
            f"eff={(p.eff_win_prob or 0):.3f} edge={(p.edge or 0):.3f} "
            f"{p.consensus:<10} {p.result:<4} profit={p.profit:+.0f}"
        )


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory of graded exports (.txt/.csv/.xlsx)")
    ap.add_argument("--line-cap", type=float, default=DEFAULT_LINE_CAP)
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.data_dir.is_dir():
        ap.error(f"{args.data_dir} is not a directory")
    picks = load_dir(args.data_dir)
    run(picks, args.line_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
