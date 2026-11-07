"""Kalshi enters the blend only at the book's exact line and for the pick's own team (owner 2026-11-06).

The integrator (``app_core.kalshi_integrator``) still matches a Kalshi contract up to
``MAX_LINE_TOLERANCE`` points away (10.5 for NCAAB spreads, 6.5 for NCAAB totals) and
shifts it toward the pick's line, and its title matching can take a different school. On
the 2026-11-06 slate that produced the board's three largest "edges":

* Oklahoma St -18.5: "Oklahoma wins by over 20.5" at 89% (another team, 2 points away)
  -> 76% win estimate, +49% EV;
* South Carolina State +3.5: a 1.5-point contract -> 62%, +21% EV;
* UConn/Michigan Over 143.5: an Over 144.5 contract at 67% -> 57%, +15% EV.

Because Kalshi >= 55% switches the blend to its 40% Kalshi tier, one wrong contract moves
the estimate far from the market. This keeps the integrator's match (and its diagnostics)
unchanged and only decides what the **blend** receives:

* spreads and totals: Kalshi only when its contract line equals the book line
  (``kalshi_line_diff`` == 0, else ``kalshi_matched_line`` == |line|);
* spreads: and, when the contract title names a team ("X wins by over L"), only when X is
  the home or away team (school names compared with "St."/"St" read as "State");
* other markets (moneylines) unchanged;
* a frame with no Kalshi match diagnostics at all (``kalshi_line_diff`` / ``kalshi_matched_line``
  absent: not integrator output) passes unchanged. Within integrator output, a spread or total
  with no recorded contract line is dropped.

A dropped Kalshi value is NaN: the blend falls to its tier without Kalshi, exactly as when
Kalshi has no market. ``kalshi_probability`` itself is kept for research reports;
``kalshi_blend_gate`` records why a value was dropped.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SUBJECT = re.compile(r"^\s*(.+?)\s+wins\s+by\b", re.IGNORECASE)
TOLERANCE = 1e-9


def _school(name) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(name or "").lower())
    words = ["state" if w == "st" else w for w in text.split()]
    return " ".join(words)


def _column(frame, name):
    return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)


def gate(frame: pd.DataFrame, kalshi: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(Kalshi probability for the blend, reason per row: '' kept / code dropped)."""
    kalshi = pd.to_numeric(kalshi, errors="coerce").astype("float64")
    reasons = pd.Series("", index=frame.index, dtype="object")
    if frame.empty:
        return kalshi, reasons
    if "kalshi_line_diff" not in frame.columns and "kalshi_matched_line" not in frame.columns:
        # No Kalshi match diagnostics at all: these values did not come from the integrator
        # (tests, legacy inputs), so there is nothing to judge and they pass unchanged.
        # The integrator always writes these columns.
        return kalshi, reasons
    market = _column(frame, "market_type").astype(str).str.lower()
    is_spread = market.str.startswith("spread")
    is_total = market.str.startswith("total")
    present = kalshi.notna()

    line = pd.to_numeric(_column(frame, "market_line_used"), errors="coerce")
    line = line.fillna(pd.to_numeric(_column(frame, "spread_line"), errors="coerce").where(is_spread))
    line = line.fillna(pd.to_numeric(_column(frame, "total_line"), errors="coerce").where(is_total))
    diff = pd.to_numeric(_column(frame, "kalshi_line_diff"), errors="coerce")
    matched = pd.to_numeric(_column(frame, "kalshi_matched_line"), errors="coerce")
    diff = diff.fillna((matched.abs() - line.abs()).abs())
    off_line = present & (is_spread | is_total) & ~(diff.abs() <= TOLERANCE)
    reasons[off_line] = "KALSHI_LINE_NOT_EXACT"

    titles = _column(frame, "kalshi_market_title").astype(str)
    home = _column(frame, "home_team").map(_school)
    away = _column(frame, "away_team").map(_school)
    subject = titles.map(lambda t: (SUBJECT.match(t).group(1) if SUBJECT.match(t) else None))
    named = subject.notna()
    subject = subject.map(_school)
    wrong_team = present & is_spread & ~off_line & named & ~(subject.eq(home) | subject.eq(away))
    reasons[wrong_team] = "KALSHI_TEAM_NOT_EXACT"

    return kalshi.where(reasons.eq("")), reasons
