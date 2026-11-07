"""Gemini as a research reviewer of the top board picks (owner request 2026-11-07, Gemini role #1).

The wager gate (``app_core.gemini_bet_gate``) reviews a pick only after it passes every
deterministic wager check, so while no market is validated Gemini reviews nothing. This
asks Gemini, once per Refresh, to review the top ``limit`` picks with the same structured
review the gate uses (``integrations.gemini_client.run_gemini_analysis``), adding the
research references the board now carries (Pinnacle's no-vig price and EV, the TheOver
signals, why Kalshi was left out) and giving it the Kalshi value the blend actually used.

Research only. The results land in their own ``gemini_research_*`` columns and are saved
with each pick so the track record can measure whether Gemini's agreement predicts
results. They never touch ``gemini_*`` gate columns, stakes, approvals or the pick.

* Off unless the app's Gemini toggle is on; ``PARLAYPICKER_GEMINI_RESEARCH_PICKS`` sets the
  number of picks (default 12, one structured request; 0 turns it off).
* Uses the shared daily request budget (``PARLAYPICKER_GEMINI_DAILY_REQUESTS``) and cache.
* Ranking: Pinnacle EV first (picks that beat Pinnacle), then expected value; pregame only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os

import pandas as pd

LIMIT_ENV = "PARLAYPICKER_GEMINI_RESEARCH_PICKS"
DEFAULT_LIMIT = 12
EXTRA_PAYLOAD = ("pinnacle_fair_probability", "pinnacle_ev", "kalshi_blend_gate", "theover_pick_agrees",
                 "theover_pick_line", "theover_public_pct", "theover_money_pct", "theover_sharp_agrees",
                 "theover_purple_star", "theover_fade_public", "theover_model_hit_rate")
COLUMNS = ("gemini_research_pick", "gemini_research_agrees", "gemini_research_confidence",
           "gemini_research_flags", "gemini_research_explanation", "gemini_research_risk_notes",
           "gemini_research_reviewed_at", "gemini_research_model")


def limit(setting=os.environ.get) -> int:
    try:
        return max(0, int(str(setting(LIMIT_ENV, DEFAULT_LIMIT) or DEFAULT_LIMIT).strip()))
    except ValueError:
        return DEFAULT_LIMIT


def select(best: pd.DataFrame, count: int) -> pd.Index:
    """Index labels of the picks to review: pregame, ranked by Pinnacle EV then EV."""
    if best is None or best.empty or count <= 0:
        return pd.Index([])
    frame = best
    context = frame.get("candidate_context", pd.Series("", index=frame.index)).astype(str).str.upper()
    started = frame.get("game_already_started_flag", pd.Series(False, index=frame.index))
    started = started.astype(str).str.lower().isin({"true", "1"})
    eligible = frame[~context.eq("POST_START_DIAGNOSTIC") & ~started]
    pin = pd.to_numeric(eligible.get("pinnacle_ev", pd.Series(float("nan"), index=eligible.index)), errors="coerce")
    ev = pd.to_numeric(eligible.get("expected_value", pd.Series(float("nan"), index=eligible.index)), errors="coerce")
    order = pd.DataFrame({"pin": pin.fillna(-9.0), "ev": ev.fillna(-9.0)}, index=eligible.index)
    return order.sort_values(["pin", "ev"], ascending=False).head(count).index


def _agrees(pick, best_pick):
    pick, best_pick = str(pick or "").strip().casefold(), str(best_pick or "").strip().casefold()
    if not pick or pick in {"no gemini pick", "none", "gemini analysis unavailable"}:
        return float("nan")
    return 1.0 if pick == best_pick else 0.0


def review(best: pd.DataFrame, *, session_state=None, analysis: pd.DataFrame | None = None, enabled: bool,
           setting=os.environ.get, run=None) -> tuple[pd.DataFrame, dict]:
    """(best with ``gemini_research_*`` columns, diagnostic). Never raises; never changes other columns."""
    out = best.copy() if best is not None else pd.DataFrame()
    for column in COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    count = limit(setting)
    if not enabled or out.empty or count == 0:
        return out, {"status": "OFF" if not enabled or count == 0 else "NO_PICKS", "reviewed": 0}
    chosen = select(out, count)
    if len(chosen) == 0:
        return out, {"status": "NO_PREGAME_PICKS", "reviewed": 0}
    targets = out.loc[chosen].copy()
    if "blend_in_kalshi" in targets.columns:
        # Show Gemini the Kalshi value the blend used (off-line contracts are dropped).
        targets["kalshi_probability"] = pd.to_numeric(targets["blend_in_kalshi"], errors="coerce")
    try:
        if run is None:
            from integrations.gemini_client import run_gemini_analysis as run
        reviewed = run(targets, session_state, analysis, eligible_only=False,
                       extra_columns=tuple(c for c in EXTRA_PAYLOAD if c in targets.columns))
    except Exception as exc:  # research only: never break Refresh
        return out, {"status": "ERROR", "error": type(exc).__name__, "reviewed": 0}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ok = reviewed.get("gemini_reviewed", pd.Series(False, index=reviewed.index)).fillna(False).astype(bool)
    for label in chosen:
        if label not in reviewed.index or not ok.get(label, False):
            continue
        row = reviewed.loc[label]
        out.at[label, "gemini_research_pick"] = row.get("gemini_pick")
        out.at[label, "gemini_research_agrees"] = _agrees(row.get("gemini_pick"), out.at[label, "best_pick"])
        out.at[label, "gemini_research_confidence"] = row.get("gemini_confidence")
        out.at[label, "gemini_research_flags"] = row.get("gemini_flags")
        out.at[label, "gemini_research_explanation"] = row.get("gemini_explanation")
        out.at[label, "gemini_research_risk_notes"] = row.get("gemini_risk_notes")
        stamp = row.get("gemini_reviewed_at")
        out.at[label, "gemini_research_reviewed_at"] = stamp if isinstance(stamp, str) and stamp.strip() else now
        out.at[label, "gemini_research_model"] = row.get("gemini_review_model")
    done = int(out.loc[chosen, "gemini_research_pick"].notna().sum())
    return out, {"status": "REVIEWED" if done else "NO_REVIEW_RETURNED", "selected": len(chosen), "reviewed": done}
