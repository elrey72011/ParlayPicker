from __future__ import annotations

from typing import Any
import pandas as pd

# Analytic inputs Gemini is allowed to see: market/model signals and game
# identity only. Deliberately an ALLOWLIST, not a blocklist of date columns —
# best_picks_df carries ~80 internal pipeline columns (Pick_Status,
# Status_Reason, status_blocker_reason, production_*, kelly_*,
# empty_card_recovery_*, Pick_Quality, Triple_Filter_Rank, ...) that state
# this system's own verdict on the bet. Passing those through let Gemini see
# "the system already rejected this" and simply echo that back as
# recommended_bet: "none" instead of forming an independent judgment — every
# non-Actionable row got a real explanation/risk_notes but no real pick. An
# allowlist also means newly added verdict/diagnostic columns don't leak in
# by default.
LLM_PAYLOAD_COLUMNS = [
    "game_id", "league", "home_team", "Home", "away_team", "Away",
    "market_type", "best_pick", "odds_american",
    "market_probability", "kalshi_probability", "ml_probability",
    "theover_probability", "calibrated_probability", "WinProbability",
    "expected_value", "edge", "consensus_agreement", "is_live_data",
]


def run_gemini_analysis(df: pd.DataFrame, session_state: Any = None) -> pd.DataFrame:
    """Best-effort Gemini annotation that preserves UI behavior when Gemini is unavailable."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    result = df.copy()

    try:
        from app_core.llm_assistant import generate_batch_confidence_explanation

        # Derive is_live_data from the full result before narrowing to the
        # allowlisted payload below (stats_quality/used_stale_features aren't
        # themselves part of the analytic payload Gemini reasons over).
        if "stats_quality" in result.columns:
            result["is_live_data"] = result["stats_quality"].isin(["REAL", "ESPN"])
        elif "used_stale_features" in result.columns:
            result["is_live_data"] = ~result["used_stale_features"]
        elif "is_live_data" not in result.columns:
            result["is_live_data"] = False

        # Restrict the LLM payload to analytic inputs (see LLM_PAYLOAD_COLUMNS) so
        # Gemini reasons from the same raw signals a human would, not our own
        # Pick_Status/Status_Reason verdict on those signals.
        available_cols = [c for c in LLM_PAYLOAD_COLUMNS if c in result.columns]
        llm_payload = result[available_cols].copy()

        # Inject a fallback game_id if one doesn't exist
        if "game_id" not in llm_payload.columns:
            llm_payload["game_id"] = [str(i) for i in range(len(llm_payload))]

        if "is_live_data" not in llm_payload.columns:
            llm_payload["is_live_data"] = False

        games_list = llm_payload.to_dict('records') # Convert to List[Dict]

        # Call with session_state
        analyses = generate_batch_confidence_explanation(games_list, session_state)

        # Unpack dictionary into the two expected export columns based on game_id
        # Use result.index to ensure alignment
        analyses_results = [analyses.get(str(gid), {}) for gid in llm_payload["game_id"]]

        explanations = []
        risk_notes = []
        picks = []
        for res in analyses_results:
            expl = res.get("explanation")
            if expl is None or expl == "":
                expl = "Gemini analysis unavailable"

            risk = res.get("risk_notes")
            if risk is None or risk == "":
                # The user specified defaulting gemini_risk_notes to "Gemini analysis unavailable" if it's missing
                risk = "Gemini analysis unavailable"

            pick = res.get("recommended_bet")
            if pick is None or pick == "":
                pick = "No Gemini pick"

            explanations.append(str(expl))
            risk_notes.append(str(risk))
            picks.append(str(pick))

        result["gemini_explanation"] = explanations
        result["gemini_risk_notes"] = risk_notes
        result["gemini_pick"] = picks

    except Exception as exc:  # pragma: no cover
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Gemini integration mapping failed: {exc}", exc_info=True)
        result["gemini_explanation"] = "Gemini analysis unavailable"
        result["gemini_risk_notes"] = "Gemini analysis unavailable"
        result["gemini_pick"] = "No Gemini pick"

    return result
