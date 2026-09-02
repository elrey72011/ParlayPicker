from __future__ import annotations

import json
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

# Fields pulled into each side's sub-payload for a head-to-head comparison.
# Excludes identity fields (those live at the top level of the game dict).
SIDE_FIELDS = [
    "best_pick", "odds_american", "market_probability", "kalshi_probability",
    "ml_probability", "theover_probability", "calibrated_probability",
    "expected_value", "edge", "consensus_agreement",
]

# The other candidate row for the same game/market family — e.g. a
# total_under row's opposing side is that same game's total_over row. Both
# naming conventions seen in this codebase (h2h_*/moneyline_*) are mapped.
OPPOSITE_MARKET_TYPE = {
    "spread_home": "spread_away", "spread_away": "spread_home",
    "total_over": "total_under", "total_under": "total_over",
    "h2h_home": "h2h_away", "h2h_away": "h2h_home",
    "moneyline_home": "moneyline_away", "moneyline_away": "moneyline_home",
}


PROP_PAYLOAD_MAP = {
    "best_pick": "best_pick",
    "odds_american": "odds_american",
    "MarketProbability": "market_probability",
    "RawWinProbability": "raw_model_probability",
    "CalibratedProbability": "calibrated_probability",
    "ConservativeWinProbability": "conservative_probability",
    "WinProbability": "win_probability",
    "expected_value": "expected_value",
    "edge": "edge",
    "raw_model_edge": "raw_model_edge",
    "line": "line",
    "expected_count": "projected_count",
    "FormSampleSize": "form_sample_size",
    "CalibrationSampleSize": "calibration_sample_size",
    "CalibrationSource": "calibration_source",
    "book": "book",
}


def _normalized_flags(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()][:12]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()][:12]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return [value.strip()[:120]]
    return []


def _attach_gemini_results(
    result: pd.DataFrame,
    row_ids: list[str],
    analyses: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Map validated batch results onto a card without granting wager authority."""
    explanations: list[str] = []
    risk_notes: list[str] = []
    picks: list[str] = []
    confidences: list[str] = []
    flags: list[str] = []
    reviewed: list[bool] = []
    for row_id in row_ids:
        payload = analyses.get(str(row_id), {})
        raw_explanation = str(payload.get("explanation") or "").strip()
        raw_risk = str(payload.get("risk_notes") or "").strip()
        raw_pick = str(payload.get("recommended_bet") or "").strip()
        explanation = raw_explanation or "Gemini analysis unavailable"
        risk = raw_risk or "Gemini analysis unavailable"
        pick = raw_pick or "No Gemini pick"
        confidence = str(payload.get("confidence") or "").strip().upper()
        raw_flags = payload.get("flags")
        normalized_flags = _normalized_flags(raw_flags)
        explanations.append(explanation)
        risk_notes.append(risk)
        picks.append(pick)
        confidences.append(confidence)
        flags.append("|".join(normalized_flags))
        reviewed.append(
            bool(payload)
            and bool(raw_pick)
            and bool(raw_explanation)
            and bool(raw_risk)
            and confidence in {"HIGH", "MEDIUM", "LOW"}
            and isinstance(raw_flags, list)
        )

    result["gemini_explanation"] = explanations
    result["gemini_risk_notes"] = risk_notes
    result["gemini_pick"] = picks
    result["gemini_confidence"] = confidences
    result["gemini_flags"] = flags
    result["gemini_reviewed"] = reviewed
    return result


def _opposing_side_lookup(analysis_df: pd.DataFrame) -> dict:
    """matchup_id -> {market_type: {field: value}} for every analytic-relevant
    row in analysis_df, so each candidate can be paired with its other side."""
    if analysis_df is None or analysis_df.empty or "matchup_id" not in analysis_df.columns:
        return {}
    cols = [c for c in (["matchup_id", "market_type"] + SIDE_FIELDS) if c in analysis_df.columns]
    sub = analysis_df[cols].copy()
    lookup: dict = {}
    for rec in sub.to_dict("records"):
        mid = rec.get("matchup_id")
        mtype = str(rec.get("market_type") or "").strip().lower()
        if not mid or not mtype:
            continue
        lookup.setdefault(str(mid), {})[mtype] = {k: v for k, v in rec.items() if k in SIDE_FIELDS}
    return lookup


def run_gemini_analysis(df: pd.DataFrame, session_state: Any = None, analysis_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Best-effort Gemini annotation that preserves UI behavior when Gemini is unavailable.

    When analysis_df is provided, each candidate row is paired with its opposing
    side from the same game/market (e.g. total_under paired with total_over) so
    Gemini can do a genuine head-to-head comparison instead of only auditing the
    one side this pipeline already selected.
    """
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

        # Pair each candidate with its opposing side, when available, for a
        # genuine head-to-head comparison rather than a one-sided audit.
        opposing_lookup = _opposing_side_lookup(analysis_df)
        if opposing_lookup and "matchup_id" in result.columns:
            matchup_ids = result["matchup_id"].astype(str).tolist()
            for game, mid in zip(games_list, matchup_ids):
                mtype = str(game.get("market_type") or "").strip().lower()
                opp_mtype = OPPOSITE_MARKET_TYPE.get(mtype)
                side_a = {k: game.pop(k) for k in SIDE_FIELDS if k in game}
                game["side_a"] = side_a
                opp_row = opposing_lookup.get(mid, {}).get(opp_mtype) if opp_mtype else None
                game["side_b"] = opp_row if opp_row else None

        # Call with session_state
        analyses = generate_batch_confidence_explanation(games_list, session_state)

        result = _attach_gemini_results(
            result,
            [str(value) for value in llm_payload["game_id"].tolist()],
            analyses,
        )

    except Exception as exc:  # pragma: no cover
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Gemini integration mapping failed: {exc}", exc_info=True)
        result["gemini_explanation"] = "Gemini analysis unavailable"
        result["gemini_risk_notes"] = "Gemini analysis unavailable"
        result["gemini_pick"] = "No Gemini pick"
        result["gemini_confidence"] = ""
        result["gemini_flags"] = ""
        result["gemini_reviewed"] = False

    return result


def run_gemini_prop_analysis(
    df: pd.DataFrame,
    session_state: Any = None,
) -> pd.DataFrame:
    """Run the same structured Gemini review over MLB and NFL player props.

    The payload intentionally excludes the existing production verdict and
    stake. Gemini sees the offered price, model/market probabilities, projection,
    and calibration evidence, then reviews the selected side independently.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    result = df.copy()
    try:
        from app_core.llm_assistant import generate_batch_confidence_explanation

        games: list[dict[str, Any]] = []
        row_ids: list[str] = []
        for position, (_, row) in enumerate(result.iterrows()):
            row_id = f"prop-{position}"
            row_ids.append(row_id)
            side = {
                payload_name: row.get(column_name)
                for column_name, payload_name in PROP_PAYLOAD_MAP.items()
                if column_name in result.columns
            }
            live_fields = (
                pd.notna(row.get("odds_american"))
                and pd.notna(row.get("line"))
                and pd.notna(row.get("expected_count"))
            )
            games.append(
                {
                    "game_id": row_id,
                    "league": row.get("league"),
                    "matchup": row.get("matchup"),
                    "player": row.get("player"),
                    "participant_type": row.get("participant_type"),
                    "market_type": row.get("market_type"),
                    "game_date": row.get("game_date"),
                    "is_player_prop": True,
                    "is_live_data": bool(live_fields),
                    "side_a": side,
                    "side_b": None,
                }
            )
        analyses = generate_batch_confidence_explanation(games, session_state)
        return _attach_gemini_results(result, row_ids, analyses)
    except Exception as exc:  # pragma: no cover - external SDK/runtime boundary
        import logging

        logging.getLogger(__name__).error(
            "Gemini prop integration mapping failed: %s", exc, exc_info=True
        )
        result["gemini_explanation"] = "Gemini analysis unavailable"
        result["gemini_risk_notes"] = "Gemini analysis unavailable"
        result["gemini_pick"] = "No Gemini pick"
        result["gemini_confidence"] = ""
        result["gemini_flags"] = ""
        result["gemini_reviewed"] = False
        return result
