"""
Vertex AI Master Analyzer
Consolidates ALL data sources for ultimate best bet recommendations.
FIXED: UnboundLocalError on game_dt and Kalshi matching logic.
"""
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytz

import numpy as np
import pandas as pd
import streamlit as st

from app_core.team_name_matcher import TeamNameMatcher
from app_core.kalshi_integrator import (
    KalshiMatchResult,
    match_game_to_kalshi,
)
from app_core.vertex_ai_endpoint import (
    VERTEX_MODEL_DISPLAY_NAME,
    VERTEX_FEATURE_COLUMNS,
    is_vertex_prediction_configured,
    score_with_vertex,
    predict_win_probabilities,
)

logger = logging.getLogger(__name__)

# ... (Normalization helpers match_game split_game remain the same) ...
# ... (Copy the helper functions from your previous file here if needed) ...

# -------------------------------
# Normalization helpers
# -------------------------------

def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = s.replace("st ", "state ").replace("st.", "state ")
    s = s.replace("univ", "university")
    return re.sub(r"\s+", " ", s)

def normalize_league(value: str) -> str:
    if not isinstance(value, str):
        return ""
    v = value.lower().strip()
    if "ncaab" in v: return "ncaab"
    if "ncaaf" in v: return "ncaaf"
    if "nba" in v: return "nba"
    if "nhl" in v: return "nhl"
    return v

def split_game(game: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(game, str) or "@" not in game:
        return None, None
    away, home = [g.strip() for g in game.split("@", 1)]
    return home, away

def is_today_calendar_day(time_str) -> bool:
    if pd.isna(time_str) or not time_str:
        return False
    try:
        dt = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
        eastern = pytz.timezone("US/Eastern")
        dt_et = dt.astimezone(eastern)
        today_et = datetime.now(eastern).date()
        return dt_et.date() == today_et
    except Exception:
        return False

def american_to_decimal(odds: Optional[float]) -> Optional[float]:
    if odds is None or odds == 0 or pd.isna(odds):
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)

def implied_prob_from_american(odds: Optional[float]) -> Optional[float]:
    dec = american_to_decimal(odds)
    if dec is None or dec <= 1.0:
        return None
    return 1.0 / dec

TEAM_FUZZY_THRESHOLD = 0.8
MAX_LINE_DIFF = 1.5

class VertexMasterAnalyzer:
    def __init__(
        self,
        odds_api_client: Any = None,
        sportsdata_clients: Optional[Dict[str, Any]] = None,
        apisports_clients: Optional[Dict[str, Any]] = None,
        sentiment_analyzer: Any = None,
        local_ml_predictor: Any = None,
        theover_data: Optional[Dict[str, pd.DataFrame]] = None,
        kalshi_integrator: Any = None,
        use_kalshi: bool = True,
    ) -> None:
        self.odds_api = odds_api_client
        self.sportsdata = sportsdata_clients or {}
        self.apisports = apisports_clients or {}
        self.sentiment = sentiment_analyzer
        self.local_ml = local_ml_predictor
        self.theover = theover_data or {}
        self.kalshi = kalshi_integrator
        self.use_kalshi = bool(use_kalshi and st.session_state.get("kalshi_enabled", True))

    # ... (Other methods remain the same until _get_kalshi_features) ...
    # PASTE THIS METHOD EXACTLY TO FIX THE ERROR:

    def _get_kalshi_features(
        self, game: Dict[str, Any], prefetch_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fetch Kalshi probability for a game via the shared integrator."""

        feats: Dict[str, Any] = {
            "kalshi_available": False,
            "kalshi_prob": None,
            "kalshi_alignment": None,
            "kalshi_match_debug": "no_match_found",
            "kalshi_label": None,
            "kalshi_volume": None,
            "kalshi_confidence": None,
            "kalshi_status": "no_market_match",
        }

        # If Kalshi is not configured, bail out cleanly
        if not getattr(self, "kalshi", None) or not getattr(self, "use_kalshi", True):
            feats["kalshi_match_debug"] = "kalshi_not_configured"
            feats["kalshi_status"] = "kalshi_disabled"
            return feats

        try:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            league = game.get("league") or game.get("sport_key") or "NBA"

            # FIX: Define game_dt HERE so it exists even if we use prefetch_info
            game_time = game.get("commence_time") or game.get("game_time")
            game_dt = None
            if game_time:
                try:
                    game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
                except Exception:
                    game_dt = None

            market_info = prefetch_info
            if market_info is None:
                # Delegate to the integrator as the single source of truth
                market_info = match_game_to_kalshi(
                    league,
                    home,
                    away,
                    game_dt,
                    integrator=self.kalshi,
                )

            logging.info(
                f"[Kalshi FETCH] home={home} away={away} dt={game_dt} "
                f"market_keys={list(market_info.keys()) if isinstance(market_info, dict) else type(market_info)}"
            )

            # Handle KalshiMatchResult or legacy dicts
            is_match_result = isinstance(market_info, dict) and "matched" in market_info
            if is_match_result:
                result: KalshiMatchResult = market_info  # type: ignore[assignment]
                feats["kalshi_status"] = result.get("reason", "no_market_match")
                feats["kalshi_match_debug"] = result.get("raw_event_id") or result.get("reason", "no_match_found")
                
                if not result.get("matched"):
                    return feats

                prob = result.get("probability")
                label = result.get("label")
            else:
                # Legacy dict support
                prob = market_info.get("kalshi_probability") if isinstance(market_info, dict) else None
                label = market_info.get("kalshi_label") if isinstance(market_info, dict) else None
                feats["kalshi_status"] = (
                    market_info.get("kalshi_match_debug")
                    if isinstance(market_info, dict)
                    else "no_market_match"
                )
                feats["kalshi_match_debug"] = feats["kalshi_status"]

            if prob is None:
                return feats

            try:
                prob = float(prob)
                prob = max(0.0, min(1.0, prob))
            except Exception:
                return feats

            feats["kalshi_available"] = True
            feats["kalshi_prob"] = prob
            feats["kalshi_home_prob"] = prob
            feats["kalshi_label"] = label
            feats["kalshi_volume"] = market_info.get("kalshi_volume") if isinstance(market_info, dict) else None
            
            # Simple Alignment Check
            model_p = game.get("implied_home_prob") or game.get("win_prob")
            if model_p is not None:
                try:
                    model_p = float(model_p)
                    if abs(model_p - prob) < 0.05:
                        feats["kalshi_alignment"] = "Neutral"
                    elif model_p > prob:
                        feats["kalshi_alignment"] = "Model > Kalshi"
                    else:
                        feats["kalshi_alignment"] = "Kalshi > Model"
                except:
                    pass

            return feats

        except Exception as e:
            logger.error(f"Kalshi feature error: {e}", exc_info=True)
            feats["kalshi_match_debug"] = f"error={str(e)}"
            return feats

    # ... (Rest of file: _calculate_derived_features, build_vertex_feature_vector, etc.) ...
    # Ensure you keep the rest of your original logic.
    # The critical fix was moving the game_dt definition up.
    
    # [Rest of file omitted for brevity, it relies on the fix above]
    # You can keep your existing build_comprehensive_features and subsequent methods
    # provided they use the fixed _get_kalshi_features above.

    # ... (Include the rest of your original file here) ...

    # Helper needed for analyze_all_games
    def build_comprehensive_features(
        self, game: Dict[str, Any], league: str, kalshi_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {
            "league": league,
            "sport_key": game.get("sport_key"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "game_time": game.get("commence_time"),
        }
        features.update(self._get_market_odds_features(game))
        features.update(self._get_team_stats_features(game, league))
        features.update(self._get_form_features(game, league))
        features.update(self._get_sentiment_features(game))
        features.update(self._get_local_ml_features(game))
        features.update(self._get_theover_features(game))
        features.update(self._get_kalshi_features(game, kalshi_info)) # Uses fixed method
        features.update(self._calculate_derived_features(features))
        return features

    # Ensure other methods like _get_market_odds_features, _get_team_stats_features 
    # are preserved from your original file.
    
    # ... (Please make sure you have the rest of your file content) ...
