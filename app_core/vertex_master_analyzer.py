"""
Vertex AI Master Analyzer
Consolidates ALL data sources for ultimate best bet recommendations.
Rewritten clean version – no Anthropic, uses Google Gemini (Vertex AI)
plus spread-derived and fallback logic.
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
<<<<<<< HEAD:app_core/vertex_master_analyzer.py
    fetch_kalshi_for_game,
=======
>>>>>>> origin/main:vertex_master_analyzer.py
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


# -------------------------------
# Normalization helpers
# -------------------------------


def normalize_team(name: str) -> str:
    """Lowercase, strip punctuation, and smooth common variations for matching."""

    if not isinstance(name, str):
        return ""

    s = name.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = s.replace("st ", "state ").replace("st.", "state ")
    s = s.replace("univ", "university")
    s = s.replace("university of ", "")
    s = s.replace("ucla bruins", "ucla").replace("ucla ", "ucla")
    s = s.replace("san jose st", "san jose state")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_league(value: str) -> str:
    """Normalize sport/league labels for matching between datasets."""

    if not isinstance(value, str):
        return ""

    v = value.lower().strip()
    if "ncaab" in v or "ncaam" in v or "college basketball" in v or "ncaa" in v:
        return "ncaab"
    if "ncaaf" in v or "college football" in v:
        return "ncaaf"
    if "nba" in v:
        return "nba"
    if "nhl" in v:
        return "nhl"
    return v


def split_game(game: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (home, away) parsed from strings like 'Away @ Home'."""

    if not isinstance(game, str) or "@" not in game:
        return None, None
    away, home = [g.strip() for g in game.split("@", 1)]
    return home, away


# Fuzzy/team matching thresholds used across Kalshi + TheOver enrichment
TEAM_FUZZY_THRESHOLD = 0.8
MAX_LINE_DIFF = 1.5  # allowable difference in spread/total when picking closest match
BEST_PICK_PRIORITY_EDGE = 1  # primary sort on edge, then EV
DEBUG_FORCE_KALSHI = False  # set to False in production to use only real Kalshi matches


def is_today_calendar_day(time_str) -> bool:
    """Return True if the given ISO timestamp is on today's calendar day in US/Eastern."""
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


# ---------------------------------------------------------------------------
# Helper functions (module-level)
# ---------------------------------------------------------------------------

def american_to_decimal(odds: Optional[float]) -> Optional[float]:
    """Convert American odds to decimal. Returns None if odds is falsy."""
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
    """Convert American odds to implied win probability."""
    dec = american_to_decimal(odds)
    if dec is None or dec <= 1.0:
        return None
    return 1.0 / dec


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------


class VertexMasterAnalyzer:
    """Master analyzer that combines ALL data sources and uses Vertex AI."""

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
        # Allow UI/ENV to disable Kalshi without removing the integrator instance
        self.use_kalshi = bool(use_kalshi and st.session_state.get("kalshi_enabled", True))

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def build_comprehensive_features(
        self, game: Dict[str, Any], league: str, kalshi_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a comprehensive feature dict for a game.

        The incoming *game* object is already pre-processed in streamlit_app
        using TheOver.ai data + TheOddsAPI where available.
        We treat those fields as ground truth and avoid additional API calls
        inside this module – that keeps this analyzer side-effect free.
        """
        features: Dict[str, Any] = {
            "league": league,
            "sport_key": game.get("sport_key"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "game_time": game.get("commence_time"),
        }

        # 1) Market odds / spread / totals
        features.update(self._get_market_odds_features(game))

        # 2) Team statistics (from SportsData.io if configured)
        features.update(self._get_team_stats_features(game, league))

        # 3) Recent form
        features.update(self._get_form_features(game, league))

        # 4) Sentiment
        features.update(self._get_sentiment_features(game))

        # 5) Local ML model (if attached)
        features.update(self._get_local_ml_features(game))

        # 6) TheOver.ai spread / total picks
        features.update(self._get_theover_features(game))

        # 7) Kalshi prediction market validation
        features.update(self._get_kalshi_features(game, kalshi_info))

        # 8) Derived / engineered features
        features.update(self._calculate_derived_features(features))

        return features

    # ---- market odds -------------------------------------------------

    def _get_market_odds_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market odds features.

        Priority order:
        1) Game fields prepared in streamlit_app from TheOver.ai spreads.
        2) Raw bookmaker data from TheOddsAPI (if present).
        """
        bookmakers = game.get("bookmakers", []) or []

        feats: Dict[str, Any] = {
            "home_ml_odds": game.get("home_ml_odds"),
            "away_ml_odds": game.get("away_ml_odds"),
            "home_spread": game.get("home_spread"),
            "away_spread": game.get("away_spread"),
            "home_spread_odds": game.get("home_spread_odds"),
            "away_spread_odds": game.get("away_spread_odds"),
            "total_line": game.get("total_line") or game.get("theover_total"),
            "over_odds": game.get("over_odds"),
            "under_odds": game.get("under_odds"),
            "num_bookmakers": len(bookmakers),
            "theover_probability": game.get("theover_probability"),
            "theover_pick": game.get("theover_pick"),
            "theover_spread": game.get("theover_spread"),
            "theover_total": game.get("theover_total"),
            "theover_total_pick": game.get("theover_total_pick"),
            "theover_total_probability": game.get("theover_total_probability"),
            "implied_home_prob": game.get("implied_home_prob"),
            # Novig-specific odds
            "novig_home_spread": None,
            "novig_away_spread": None,
            "novig_home_spread_odds": None,
            "novig_away_spread_odds": None,
        }

        # If we have explicit away_spread missing but know home_spread,
        # infer the away side so math stays consistent.
        if feats["away_spread"] is None and feats["home_spread"] is not None:
            try:
                feats["away_spread"] = -float(feats["home_spread"])
            except Exception:
                feats["away_spread"] = None

        # If bookmaker data is present, use it to fill missing odds/spreads.
        if bookmakers:
            try:
                for bookmaker in bookmakers:
                    book_name = bookmaker.get("title", "").lower()
                    
                    for market in bookmaker.get("markets", []):
                        if market.get("key") == "h2h":
                            for outcome in market.get("outcomes", []):
                                name = outcome.get("name")
                                price = outcome.get("price")
                                if name == game.get("home_team") and feats["home_ml_odds"] is None:
                                    feats["home_ml_odds"] = price
                                elif name == game.get("away_team") and feats["away_ml_odds"] is None:
                                    feats["away_ml_odds"] = price

                        elif market.get("key") == "spreads":
                            for outcome in market.get("outcomes", []):
                                name = outcome.get("name")
                                point = outcome.get("point")
                                price = outcome.get("price")
                                
                                # Extract Novig spreads specifically (Novig's API key is "lowvig")
                                if "novig" in book_name or "lowvig" in book_name:
                                    if name == game.get("home_team"):
                                        feats["novig_home_spread"] = point
                                        feats["novig_home_spread_odds"] = price
                                    elif name == game.get("away_team"):
                                        feats["novig_away_spread"] = point
                                        feats["novig_away_spread_odds"] = price
                                
                                # General spread extraction (any bookmaker)
                                if name == game.get("home_team") and feats["home_spread"] is None:
                                    feats["home_spread"] = point
                                    feats["home_spread_odds"] = price
                                elif name == game.get("away_team") and feats.get("away_spread") is None:
                                    feats["away_spread"] = point
                                    feats["away_spread_odds"] = price

                        elif market.get("key") == "totals" and feats["total_line"] is None:
                            for outcome in market.get("outcomes", []):
                                point = outcome.get("point")
                                name = outcome.get("name")
                                price = outcome.get("price")
                                feats["total_line"] = point
                                if name == "Over":
                                    feats["over_odds"] = price
                                else:
                                    feats["under_odds"] = price
            except Exception as e:
                logger.warning(f"Error parsing bookmaker odds: {e}")

        # Final sanity check on spreads
        try:
            hs = float(feats["home_spread"]) if feats["home_spread"] is not None else None
            as_ = float(feats["away_spread"]) if feats["away_spread"] is not None else None
            if hs is not None and as_ is not None:
                if abs(hs + as_) > 1.0:
                    logger.warning(
                        "Unusual spread pair: %s spread=%s, %s spread=%s (sum=%s)",
                        game.get("home_team", "Home"),
                        hs,
                        game.get("away_team", "Away"),
                        as_,
                        hs + as_,
                    )
        except Exception:
            pass

        return feats

    # ---- team stats / form / sentiment / local ML --------------------

    def _get_team_stats_features(self, game: Dict[str, Any], league: str) -> Dict[str, Any]:
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        client = self.sportsdata.get(league.lower()) if self.sportsdata else None
        if client and hasattr(client, "get_team_stats"):
            try:
                home_stats = client.get_team_stats(home_team) or {}
                away_stats = client.get_team_stats(away_team) or {}
            except Exception as e:
                logger.warning(f"SportsData get_team_stats error: {e}")
                home_stats, away_stats = {}, {}
        else:
            home_stats, away_stats = {}, {}

        def safe(stats: Dict[str, Any], key: str, default: Any) -> Any:
            val = stats.get(key)
            return default if val is None else val

        return {
            "home_win_pct": safe(home_stats, "win_pct", 0.5),
            "away_win_pct": safe(away_stats, "win_pct", 0.5),
            "home_avg_points": safe(home_stats, "avg_points", 100.0),
            "away_avg_points": safe(away_stats, "avg_points", 100.0),
            "home_avg_points_allowed": safe(home_stats, "avg_points_allowed", 100.0),
            "away_avg_points_allowed": safe(away_stats, "avg_points_allowed", 100.0),
            "home_off_rating": safe(home_stats, "off_rating", 100.0),
            "away_off_rating": safe(away_stats, "off_rating", 100.0),
            "home_def_rating": safe(home_stats, "def_rating", 100.0),
            "away_def_rating": safe(away_stats, "def_rating", 100.0),
        }

    def _get_form_features(self, game: Dict[str, Any], league: str) -> Dict[str, Any]:
        team = game.get("home_team")

        client = self.sportsdata.get(league.lower()) if self.sportsdata else None
        recent_home = {}
        recent_away = {}
        if client and hasattr(client, "get_recent_games"):
            try:
                recent_home = client.get_recent_games(game.get("home_team"), 5) or {}
                recent_away = client.get_recent_games(game.get("away_team"), 5) or {}
            except Exception as e:
                logger.warning(f"SportsData get_recent_games error: {e}")

        def safe(stats: Dict[str, Any], key: str, default: Any) -> Any:
            val = stats.get(key)
            return default if val is None else val

        return {
            "home_last_5_wins": safe(recent_home, "wins", 2),
            "away_last_5_wins": safe(recent_away, "wins", 2),
        }

    def _get_sentiment_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        def synthetic(team: Optional[str]) -> float:
            if not team:
                return 0.0
            team_hash = sum(ord(c) for c in team[:8]) % 100
            return (team_hash - 50) / 200.0  # -0.25 .. +0.25

        def get_for_team(team: str) -> float:
            if not self.sentiment:
                return synthetic(team)
            try:
                if hasattr(self.sentiment, "get_team_sentiment"):
                    data = self.sentiment.get_team_sentiment(team)
                    if isinstance(data, dict):
                        score = float(data.get("sentiment_score", 0.0))
                        if score != 0.0:
                            return score
                    elif isinstance(data, (int, float)):
                        if float(data) != 0.0:
                            return float(data)
                if hasattr(self.sentiment, "analyze_team"):
                    res = self.sentiment.analyze_team(team)
                    if res and res.get("score", 0.0) != 0.0:
                        return float(res.get("score", 0.0))
            except Exception as e:
                logger.warning(f"Sentiment error for {team}: {e}")
            return synthetic(team)

        home_sent = get_for_team(home_team)
        away_sent = get_for_team(away_team)

        return {
            "home_sentiment": home_sent,
            "away_sentiment": away_sent,
            "sentiment_diff": home_sent - away_sent,
        }

    def _get_local_ml_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        if not self.local_ml:
            return {"local_ml_prob": None, "local_ml_confidence": 0.0}

        try:
            pred = self.local_ml.predict_game_outcome(
                home_team=game.get("home_team"),
                away_team=game.get("away_team"),
                sport_key=game.get("sport_key"),
            )
            return {
                "local_ml_prob": float(pred.get("home_win_prob")) if pred.get("home_win_prob") is not None else None,
                "local_ml_confidence": float(pred.get("confidence", 0.0)),
            }
        except Exception as e:
            logger.warning(f"Local ML prediction error: {e}")
            return {"local_ml_prob": None, "local_ml_confidence": 0.0}

    # ---- TheOver.ai + Kalshi -----------------------------------------

    def _get_theover_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Attach TheOver.ai data to a game with forgiving fuzzy matching."""

        def sf(val: Any, default: float) -> float:
            try:
                if val is None or pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        base = {
            "theover_has_pick": 1 if game.get("theover_probability") is not None else 0,
            "theover_pick": game.get("theover_pick") or "",
            "theover_probability": sf(game.get("theover_probability"), np.nan),
            "theover_spread": sf(game.get("theover_spread", 0.0), 0.0),
            "theover_total": sf(game.get("theover_total", 0.0), 0.0),
            "theover_total_pick": game.get("theover_total_pick") or "",
            "theover_total_probability": sf(game.get("theover_total_probability"), np.nan),
            "theover_match_debug": "prepopulated" if game.get("theover_probability") is not None else "no_match_found",
        }

        # If we already have a valid probability, keep it
        if not pd.isna(base["theover_probability"]):
            return base

        # Nothing pre-populated; try to match against uploaded TheOver data
        if not self.theover:
            base["theover_match_debug"] = "no_theover_dataset"
            return base

        def _teams_similarity(home: str, away: str, candidate_home: str, candidate_away: str) -> float:
            h_score = TeamNameMatcher.similarity_score(
                TeamNameMatcher.normalize(home), TeamNameMatcher.normalize(candidate_home)
            )
            a_score = TeamNameMatcher.similarity_score(
                TeamNameMatcher.normalize(away), TeamNameMatcher.normalize(candidate_away)
            )
            return min(h_score, a_score)

        def _choose_best_match(df: pd.DataFrame, market_type: str, target_line: Optional[float]) -> Optional[pd.Series]:
            best_row = None
            best_score = 0.0
            best_line_diff = 999.0
            for _, row in df.iterrows():
                sim = _teams_similarity(
                    game.get("home_team", ""),
                    game.get("away_team", ""),
                    row.get("HomeTeam") or row.get("home_team", ""),
                    row.get("AwayTeam") or row.get("away_team", ""),
                )
                if sim < TEAM_FUZZY_THRESHOLD:
                    continue

                # Pick closest line if we have one; otherwise prioritise similarity only
                cand_line = row.get("Line") if "Line" in row else row.get("line")
                try:
                    cand_line_val = float(cand_line) if cand_line not in (None, "", "-") else None
                except Exception:
                    cand_line_val = None

                line_diff = abs((target_line or 0) - cand_line_val) if (target_line is not None and cand_line_val is not None) else None
                if line_diff is not None and line_diff > MAX_LINE_DIFF:
                    continue

                score_tuple = (sim, -(line_diff or 0))
                if sim > best_score or (sim == best_score and (line_diff or 999) < best_line_diff):
                    best_score = sim
                    best_row = row
                    best_line_diff = line_diff if line_diff is not None else best_line_diff

            return best_row

        # Try spreads first, then totals
        match_row = None
        match_type = ""
        line_hint = game.get("home_spread") or game.get("theover_spread")
        spreads_df = self.theover.get("spreads")
        totals_df = self.theover.get("totals")

        if spreads_df is not None and not spreads_df.empty:
            match_row = _choose_best_match(spreads_df, "Spread", line_hint)
            match_type = "Spread" if match_row is not None else ""

        if match_row is None and totals_df is not None and not totals_df.empty:
            line_hint = game.get("total_line") or game.get("theover_total")
            match_row = _choose_best_match(totals_df, "Total", line_hint)
            match_type = "Total" if match_row is not None else ""

        if match_row is None:
            base["theover_match_debug"] = "no_match_found"
            return base

        pick = match_row.get("Pick") or match_row.get("pick") or ""
        line_val = match_row.get("Line") if "Line" in match_row else match_row.get("line")
        try:
            line_val_f = float(line_val) if line_val not in (None, "", "-") else None
        except Exception:
            line_val_f = None

        prob_col = match_row.get("Probability") if "Probability" in match_row else match_row.get("probability")
        try:
            prob_val = float(prob_col) if prob_col not in (None, "", "-") else None
        except Exception:
            prob_val = None

        # If probability missing, derive from spread/total magnitude
        if prob_val is None and line_val_f is not None:
            prob_val = 0.5 + min(0.35, abs(line_val_f) * 0.028)

        # Determine if pick refers to home team
        pick_is_home = False
        if pick:
            pick_is_home = TeamNameMatcher.similarity_score(
                TeamNameMatcher.normalize(pick), TeamNameMatcher.normalize(game.get("home_team", ""))
            ) >= TEAM_FUZZY_THRESHOLD

        if prob_val is not None:
            prob_val = max(0.0, min(1.0, prob_val))
            # Totals: probability refers directly to Over/Under pick instead of home/away
            if match_type == "Total":
                over_prob = prob_val if str(pick).lower().startswith("over") else (1.0 - prob_val)
                base.update(
                    {
                        "theover_has_pick": 1,
                        "theover_pick": base.get("theover_pick", ""),
                        "theover_probability": base.get("theover_probability", np.nan),
                        "theover_total": line_val_f if match_type == "Total" else base.get("theover_total", 0.0),
                        "theover_total_pick": pick,
                        "theover_total_probability": over_prob,
                        "theover_match_debug": f"match={match_type}, line_diff={(line_hint - line_val_f) if (line_hint is not None and line_val_f is not None) else 'n/a'}, pick={pick}",
                    }
                )
            else:
                home_prob = prob_val if pick_is_home else (1.0 - prob_val)
                base.update(
                    {
                        "theover_has_pick": 1,
                        "theover_pick": pick,
                        "theover_probability": home_prob,
                        "theover_spread": line_val_f if match_type == "Spread" else base.get("theover_spread", 0.0),
                        "theover_total": line_val_f if match_type == "Total" else base.get("theover_total", 0.0),
                        "theover_total_pick": base.get("theover_total_pick", ""),
                        "theover_total_probability": base.get("theover_total_probability", np.nan),
                        "theover_match_debug": f"match={match_type}, line_diff={(line_hint - line_val_f) if (line_hint is not None and line_val_f is not None) else 'n/a'}, pick={pick}",
                    }
                )
        else:
            base["theover_match_debug"] = "matched_no_probability"

        return base

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

        game_id = game.get("game_id") or game.get("id") or game.get("matchup") or game.get("game")
        league = game.get("league") or game.get("sport") or game.get("sport_key")
        home_team = game.get("home_team") or game.get("home") or game.get("home_name")
        away_team = game.get("away_team") or game.get("away") or game.get("away_name")
        game_time = game.get("commence_time") or game.get("game_time") or game.get("Game Time Raw")

        logging.info(
            f"[Kalshi FEATURES] league={league} home={home_team} away={away_team} "
            f"time={game_time} use_kalshi={self.use_kalshi} integrator_none={self.kalshi is None}"
        )

        # If Kalshi is not configured, bail out cleanly
        if not getattr(self, "kalshi", None) or not getattr(self, "use_kalshi", True):
            feats["kalshi_match_debug"] = "kalshi_not_configured" if not getattr(self, "kalshi", None) else "kalshi_disabled"
            feats["kalshi_status"] = feats["kalshi_match_debug"]
            return feats

        try:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            league = game.get("league") or game.get("sport_key") or "NBA"

            market_info = prefetch_info
<<<<<<< HEAD:app_core/vertex_master_analyzer.py
            game_time = game.get("commence_time") or game.get("game_time")
            game_dt = None
            if game_time:
                try:
                    game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
                except Exception:
                    game_dt = None

            if market_info is None:
                market_info = fetch_kalshi_for_game(
=======
            if market_info is None:
                game_time = game.get("commence_time") or game.get("game_time")
                game_dt = None
                if game_time:
                    try:
                        game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
                    except Exception:
                        game_dt = None

                # Delegate to the integrator as the single source of truth
                market_info = match_game_to_kalshi(
                    league,
>>>>>>> origin/main:vertex_master_analyzer.py
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

            if prob is None:
                return feats

            try:
                prob = float(prob)
            except Exception:
                feats["kalshi_match_debug"] = f"invalid_prob={prob}"
                feats["kalshi_status"] = feats["kalshi_match_debug"]
                feats["kalshi_available"] = False
                feats["kalshi_prob"] = None
                return feats

            prob = max(0.0, min(1.0, prob))

            feats["kalshi_available"] = True
            feats["kalshi_prob"] = prob
            feats["kalshi_home_prob"] = prob
            feats["kalshi_label"] = label or (market_info.get("kalshi_label") if isinstance(market_info, dict) else None)
            feats["kalshi_volume"] = market_info.get("kalshi_volume") if isinstance(market_info, dict) else None
            feats["kalshi_confidence"] = market_info.get("confidence") if isinstance(market_info, dict) else None
            if not is_match_result:
                feats["kalshi_match_debug"] = market_info.get("kalshi_match_debug", feats.get("kalshi_match_debug")) if isinstance(market_info, dict) else feats.get("kalshi_match_debug")
                feats["kalshi_status"] = market_info.get("kalshi_match_debug", feats.get("kalshi_status")) if isinstance(market_info, dict) else feats.get("kalshi_status")

            logging.info(
                f"[Kalshi RESULT] home={home_team} away={away_team} "
                f"available={feats['kalshi_available']} prob={feats['kalshi_prob']} "
                f"debug={feats['kalshi_match_debug']}"
            )

            model_p = game.get("implied_home_prob") or game.get("win_prob")
            try:
                model_p = float(model_p) if model_p is not None else None
            except Exception:
                model_p = None

            if model_p is not None:
                if abs(model_p - prob) < 0.05:
                    feats["kalshi_alignment"] = "Neutral"
                elif model_p > prob:
                    feats["kalshi_alignment"] = "Model > Kalshi"
                else:
                    feats["kalshi_alignment"] = "Kalshi > Model"

            return feats

        except Exception as e:
            logger.warning(
                "Kalshi _get_kalshi_features error for %s vs %s: %s",
                game.get("home_team"),
                game.get("away_team"),
                e,
            )
            feats["kalshi_match_debug"] = f"error={type(e).__name__}: {e}"
            feats["kalshi_available"] = False
            feats["kalshi_prob"] = None
            return feats

    # ---- Derived features --------------------------------------------

    def _calculate_derived_features(self, feats: Dict[str, Any]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        def sg(key: str, default: float) -> float:
            val = feats.get(key, default)
            try:
                if val is None or pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        home_win = sg("home_win_pct", 0.5)
        away_win = sg("away_win_pct", 0.5)
        d["win_pct_diff"] = home_win - away_win

        d["off_def_matchup_home"] = sg("home_off_rating", 100.0) - sg("away_def_rating", 100.0)
        d["off_def_matchup_away"] = sg("away_off_rating", 100.0) - sg("home_def_rating", 100.0)

        d["form_momentum_diff"] = sg("home_last_5_wins", 2.0) - sg("away_last_5_wins", 2.0)

        # Implied prob from ML odds if not already provided
        implied_home = feats.get("implied_home_prob")
        if implied_home is None:
            implied_home = implied_prob_from_american(feats.get("home_ml_odds"))
        d["implied_home_prob"] = float(implied_home) if implied_home is not None else None

        # Consensus blend
        probs = [
            d.get("implied_home_prob"),
            feats.get("local_ml_prob"),
            feats.get("theover_probability"),
        ]
        if feats.get("kalshi_available"):
            kp = feats.get("kalshi_prob")
            probs.extend([kp, kp])  # double weight Kalshi when present
        valid = [p for p in probs if p is not None and not pd.isna(p)]
        d["consensus_prob"] = float(np.mean(valid)) if valid else None

        d["kalshi_validation_score"] = (
            1.0 - abs(feats.get("kalshi_prob") - d["consensus_prob"])
            if feats.get("kalshi_available") and d.get("consensus_prob") is not None and feats.get("kalshi_prob") is not None
            else None
        )

        return d

    # ------------------------------------------------------------------
    # Vertex AI / Gemini analysis
    # ------------------------------------------------------------------

    def build_vertex_feature_vector(self, feats: Dict[str, Any]) -> List[float]:
        """Map comprehensive features -> numeric vector for Vertex."""

        row = self.build_vertex_feature_row(feats)
        cleaned: List[float] = []
        for col in VERTEX_FEATURE_COLUMNS:
            val = row.get(col, 0.0)
            try:
                cleaned.append(float(val))
            except Exception:
                cleaned.append(0.0)
        return cleaned

    def build_vertex_feature_row(self, feats: Dict[str, Any]) -> Dict[str, float]:
        """Construct the ordered feature row expected by the deployed Vertex model."""

        def gv(key: str, default: float = 0.0) -> float:
            val = feats.get(key, default)
            try:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return default
                return float(val)
            except Exception:
                return default

        home_avg = gv("home_avg_points", 0.0)
        away_avg = gv("away_avg_points", 0.0)
        win_pct_diff = gv("win_pct_diff", gv("home_win_pct", 0.5) - gv("away_win_pct", 0.5))
        form_diff = gv("home_last_5_wins", 0.0) - gv("away_last_5_wins", 0.0)
        consensus = gv("consensus_prob", gv("implied_home_prob", 0.5))
        spread_norm = gv("home_spread", 0.0)

        row = {
            "home_win_pct": gv("home_win_pct", 0.5),
            "away_win_pct": gv("away_win_pct", 0.5),
            "home_avg_points": home_avg,
            "away_avg_points": away_avg,
            "home_form_last5": gv("home_last_5_wins", 0.0),
            "away_form_last5": gv("away_last_5_wins", 0.0),
            "home_pd_last5": gv("home_pd_last5", 0.0),
            "away_pd_last5": gv("away_pd_last5", 0.0),
            "home_sos_last5": gv("home_sos_last5", 0.0),
            "away_sos_last5": gv("away_sos_last5", 0.0),
            "win_pct_diff": win_pct_diff,
            "avg_points_diff": home_avg - away_avg,
            "form_diff": form_diff,
            "pd_last5_diff": gv("pd_last5_diff", gv("home_pd_last5", 0.0) - gv("away_pd_last5", 0.0)),
            "sos_diff": gv("sos_diff", gv("home_sos_last5", 0.0) - gv("away_sos_last5", 0.0)),
            "def_rating_diff": gv("def_rating_diff", gv("home_def_rating", 0.0) - gv("away_def_rating", 0.0)),
            "streak_diff": gv("streak_diff", gv("home_streak", 0.0) - gv("away_streak", 0.0)),
            "spread_normalized": spread_norm,
            "public_betting_centered": gv("public_betting_centered", gv("sentiment_diff", 0.0)),
            "sharp_vs_public": gv("sharp_vs_public", 0.0),
            "rest_advantage": gv("rest_advantage", 0.0),
            "back_to_back": gv("back_to_back", 0.0),
            "primetime_game": gv("primetime_game", 0.0),
            "division_game": gv("division_game", 0.0),
            "injuries_impact": gv("injuries_impact", 0.0),
            "weather_factor": gv("weather_factor", 0.0),
            "line_movement": gv("line_movement", 0.0),
            "total_movement": gv("total_movement", 0.0),
            "model_consensus": consensus,
            "theover_probability": gv("theover_probability", 0.5),
        }

        return row

    # ---- candidate helpers ------------------------------------------

    def _derive_home_probability(self, feats: Dict[str, Any]) -> Optional[float]:
        """Estimate home win probability without defaulting to 50/50."""
        implied_home_prob = feats.get("implied_home_prob")
        spread = feats.get("home_spread")
        consensus = feats.get("consensus_prob")

        try:
            if spread is not None and not pd.isna(spread):
                spread_val = float(spread)
                return max(0.15, min(0.85, 0.5 - spread_val * 0.028))
            if implied_home_prob is not None:
                return float(implied_home_prob)
            if consensus is not None:
                return float(consensus)
        except Exception:
            return None
        return None

    def _spread_cover_probability(self, feats: Dict[str, Any]) -> Optional[float]:
        """Heuristic for the probability the HOME side covers the spread."""
        spread = feats.get("home_spread")
        try:
            if spread is None or pd.isna(spread):
                return None
            spread_val = float(spread)
            return max(0.1, min(0.9, 0.5 - spread_val * 0.028))
        except Exception:
            return None

    def _heuristic_probability(
        self, market_type: str, selection: str, feats: Dict[str, Any]
    ) -> Optional[float]:
        """Fallback probability estimates per market type without using 0.5 defaults."""

        selection = selection.lower()
        if market_type == "ML":
            base = self._derive_home_probability(feats)
            if base is None:
                return None
            return base if selection == "home" else (1.0 - base)

        if market_type == "Spread":
            base = self._spread_cover_probability(feats)
            if base is None:
                return None
            return base if selection == "home" else (1.0 - base)

        if market_type == "Total":
            total_prob = feats.get("theover_total_probability")
            if total_prob is None or pd.isna(total_prob):
                return None
            return total_prob if selection == "over" else (1.0 - total_prob)

        return None

    def _calc_theover_support(
        self,
        feats: Dict[str, Any],
        market_type: str,
        selection: str,
        ai_prob: Optional[float],
        pick_line: Optional[float],
    ) -> Dict[str, Any]:
        """Return TheOver alignment/edge per candidate type using structured alignment labels."""

        selection = selection.lower()
        alignment = None
        prob = None
        edge = None
        debug = feats.get("theover_match_debug") or "no_match_found"

        pick_team: Optional[str] = None
        pick_side: Optional[str] = None
        pick_line: Optional[float] = None
        selection_line = pick_line

        if market_type == "Total":
            base_prob = feats.get("theover_total_probability")
            pick_label = feats.get("theover_total_pick") or ""
            pick_line = feats.get("theover_total") if feats.get("theover_total") is not None else None
            if base_prob is not None and not pd.isna(base_prob):
                pick_side = "over" if isinstance(pick_label, str) and "over" in pick_label.lower() else "under"
                prob = base_prob if selection == "over" else (1.0 - base_prob)
                if ai_prob is not None:
                    edge = ai_prob - prob

                if selection in {"over", "under"} and pick_side in {"over", "under"}:
                    if selection_line is not None and pick_line is not None:
                        if abs(float(selection_line) - float(pick_line)) <= MAX_LINE_DIFF and selection == pick_side:
                            alignment = "Aligned"
                        elif selection != pick_side:
                            alignment = "Opposite Side"
                        else:
                            alignment = "Different Line"
                    else:
                        alignment = "Aligned" if selection == pick_side else "Opposite Side"
        else:
            base_prob = feats.get("theover_probability")
            pick_label = feats.get("theover_pick") or ""
            pick_line = feats.get("theover_spread") if feats.get("theover_spread") is not None else None
            if base_prob is not None and not pd.isna(base_prob):
                is_home = selection == "home"
                pick_team = pick_label or None
                pick_norm = TeamNameMatcher.normalize(pick_label)
                home_norm = TeamNameMatcher.normalize(feats.get("home_team") or "")
                away_norm = TeamNameMatcher.normalize(feats.get("away_team") or "")
                if pick_norm and home_norm and TeamNameMatcher.similarity_score(pick_norm, home_norm) >= TEAM_FUZZY_THRESHOLD:
                    pick_side = "home"
                elif pick_norm and away_norm and TeamNameMatcher.similarity_score(pick_norm, away_norm) >= TEAM_FUZZY_THRESHOLD:
                    pick_side = "away"

                prob = base_prob if is_home else (1.0 - base_prob)
                if ai_prob is not None:
                    edge = ai_prob - prob

                if selection in {"home", "away"} and pick_side in {"home", "away"}:
                    if selection_line is not None and pick_line is not None:
                        if abs(float(selection_line) - float(pick_line)) <= MAX_LINE_DIFF and selection == pick_side:
                            alignment = "Aligned"
                        elif selection != pick_side:
                            alignment = "Opposite Side"
                        else:
                            alignment = "Different Line"
                    else:
                        alignment = "Aligned" if selection == pick_side else "Opposite Side"

        if alignment is None:
            alignment = "Unknown" if prob is not None else "No Coverage"

        return {
            "theover_probability_for_pick": prob,
            "theover_edge": edge,
            "theover_alignment": alignment,
            "theover_match_debug": debug,
            "theover_pick_team": pick_team,
            "theover_pick_side": pick_side,
            "theover_pick_line": pick_line,
        }

    def _format_pick_text(
        self,
        market_type: str,
        selection: str,
        line: Optional[float],
        home_team: str,
        away_team: str,
    ) -> str:
        """Human readable pick string that encodes market type and line."""

        if market_type == "ML":
            team = home_team if selection == "home" else away_team
            return f"{team} ML"
        if market_type == "Spread":
            team = home_team if selection == "home" else away_team
            if line is None:
                return f"{team} Spread"
            sign = "+" if line > 0 else ""
            return f"{team} {sign}{line:.1f} Spread"
        if market_type == "Total":
            if line is None:
                return f"{selection.title()} Total"
            return f"{selection.title()} {line:.1f} Total"
        return selection

    def _evaluate_candidate(
        self,
        feats: Dict[str, Any],
        market_type: str,
        selection: str,
        line: Optional[float],
        odds: Optional[float],
        game_league: str,
        vertex_enabled: bool,
        numeric_vec: List[float],
        vertex_home_prob: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a single candidate bet with AI + market metrics."""

        if odds is None or pd.isna(odds):
            return None

        market_prob = implied_prob_from_american(odds)
        if market_prob is None:
            return None

        home_team = feats.get("home_team", "Home")
        away_team = feats.get("away_team", "Away")

        feature_payload = {
            "league": game_league,
            "home_team": home_team,
            "away_team": away_team,
            "market_type": market_type,
            "selection": selection,
            "line": line,
            "odds": odds,
            "numeric_features": numeric_vec,
            "home_ml_odds": feats.get("home_ml_odds"),
            "away_ml_odds": feats.get("away_ml_odds"),
            "home_spread": feats.get("home_spread"),
            "total_line": feats.get("total_line"),
            "implied_home_prob": feats.get("implied_home_prob"),
            "theover_probability": feats.get("theover_probability"),
            "theover_total_probability": feats.get("theover_total_probability"),
            "kalshi_prob": feats.get("kalshi_prob"),
            "local_ml_prob": feats.get("local_ml_prob"),
            "consensus_prob": feats.get("consensus_prob"),
        }

        context = (
            f"{game_league} game: {away_team} at {home_team}. "
            f"Market={market_type}, selection={selection}, line={line}, odds={odds}."
        )

        ai_prob = None
        ai_source = "heuristic"
        if vertex_enabled and vertex_home_prob is not None:
            if market_type in ("ML", "Spread"):
                ai_prob = vertex_home_prob if selection == "home" else (1.0 - vertex_home_prob)
                ai_source = "vertex_ai"

        if ai_prob is None:
            ai_prob = self._heuristic_probability(market_type, selection, feats)
            ai_source = "heuristic" if ai_prob is not None else ai_source

        if ai_prob is None:
            return None

        try:
            ai_prob = float(ai_prob)
        except Exception:
            return None

        ai_prob = max(0.0, min(1.0, ai_prob))
        edge = ai_prob - market_prob

        dec = american_to_decimal(odds)
        ev = ai_prob * (dec - 1.0) - (1.0 - ai_prob) if dec is not None else 0.0

        pick_text = self._format_pick_text(market_type, selection, line, home_team, away_team)

        # Kalshi comparison is only meaningful for home/away style markets
        kalshi_prob = feats.get("kalshi_prob")
        kalshi_for_pick = None
        edge_vs_kalshi = None
        if kalshi_prob is not None and not pd.isna(kalshi_prob) and market_type in ("ML", "Spread"):
            kalshi_for_pick = kalshi_prob if selection == "home" else (1.0 - kalshi_prob)
            edge_vs_kalshi = ai_prob - kalshi_for_pick

        theover_info = self._calc_theover_support(feats, market_type, selection, ai_prob, line)

        # Favorite indicator is only relevant for home/away selections
        home_ml = feats.get("home_ml_odds")
        away_ml = feats.get("away_ml_odds")
        pick_is_favorite = None
        if market_type in ("ML", "Spread"):
            home_ip = implied_prob_from_american(home_ml)
            away_ip = implied_prob_from_american(away_ml)
            if home_ip is not None and away_ip is not None:
                home_is_favorite = home_ip >= away_ip
                pick_is_favorite = home_is_favorite if selection == "home" else (not home_is_favorite)

        return {
            "league": game_league,
            "game": f"{away_team} @ {home_team}",
            "home_team": home_team,
            "away_team": away_team,
            "the_pick": pick_text,
            "pick_team": home_team if selection == "home" else (away_team if selection == "away" else selection.title()),
            "pick_selection": selection,
            "pick_spread": line if market_type == "Spread" else np.nan,
            "pick_market_type": market_type,
            "pick_line": line,
            "pick_odds": odds,
            "win_prob": ai_prob,
            "market_prob": market_prob,
            "edge_vs_market": edge,
            "edge_vs_kalshi": edge_vs_kalshi,
            "ml_source": ai_source,
            "is_favorite": pick_is_favorite,
            "home_ml_odds": home_ml,
            "away_ml_odds": away_ml,
            "kalshi_home_prob": feats.get("kalshi_prob"),
            "kalshi_prob": kalshi_for_pick,
            "kalshi_alignment": feats.get("kalshi_alignment"),
            "kalshi_match_debug": feats.get("kalshi_match_debug", ""),
            "sentiment_diff": feats.get("sentiment_diff"),
            "ev": ev,
            "theover_edge": theover_info.get("theover_edge"),
            "theover_alignment": theover_info.get("theover_alignment"),
            "theover_match_debug": theover_info.get("theover_match_debug"),
            "theover_probability_for_pick": theover_info.get("theover_probability_for_pick"),
            "theover_pick_team": theover_info.get("theover_pick_team"),
            "theover_pick_side": theover_info.get("theover_pick_side"),
            "theover_pick_line": theover_info.get("theover_pick_line"),
            "game_time": feats.get("game_time"),
            "theover_pick": feats.get("theover_pick"),
            "theover_total_pick": feats.get("theover_total_pick"),
            "novig_home_spread": feats.get("novig_home_spread"),
            "novig_away_spread": feats.get("novig_away_spread"),
            "novig_home_spread_odds": feats.get("novig_home_spread_odds"),
            "novig_away_spread_odds": feats.get("novig_away_spread_odds"),
        }

    def analyze_all_games(self, games: List[Dict[str, Any]], league: str = "NBA") -> pd.DataFrame:
        """Run Vertex / Gemini + fallbacks across all games.

        Returns a DataFrame with one row per game.
        """
        logging.info(
            f"[VMA] run_full_master_analysis | use_kalshi={getattr(self, 'use_kalshi', None)} "
            f"kalshi_is_none={getattr(self, 'kalshi', None) is None}"
        )
        if not games:
            return pd.DataFrame()

        vertex_enabled = is_vertex_prediction_configured()

        ml_sources_used = {
            "vertex_ai": 0,
            "heuristic": 0,
        }

        kalshi_attempts = 0
        kalshi_hits = 0
        kalshi_errors = 0
        kalshi_fail_reasons: Dict[str, int] = {}

        vertex_feature_rows: List[Dict[str, Any]] = []

        rows: List[Dict[str, Any]] = []
        progress = st.progress(0)

        for idx, game in enumerate(games):
            try:
                sport_key = game.get("sport_key", "") or ""
                if "basketball_nba" in sport_key:
                    game_league = "NBA"
                elif "basketball_ncaab" in sport_key:
                    game_league = "NCAAB"
                elif "americanfootball_nfl" in sport_key or "nfl" in sport_key:
                    game_league = "NFL"
                elif "americanfootball_ncaaf" in sport_key or "ncaaf" in sport_key:
                    game_league = "NCAAF"
                elif "icehockey_nhl" in sport_key or "nhl" in sport_key:
                    game_league = "NHL"
                else:
                    game_league = league

                home_team = game.get("home_team")
                away_team = game.get("away_team")

                kalshi_info: Optional[Dict[str, Any]] = None
                if getattr(self, "kalshi", None) and getattr(self, "use_kalshi", True):
                    kalshi_attempts += 1
                    try:
                        game_time = game.get("commence_time") or game.get("game_time")
                        kalshi_info = match_game_to_kalshi(
                            game_league,
                            home_team or "",
                            away_team or "",
                            game_time,
                            integrator=self.kalshi,
                        )
                        if isinstance(kalshi_info, dict) and kalshi_info.get("matched"):
                            kalshi_hits += 1
                        else:
                            reason = None
                            if isinstance(kalshi_info, dict):
                                reason = kalshi_info.get("reason")
                            reason = reason or "no_match"
                            kalshi_fail_reasons[reason] = kalshi_fail_reasons.get(reason, 0) + 1
                    except Exception as e:
                        kalshi_errors += 1
                        reason = f"api_error:{type(e).__name__}"
                        kalshi_fail_reasons[reason] = kalshi_fail_reasons.get(reason, 0) + 1
                        print(
                            f"[Kalshi] Error for game {home_team} vs {away_team}: {e}"
                        )

                feats = self.build_comprehensive_features(
                    game, game_league, kalshi_info=kalshi_info
                )
                vec = self.build_vertex_feature_vector(feats)

                feature_row: Dict[str, Any] = {}
                try:
                    feature_row = self.build_vertex_feature_row(feats)
                except Exception as e:
                    logger.warning(
                        "Vertex feature row build failed for %s vs %s: %s",
                        feats.get("home_team"),
                        feats.get("away_team"),
                        e,
                    )

                vertex_home_prob = None
                if vertex_enabled and feature_row:
                    try:
                        probs = predict_win_probabilities(
                            pd.DataFrame([feature_row]), VERTEX_FEATURE_COLUMNS
                        )
                        if probs is not None and len(probs) > 0:
                            vertex_home_prob = float(probs[0])
                            feats["vertex_home_prob"] = vertex_home_prob
                    except Exception as e:
                        logger.warning(
                            "Vertex endpoint prediction failed for %s vs %s: %s",
                            feats.get("home_team"),
                            feats.get("away_team"),
                            e,
                        )

                candidates: List[Dict[str, Any]] = []

                home_ml = feats.get("home_ml_odds")
                away_ml = feats.get("away_ml_odds")
                for selection, odds in (("home", home_ml), ("away", away_ml)):
                    cand = self._evaluate_candidate(
                        feats,
                        market_type="ML",
                        selection=selection,
                        line=None,
                        odds=odds,
                        game_league=game_league,
                        vertex_enabled=vertex_enabled,
                        numeric_vec=vec,
                        vertex_home_prob=vertex_home_prob,
                    )
                    if cand:
                        candidates.append(cand)

                home_spread = feats.get("home_spread")
                away_spread = feats.get("away_spread")
                try:
                    if away_spread is None and home_spread is not None:
                        away_spread = -float(home_spread)
                    if home_spread is None and away_spread is not None:
                        home_spread = -float(away_spread)
                except Exception:
                    pass

                for selection, line, odds in (
                    ("home", home_spread, feats.get("home_spread_odds")),
                    ("away", away_spread, feats.get("away_spread_odds")),
                ):
                    cand = self._evaluate_candidate(
                        feats,
                        market_type="Spread",
                        selection=selection,
                        line=line,
                        odds=odds,
                        game_league=game_league,
                        vertex_enabled=vertex_enabled,
                        numeric_vec=vec,
                        vertex_home_prob=vertex_home_prob,
                    )
                    if cand:
                        candidates.append(cand)

                total_line = feats.get("total_line")
                over_odds = feats.get("over_odds")
                under_odds = feats.get("under_odds")
                for selection, odds in (("over", over_odds), ("under", under_odds)):
                    cand = self._evaluate_candidate(
                        feats,
                        market_type="Total",
                        selection=selection,
                        line=total_line,
                        odds=odds,
                        game_league=game_league,
                        vertex_enabled=vertex_enabled,
                        numeric_vec=vec,
                        vertex_home_prob=vertex_home_prob,
                    )
                    if cand:
                        candidates.append(cand)

                valid_candidates = [
                    c
                    for c in candidates
                    if c.get("win_prob") is not None and c.get("market_prob") is not None
                ]

                if not valid_candidates:
                    logger.warning(
                        "Skipping game %s vs %s - no usable candidates", away_team, home_team
                    )
                    continue

                def _sort_key(cand: Dict[str, Any]):
                    edge_val = cand.get("edge_vs_market")
                    ev_val = cand.get("ev", -999)
                    return (edge_val if edge_val is not None else -999, ev_val)

                best_candidate = sorted(valid_candidates, key=_sort_key, reverse=True)[0]
                # Merge explicit Kalshi features for this game so downstream tables/export always
                # have kalshi_* columns populated even when the winning pick is a Total and the
                # candidate-level kalshi_prob was left unset.
                kalshi_feats: Dict[str, Any] = {}
                try:
                    kalshi_feats = self._get_kalshi_features(game, prefetch_info=kalshi_info) or {}
                except Exception as e:
                    logger.warning(
                        "Kalshi feature error for game %s vs %s: %s",
                        game.get("home_team"),
                        game.get("away_team"),
                        e,
                    )
                    kalshi_feats = {
                        "kalshi_available": False,
                        "kalshi_prob": None,
                        "kalshi_alignment": None,
                        "kalshi_match_debug": f"error={type(e).__name__}",
                    }

                for k, v in kalshi_feats.items():
                    best_candidate[k] = v

                # Ensure pick-level Kalshi alignment is filled when a game-level match exists
                if kalshi_feats.get("kalshi_available") and kalshi_feats.get("kalshi_prob") is not None:
                    pick_prob = (
                        kalshi_feats.get("kalshi_prob")
                        if best_candidate.get("pick_selection") == "home"
                        else 1.0 - kalshi_feats.get("kalshi_prob")
                    )
                    best_candidate["kalshi_prob"] = pick_prob
                    best_candidate["edge_vs_kalshi"] = (
                        best_candidate.get("win_prob") - pick_prob
                        if best_candidate.get("win_prob") is not None
                        else None
                    )
                    best_candidate.setdefault("kalshi_label", kalshi_feats.get("kalshi_label"))
                # Track source usage for the winning candidate
                src = best_candidate.get("ml_source")
                if src in ml_sources_used:
                    ml_sources_used[src] += 1

                best_candidate["vertex_feature_row"] = feature_row or {}
                vertex_feature_rows.append(feature_row or {})
                rows.append(best_candidate)
            except Exception as e:
                logger.error(f"Error analyzing game: {e}", exc_info=True)

            progress.progress((idx + 1) / len(games))

        df = pd.DataFrame(rows)
        df.attrs["ml_sources_used"] = ml_sources_used

        ai_status = "fallback:not_run"
        ai_model = VERTEX_MODEL_DISPLAY_NAME
        if not df.empty:
            feature_df = pd.DataFrame(vertex_feature_rows, index=df.index)
            probs_series, ai_status, ai_model = score_with_vertex(feature_df, VERTEX_FEATURE_COLUMNS)
            df["AI Win %"] = (probs_series * 100).round(1)
            df["ai_status"] = ai_status
            df["AI Model"] = ai_model
            logging.info(
                "[Vertex] scoring result | status=%s model=%s valid_probs=%s",
                ai_status,
                ai_model,
                probs_series.notna().sum(),
            )
            print(
                f"[Vertex] status={ai_status}, model={ai_model}, n_rows={len(df)}, n_valid_probs={df['AI Win %'].notna().sum()}"
            )

            if ai_status == "vertex":
                df["win_prob"] = probs_series
                df["edge_vs_market"] = pd.to_numeric(df["win_prob"], errors="coerce") - pd.to_numeric(
                    df.get("market_prob"), errors="coerce"
                )
                if "kalshi_prob" in df.columns:
                    df["edge_vs_kalshi"] = df.apply(
                        lambda r: r.get("win_prob") - r.get("kalshi_prob")
                        if r.get("kalshi_prob") is not None and not pd.isna(r.get("kalshi_prob"))
                        else np.nan,
                        axis=1,
                    )
                df["ml_source"] = "vertex_ai"

        if "AI Win %" not in df.columns:
            df["AI Win %"] = pd.Series(np.nan, index=df.index)
        if "ai_status" not in df.columns:
            df["ai_status"] = ai_status
        if "AI Model" not in df.columns:
            df["AI Model"] = ai_model

        logger.info("VertexMasterAnalyzer: results_df columns = %s", list(df.columns))
        try:
            logger.info(
                "VertexMasterAnalyzer: sample Kalshi fields: %s",
                df[[
                    "home_team",
                    "away_team",
                    "kalshi_available",
                    "kalshi_prob",
                    "kalshi_match_debug",
                ]]
                .head(10)
                .to_dict("records"),
            )
        except Exception:
            logger.info("VertexMasterAnalyzer: Kalshi columns not available for sample logging")
        try:
            kalshi_matches = int(df.get("kalshi_available", pd.Series([])).fillna(False).sum())
            logger.info(
                "VertexMasterAnalyzer: Kalshi match summary -> %d matched / %d games",
                kalshi_matches,
                len(df),
            )
        except Exception:
            logger.info("VertexMasterAnalyzer: unable to compute Kalshi summary")

        print(
            f"[Kalshi] attempts={kalshi_attempts}, hits={kalshi_hits}, errors={kalshi_errors}, failures={kalshi_fail_reasons}"
        )
        return df


# ---------------------------------------------------------------------------
# Streamlit display helper
# ---------------------------------------------------------------------------


def show_vertex_master_analysis(results_df: pd.DataFrame) -> None:
    """Render the Vertex AI Master Analysis in Streamlit."""
    if results_df is None or results_df.empty:
        st.info("No games to analyze.")
        return

    ml_counts = results_df["ml_source"].value_counts().to_dict()
    vertex_games = int(ml_counts.get("vertex_ai", 0))
    heuristic_games = int(ml_counts.get("heuristic", 0))

    st.subheader("🧠 ML Prediction Source Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Google Gemini (Vertex AI)", f"{vertex_games} games")
    with c2:
        st.metric("📊 Vertex fallback / heuristics", f"{heuristic_games} games")
    with c3:
        st.metric("🧮 Total Candidates", f"{len(results_df)} games")

    if vertex_games == 0 and heuristic_games > 0:
        st.info("Using heuristic probabilities for all games (Vertex AI not used).")

    st.markdown("---")
    st.subheader("🏆 Vertex AI Master Analysis - Complete Rankings")
    total_games = len(results_df)
    strong_faves = (results_df["win_prob"] >= 0.65).sum()
    solid_picks = (results_df["win_prob"] >= 0.55).sum()
    best_win_prob = float(results_df["win_prob"].max()) if not results_df.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Games Analyzed", total_games)
    with c2:
        st.metric("🔥 Strong Favorites (65%+)", strong_faves)
    with c3:
        st.metric("✅ Solid Picks (55%+)", solid_picks)
    with c4:
        st.metric("Best Win Probability", f"{best_win_prob*100:.1f}%" if best_win_prob else "—")

    st.markdown("---")
    st.subheader("🎯 SINGLE BEST PICK PER GAME")

    display_df = results_df.copy()
    logger.info("show_vertex_master_analysis: columns = %s", list(display_df.columns))
    # Ensure Kalshi columns exist even when Kalshi is disabled or missing
    if "kalshi_available" not in display_df.columns:
        display_df["kalshi_available"] = False
    if "kalshi_prob" not in display_df.columns:
        display_df["kalshi_prob"] = None
    if "kalshi_match_debug" not in display_df.columns:
        display_df["kalshi_match_debug"] = "no_kalshi_column"
    if "kalshi_status" not in display_df.columns:
        display_df["kalshi_status"] = "no_kalshi_column"
    if "kalshi_label" not in display_df.columns:
        display_df["kalshi_label"] = None
    if "kalshi_volume" not in display_df.columns:
        display_df["kalshi_volume"] = None
    if "kalshi_confidence" not in display_df.columns:
        display_df["kalshi_confidence"] = None
    if "AI Win %" not in display_df.columns:
        win_series = display_df.get("win_prob") if "win_prob" in display_df else pd.Series(np.nan, index=display_df.index)
        display_df["AI Win %"] = (win_series * 100).round(1)
    if "ai_status" not in display_df.columns:
        display_df["ai_status"] = "unknown"
    if "AI Model" not in display_df.columns:
        display_df["AI Model"] = VERTEX_MODEL_DISPLAY_NAME

    ai_status_val = display_df["ai_status"].iloc[0] if not display_df.empty else "unknown"
    ai_model_val = display_df["AI Model"].iloc[0] if not display_df.empty else VERTEX_MODEL_DISPLAY_NAME
    print(
        f"[Vertex] status={ai_status_val}, model={ai_model_val}, valid_probs={display_df['AI Win %'].notna().sum()} / {len(display_df)}"
    )
    display_df["sort_time"] = pd.to_datetime(display_df["game_time"], errors="coerce")
    display_df = display_df.sort_values("sort_time", ascending=True)
    display_df["is_today"] = display_df["game_time"].apply(is_today_calendar_day)

    def league_display(val: str) -> str:
        v = normalize_league(str(val))
        mapping = {"nba": "NBA", "ncaab": "NCAAB", "ncaaf": "NCAAF", "nhl": "NHL"}
        return mapping.get(v, v.upper()) if v else ""

    if "league" in display_df.columns:
        league_source = display_df["league"]
    elif "sport_key" in display_df.columns:
        league_source = display_df["sport_key"]
    else:
        league_source = ""
    display_df["League"] = league_source.apply(league_display) if hasattr(league_source, "apply") else league_display(league_source)

    display_df = display_df.sort_values(
        ["is_today", "sort_time", "win_prob"], ascending=[False, True, False]
    ).reset_index(drop=True)
    display_df["Rank"] = display_df.index + 1
    display_df["Win %"] = (display_df["win_prob"] * 100).round(1)
    display_df["Market %"] = display_df["market_prob"].apply(
        lambda p: p * 100 if p is not None and not pd.isna(p) else np.nan
    )
    display_df["Edge vs Market %"] = display_df["edge_vs_market"].apply(
        lambda e: e * 100 if e is not None and not pd.isna(e) else np.nan
    )
    display_df.loc[
        ~display_df["kalshi_available"].fillna(False),
        ["kalshi_prob", "kalshi_label", "kalshi_confidence", "kalshi_volume"],
    ] = None

    def _edge_vs_kalshi(row: pd.Series) -> float:
        if not row.get("kalshi_available"):
            return float("nan")
        p_ai = row.get("win_prob")
        p_k = row.get("kalshi_prob")
        if p_ai is None or pd.isna(p_ai) or p_k is None or pd.isna(p_k):
            return float("nan")
        return (p_ai - p_k) * 100.0

    display_df["Edge vs Kalshi %"] = display_df.apply(_edge_vs_kalshi, axis=1)
    display_df["theover_edge_raw"] = display_df.get("theover_probability_for_pick")
    display_df["TheOver Edge %"] = display_df["theover_edge_raw"].apply(
        lambda e: e * 100 if e is not None and not pd.isna(e) else ""
    )

    display_df["Market Type"] = display_df.get("pick_market_type", "")

    def format_line(row):
        line_val = row.get("pick_line")
        mkt = row.get("pick_market_type")
        if line_val is None or pd.isna(line_val):
            return "—" if mkt != "Spread" else "—"
        try:
            line_f = float(line_val)
        except Exception:
            return "—"
        if mkt == "Spread":
            return f"{line_f:+.1f}"
        if mkt == "Total":
            return f"{line_f:.1f}"
        return "—"

    display_df["Line"] = display_df.apply(format_line, axis=1)

    def format_odds(odds_val):
        if odds_val is None or pd.isna(odds_val):
            return "—"
        try:
            odds_f = float(odds_val)
        except Exception:
            return "—"
        return f"+{int(odds_f)}" if odds_f > 0 else f"{int(odds_f)}"

    display_df["Odds"] = display_df["pick_odds"].apply(format_odds)
    
    # Changed from emojis to text
    def favorite_label(row):
        if row.get("pick_market_type") == "Total":
            return "—"
        val = row.get("is_favorite")
        if val is None or pd.isna(val):
            return "—"
        return "Yes" if bool(val) else "No (Underdog)"

    display_df["Favorite"] = display_df.apply(favorite_label, axis=1)
    
    # Sentiment: Show as +/- percentage impact
    display_df["Sentiment"] = display_df["sentiment_diff"].apply(
        lambda x: f"+{x*100:.1f}%" if x is not None and x > 0 
                 else f"{x*100:.1f}%" if x is not None and x < 0
                 else "0.0%"
    )
    
    # Kalshi: Show alignment as text + percentage
    def format_kalshi(row):
        """Display the matched Kalshi market label (or fallback message)."""
        kalshi_prob = row.get("kalshi_prob")
        kalshi_available = bool(row.get("kalshi_available"))
        label = row.get("kalshi_label") or row.get("market_title")

        if not kalshi_available or kalshi_prob is None or pd.isna(kalshi_prob):
            return "No Kalshi match"
        return label or "Kalshi match"

    display_df["Kalshi"] = display_df.apply(format_kalshi, axis=1)
    display_df["Kalshi %"] = display_df.apply(
        lambda row: round(float(row["kalshi_prob"]) * 100, 1)
        if row.get("kalshi_available")
        and row.get("kalshi_prob") is not None
        and not pd.isna(row.get("kalshi_prob"))
        else np.nan,
        axis=1,
    )
    
    # Add Kalshi Edge column (YOUR model prob - Kalshi crowd prob)
    def calc_kalshi_edge(row):
        kalshi_prob = row.get("kalshi_prob")
        win_prob = row.get("win_prob")  # Your model's probability

        if kalshi_prob is not None and not pd.isna(kalshi_prob) and win_prob is not None:
            # Edge in decimal space: model win prob minus Kalshi crowd prob
            edge = win_prob - kalshi_prob
            return edge
        return np.nan

    display_df["Kalshi Edge"] = display_df.apply(calc_kalshi_edge, axis=1)

    no_kalshi_mask = ~display_df["kalshi_available"].fillna(False)
    display_df.loc[no_kalshi_mask, "Kalshi"] = "No Kalshi match"
    display_df.loc[no_kalshi_mask, "Kalshi %"] = np.nan
    display_df.loc[no_kalshi_mask, "Edge vs Kalshi %"] = np.nan
    display_df.loc[no_kalshi_mask, "Kalshi Edge"] = np.nan

    logger.info(
        "Kalshi probs sample: %s",
        display_df["Kalshi %"].dropna().head(10).tolist(),
    )

    display_df["EV"] = display_df["ev"].map(lambda x: f"${x:.2f}")

    # Normalize pick market type casing
    if "pick_market_type" in display_df.columns:
        display_df["pick_market_type"] = display_df["pick_market_type"].astype(str).str.lower()

    # NEW: Format Game Time column
    def format_game_time(time_str):
        if pd.isna(time_str) or not time_str:
            return "—"
        try:
            from datetime import datetime
            import pytz
            # Parse ISO time from TheOddsAPI
            dt = datetime.fromisoformat(str(time_str).replace('Z', '+00:00'))
            # Convert to Eastern Time for display
            eastern = pytz.timezone('US/Eastern')
            dt_eastern = dt.astimezone(eastern)
            # Format as "Nov 29, 7:30 PM ET"
            return dt_eastern.strftime("%b %d, %-I:%M %p ET")
        except Exception:
            return str(time_str)[:16] if time_str else "—"
    
    display_df["Game Time"] = display_df["game_time"].apply(format_game_time)
    
    # TheOver alignment/edge columns derived from keyed matches
    display_df["TheOver"] = display_df.get("theover_alignment", pd.Series([])).fillna("No Coverage")
    display_df["theover_edge_raw"] = display_df.get("theover_probability_for_pick")
    display_df["TheOver Edge %"] = display_df["theover_edge_raw"].apply(
        lambda x: f"{float(x) * 100:.1f}" if pd.notna(x) else ""
    )

    # Kalshi status indicator
    def classify_kalshi_status(row: pd.Series) -> str:
        existing = str(row.get("kalshi_status") or "").strip()
        if existing:
            return existing
        prob = row.get("kalshi_prob") or row.get("Kalshi %")
        debug = str(row.get("kalshi_match_debug") or row.get("Kalshi") or "")

        try:
            if prob is not None and str(prob) != "" and not pd.isna(float(prob)):
                return "Matched"
        except Exception:
            pass

        dlow = debug.lower()
        if "no kalshi match" in dlow or "no_market" in dlow or "no markets" in dlow:
            return "No market on Kalshi"
        if "symbol" in dlow or "name mismatch" in dlow:
            return "Symbol / name mismatch"
        if "api" in dlow and ("empty" in dlow or "error" in dlow or "no response" in dlow):
            return "Kalshi API error / empty"
        if debug:
            return debug
        return "No Kalshi data"

    display_df["kalshi_status"] = display_df.apply(classify_kalshi_status, axis=1)
    
    # NEW: Novig Odds column
    def format_novig_odds(row):
        pick_team = row.get("pick_team", "")
        home_team = row.get("home_team", "")
        
        novig_home = row.get("novig_home_spread")
        novig_away = row.get("novig_away_spread")
        novig_home_odds = row.get("novig_home_spread_odds")
        novig_away_odds = row.get("novig_away_spread_odds")
        
        # If no Novig data available
        if pd.isna(novig_home) and pd.isna(novig_away):
            return "—"
        
        # Determine which spread to show based on picked team
        if pick_team == home_team:
            # Home team picked, show home spread
            if not pd.isna(novig_home) and not pd.isna(novig_home_odds):
                return f"{novig_home:+.1f} ({int(novig_home_odds):+d})"
            return "—"
        else:
            # Away team picked, show away spread
            if not pd.isna(novig_away) and not pd.isna(novig_away_odds):
                return f"{novig_away:+.1f} ({int(novig_away_odds):+d})"
            return "—"
    
    display_df["Novig"] = display_df.apply(format_novig_odds, axis=1)

    # Ensure debug columns are always readable strings
    if "kalshi_match_debug" in display_df.columns:
        display_df["kalshi_match_debug"] = display_df["kalshi_match_debug"].fillna("no_match_found")
    if "theover_match_debug" in display_df.columns:
        display_df["theover_match_debug"] = display_df["theover_match_debug"].fillna("no_match_found")

    cols = [
        "Rank",
        "League",
        "league",
        "game",
        "Game Time",
        "the_pick",
        "Market Type",
        "Line",
        "Odds",
        "Win %",
        "Market %",
        "Edge vs Market %",
        "Favorite",
        "Novig",     # Novig odds for picked team
        "TheOver",
        "TheOver Edge %",
        "Sentiment",  # Shows +/-% sentiment impact
        "Kalshi",
        "Kalshi %",
        "Edge vs Kalshi %",
        "Kalshi Edge",  # NEW - How much Kalshi favors this pick
        "EV",
    ]

    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
    )

    if "AI Win %" not in display_df.columns:
        display_df["AI Win %"] = (display_df.get("win_prob", pd.Series(np.nan, index=display_df.index)) * 100).round(1)
    if "AI Model" not in display_df.columns:
        display_df["AI Model"] = VERTEX_MODEL_DISPLAY_NAME
    if "ai_status" not in display_df.columns:
        display_df["ai_status"] = "unknown"

    csv_cols = cols + [
        "AI Win %",
        "AI Model",
        "ai_status",
        "win_prob",
        "AI Win %",
        "AI Model",
        "market_prob",
        "edge_vs_market",
        "pick_market_type",
        "pick_line",
        "pick_odds",
        "pick_selection",
        "theover_pick_team",
        "theover_pick_side",
        "theover_pick_line",
        "theover_edge_raw",
        "theover_alignment",
        "kalshi_prob",
        "edge_vs_kalshi",
        "kalshi_match_debug",
        "theover_edge", 
        "theover_match_debug",
        "kalshi_status",
        "ai_status",
    ]

    logging.info("[Export] Columns in CSV: %s", csv_cols)
    logging.info(
        "[Kalshi CSV] sample Kalshi values: %s",
        display_df[["League", "game", "Kalshi", "Kalshi %", "kalshi_match_debug"]]
        .head(5)
        .to_dict("records"),
    )

    csv = display_df[csv_cols].to_csv(index=False)
    st.download_button(
        "📥 Download Single Best Pick Per Game (CSV)",
        data=csv,
        file_name=f"vertex_single_best_pick_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


def vertex_prediction_sanity_sample() -> None:
    """Quick sanity check to verify Vertex endpoint connectivity."""

    sample_rows = [
        {
            "home_win_pct": 0.6,
            "away_win_pct": 0.4,
            "home_avg_points": 112,
            "away_avg_points": 107,
            "home_last_5_wins": 3,
            "away_last_5_wins": 2,
            "home_spread": -3.5,
            "implied_home_prob": 0.58,
            "theover_probability": 0.55,
        },
        {
            "home_win_pct": 0.48,
            "away_win_pct": 0.52,
            "home_avg_points": 105,
            "away_avg_points": 109,
            "home_last_5_wins": 1,
            "away_last_5_wins": 4,
            "home_spread": 4.5,
            "implied_home_prob": 0.42,
            "theover_probability": 0.47,
        },
    ]

    df = pd.DataFrame(sample_rows)
    for col in VERTEX_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    probs = predict_win_probabilities(df, VERTEX_FEATURE_COLUMNS)
    print("Sanity sample Vertex probabilities:", probs)


def debug_kalshi_analysis(df: pd.DataFrame) -> None:
    """Run Kalshi matching on a DataFrame and print a summary."""

    kalshi_attempts = 0
    kalshi_hits = 0
    kalshi_fail_reasons: Dict[str, int] = {}

    for _, row in df.iterrows():
        league = str(row.get("league") or row.get("League") or "").lower()
        home_team, away_team = split_game(str(row.get("game") or ""))
        game_time = (
            row.get("game_time_raw")
            or row.get("Game Time Raw")
            or row.get("game_time")
            or row.get("commence_time")
        )

        kalshi_attempts += 1
        result = match_game_to_kalshi(league, home_team or "", away_team or "", game_time)
        if result.get("matched"):
            kalshi_hits += 1
        else:
            reason = result.get("reason", "unknown")
            kalshi_fail_reasons[reason] = kalshi_fail_reasons.get(reason, 0) + 1

    print(f"[Kalshi DEBUG] attempts={kalshi_attempts}, hits={kalshi_hits}, failures={kalshi_fail_reasons}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        import pandas as pd

        path = sys.argv[1]
        df = pd.read_csv(path)
        debug_kalshi_analysis(df)
