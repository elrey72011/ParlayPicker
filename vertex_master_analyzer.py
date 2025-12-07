"""
Vertex AI Master Analyzer
Consolidates ALL data sources for ultimate best bet recommendations.
"""
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytz
import numpy as np
import pandas as pd
import streamlit as st

# Import shared modules
try:
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
except ImportError as e:
    # Fallback to prevent crash if dependencies are missing during refactor
    logging.warning(f"VertexMasterAnalyzer import warning: {e}")
    TeamNameMatcher = None
    KalshiMatchResult = dict
    match_game_to_kalshi = lambda *args, **kwargs: {}
    VERTEX_FEATURE_COLUMNS = []
    VERTEX_MODEL_DISPLAY_NAME = "Unknown"
    is_vertex_prediction_configured = lambda: False
    score_with_vertex = lambda *args: (pd.Series([]), "error", "none")
    predict_win_probabilities = lambda *args: []

logger = logging.getLogger(__name__)

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
        self.use_kalshi = bool(use_kalshi and st.session_state.get("kalshi_enabled", True))

    def build_comprehensive_features(
        self, game: Dict[str, Any], league: str, kalshi_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a comprehensive feature dict for a game."""
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
        features.update(self._get_kalshi_features(game, league, kalshi_info))
        features.update(self._calculate_derived_features(features))

        return features

    def _get_market_odds_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
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
            "novig_home_spread": None,
            "novig_away_spread": None,
            "novig_home_spread_odds": None,
            "novig_away_spread_odds": None,
        }

        if feats["away_spread"] is None and feats["home_spread"] is not None:
            try:
                feats["away_spread"] = -float(feats["home_spread"])
            except Exception:
                feats["away_spread"] = None

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
                                if "novig" in book_name or "lowvig" in book_name:
                                    if name == game.get("home_team"):
                                        feats["novig_home_spread"] = point
                                        feats["novig_home_spread_odds"] = price
                                    elif name == game.get("away_team"):
                                        feats["novig_away_spread"] = point
                                        feats["novig_away_spread_odds"] = price
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
        return feats

    def _get_team_stats_features(self, game: Dict[str, Any], league: str) -> Dict[str, Any]:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        client = self.sportsdata.get(league.lower()) if self.sportsdata else None
        
        home_stats = {}
        away_stats = {}
        if client and hasattr(client, "get_team_stats"):
            try:
                home_stats = client.get_team_stats(home_team) or {}
                away_stats = client.get_team_stats(away_team) or {}
            except Exception:
                pass

        def safe(stats, key, default):
            return stats.get(key, default) if stats else default

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
        client = self.sportsdata.get(league.lower()) if self.sportsdata else None
        recent_home = {}
        recent_away = {}
        if client and hasattr(client, "get_recent_games"):
            try:
                recent_home = client.get_recent_games(game.get("home_team"), 5) or {}
                recent_away = client.get_recent_games(game.get("away_team"), 5) or {}
            except Exception:
                pass
        
        def safe(stats, key, default):
            return stats.get(key, default) if stats else default

        return {
            "home_last_5_wins": safe(recent_home, "wins", 2),
            "away_last_5_wins": safe(recent_away, "wins", 2),
        }

    def _get_sentiment_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        def get_for_team(team: str) -> float:
            if not self.sentiment:
                # Synthetic hash fallback
                return ((sum(ord(c) for c in (team or "")[:8]) % 100) - 50) / 200.0
            try:
                if hasattr(self.sentiment, "get_team_sentiment"):
                    data = self.sentiment.get_team_sentiment(team)
                    if isinstance(data, dict): return float(data.get("sentiment_score", 0.0))
                    if isinstance(data, (int, float)): return float(data)
            except Exception:
                pass
            return 0.0

        h = get_for_team(game.get("home_team"))
        a = get_for_team(game.get("away_team"))
        return {"home_sentiment": h, "away_sentiment": a, "sentiment_diff": h - a}

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
        except Exception:
            return {"local_ml_prob": None, "local_ml_confidence": 0.0}

    def _get_theover_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        def sf(val, default):
            try: return float(val) if val is not None and not pd.isna(val) else default
            except: return default

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

        if not pd.isna(base["theover_probability"]):
            return base

        if not self.theover:
            base["theover_match_debug"] = "no_theover_dataset"
            return base

        def _teams_similarity(home: str, away: str, cand_home: str, cand_away: str) -> float:
            if not TeamNameMatcher: return 0.0
            h = TeamNameMatcher.similarity_score(TeamNameMatcher.normalize(home), TeamNameMatcher.normalize(cand_home))
            a = TeamNameMatcher.similarity_score(TeamNameMatcher.normalize(away), TeamNameMatcher.normalize(cand_away))
            return min(h, a)

        def _choose_best_match(df, target_line):
            best_row = None
            best_score = 0.0
            best_line_diff = 999.0
            for _, row in df.iterrows():
                sim = _teams_similarity(
                    game.get("home_team",""), game.get("away_team",""),
                    row.get("HomeTeam") or row.get("home_team",""),
                    row.get("AwayTeam") or row.get("away_team","")
                )
                if sim < TEAM_FUZZY_THRESHOLD: continue
                
                cand_line = row.get("Line") if "Line" in row else row.get("line")
                try: cand_line_val = float(cand_line)
                except: cand_line_val = None
                
                line_diff = abs((target_line or 0) - cand_line_val) if target_line is not None and cand_line_val is not None else None
                if line_diff is not None and line_diff > MAX_LINE_DIFF: continue
                
                if sim > best_score or (sim == best_score and (line_diff or 999) < best_line_diff):
                    best_score = sim
                    best_row = row
                    best_line_diff = line_diff if line_diff is not None else best_line_diff
            return best_row

        match_row = None
        match_type = ""
        line_hint = game.get("home_spread") or game.get("theover_spread")
        
        spreads_df = self.theover.get("spreads")
        if spreads_df is not None and not spreads_df.empty:
            match_row = _choose_best_match(spreads_df, line_hint)
            match_type = "Spread" if match_row is not None else ""

        if match_row is None:
            totals_df = self.theover.get("totals")
            if totals_df is not None and not totals_df.empty:
                line_hint = game.get("total_line") or game.get("theover_total")
                match_row = _choose_best_match(totals_df, line_hint)
                match_type = "Total" if match_row is not None else ""

        if match_row is None:
            base["theover_match_debug"] = "no_match_found"
            return base

        pick = match_row.get("Pick") or match_row.get("pick") or ""
        try: line_val_f = float(match_row.get("Line") if "Line" in match_row else match_row.get("line"))
        except: line_val_f = None
        
        try: prob_val = float(match_row.get("Probability") if "Probability" in match_row else match_row.get("probability"))
        except: prob_val = None

        if prob_val is None and line_val_f is not None:
            prob_val = 0.5 + min(0.35, abs(line_val_f) * 0.028)

        pick_is_home = False
        if pick and TeamNameMatcher:
            pick_is_home = TeamNameMatcher.similarity_score(
                TeamNameMatcher.normalize(pick), TeamNameMatcher.normalize(game.get("home_team", ""))
            ) >= TEAM_FUZZY_THRESHOLD

        if prob_val is not None:
            prob_val = max(0.0, min(1.0, prob_val))
            if match_type == "Total":
                over_prob = prob_val if str(pick).lower().startswith("over") else (1.0 - prob_val)
                base.update({
                    "theover_has_pick": 1,
                    "theover_total": line_val_f or base.get("theover_total", 0.0),
                    "theover_total_pick": pick,
                    "theover_total_probability": over_prob,
                    "theover_match_debug": f"match={match_type}, pick={pick}",
                })
            else:
                home_prob = prob_val if pick_is_home else (1.0 - prob_val)
                base.update({
                    "theover_has_pick": 1,
                    "theover_pick": pick,
                    "theover_probability": home_prob,
                    "theover_spread": line_val_f or base.get("theover_spread", 0.0),
                    "theover_match_debug": f"match={match_type}, pick={pick}",
                })
        else:
            base["theover_match_debug"] = "matched_no_probability"

        return base

    def _get_kalshi_features(
        self, game: Dict[str, Any], league: str, prefetch_info: Optional[Dict[str, Any]] = None
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

        if not getattr(self, "kalshi", None) or not getattr(self, "use_kalshi", True):
            feats["kalshi_match_debug"] = "kalshi_not_configured"
            feats["kalshi_status"] = "kalshi_disabled"
            return feats

        try:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            # Always use the normalized league from analyze_all_games to avoid sport_key mismatches
            league = league or game.get("league") or game.get("sport_key") or "NBA"

            # CRITICAL FIX: Ensure game_dt is defined
            game_time = game.get("commence_time") or game.get("game_time")
            game_dt = None
            if game_time:
                try: game_dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
                except: game_dt = None

            market_info = prefetch_info
            if market_info is None:
                market_info = match_game_to_kalshi(league, home, away, game_dt, integrator=self.kalshi)

            logging.info(f"[Kalshi FETCH] home={home} away={away} dt={game_dt}")

            # Check if market_info is a match object or dict
            is_matched = False
            prob = None
            label = None

            def _pick_prob(info: Dict[str, Any]) -> Optional[float]:
                if not isinstance(info, dict):
                    return None
                for key in ("probability", "kalshi_probability", "kalshi_prob", "prob"):
                    if key in info and info.get(key) is not None:
                        return info.get(key)
                return None
            
            if isinstance(market_info, dict):
                is_matched = market_info.get("matched", False) or market_info.get("kalshi_available", False)
                prob = _pick_prob(market_info)
                label = market_info.get("label") or market_info.get("kalshi_label")
                debug_reason = market_info.get("reason") or market_info.get("kalshi_match_debug")
                feats["kalshi_match_debug"] = str(debug_reason)
                feats["kalshi_status"] = str(debug_reason)

            if not is_matched or prob is None:
                return feats

            try:
                prob = float(prob)
                prob = max(0.0, min(1.0, prob))
            except:
                return feats

            feats["kalshi_available"] = True
            feats["kalshi_prob"] = prob
            feats["kalshi_home_prob"] = prob
            feats["kalshi_label"] = label
            feats["kalshi_volume"] = market_info.get("kalshi_volume")
            
            # Simple Alignment Check
            model_p = game.get("implied_home_prob") or game.get("win_prob")
            if model_p is not None:
                try:
                    model_p = float(model_p)
                    if abs(model_p - prob) < 0.05: feats["kalshi_alignment"] = "Neutral"
                    elif model_p > prob: feats["kalshi_alignment"] = "Model > Kalshi"
                    else: feats["kalshi_alignment"] = "Kalshi > Model"
                except: pass

            return feats

        except Exception as e:
            logger.error(f"Kalshi feature error: {e}", exc_info=True)
            feats["kalshi_match_debug"] = f"error={str(e)}"
            return feats

    def _calculate_derived_features(self, feats: Dict[str, Any]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        def sg(key, default):
            try: return float(feats.get(key, default))
            except: return float(default)

        d["win_pct_diff"] = sg("home_win_pct", 0.5) - sg("away_win_pct", 0.5)
        d["off_def_matchup_home"] = sg("home_off_rating", 100.0) - sg("away_def_rating", 100.0)
        d["off_def_matchup_away"] = sg("away_off_rating", 100.0) - sg("home_def_rating", 100.0)
        d["form_momentum_diff"] = sg("home_last_5_wins", 2.0) - sg("away_last_5_wins", 2.0)

        implied_home = feats.get("implied_home_prob")
        if implied_home is None:
            implied_home = implied_prob_from_american(feats.get("home_ml_odds"))
        d["implied_home_prob"] = float(implied_home) if implied_home is not None else None

        probs = [d.get("implied_home_prob"), feats.get("local_ml_prob"), feats.get("theover_probability")]
        if feats.get("kalshi_available"):
            kp = feats.get("kalshi_prob")
            probs.extend([kp, kp])
            
        valid = [p for p in probs if p is not None and not pd.isna(p)]
        d["consensus_prob"] = float(np.mean(valid)) if valid else None

        return d

    def build_vertex_feature_vector(self, feats: Dict[str, Any]) -> List[float]:
        row = self.build_vertex_feature_row(feats)
        cleaned = []
        for col in VERTEX_FEATURE_COLUMNS:
            try: cleaned.append(float(row.get(col, 0.0)))
            except: cleaned.append(0.0)
        return cleaned

    def build_vertex_feature_row(self, feats: Dict[str, Any]) -> Dict[str, float]:
        def gv(key, default=0.0):
            try: return float(feats.get(key, default))
            except: return float(default)

        return {
            "home_win_pct": gv("home_win_pct", 0.5),
            "away_win_pct": gv("away_win_pct", 0.5),
            "home_avg_points": gv("home_avg_points"),
            "away_avg_points": gv("away_avg_points"),
            "home_form_last5": gv("home_last_5_wins"),
            "away_form_last5": gv("away_last_5_wins"),
            "win_pct_diff": gv("win_pct_diff"),
            "avg_points_diff": gv("home_avg_points") - gv("away_avg_points"),
            "model_consensus": gv("consensus_prob", 0.5),
            "theover_probability": gv("theover_probability", 0.5),
            "spread_normalized": gv("home_spread", 0.0),
            "sentiment_diff": gv("sentiment_diff", 0.0),
        }

    def _derive_home_probability(self, feats: Dict[str, Any]) -> Optional[float]:
        spread = feats.get("home_spread")
        if spread is not None:
            try: return max(0.15, min(0.85, 0.5 - float(spread) * 0.028))
            except: pass
        
        prob = feats.get("implied_home_prob") or feats.get("consensus_prob")
        try: return float(prob) if prob else None
        except: return None

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
        if odds is None or pd.isna(odds): return None
        market_prob = implied_prob_from_american(odds)
        if market_prob is None: return None

        ai_prob = None
        if vertex_enabled and vertex_home_prob is not None:
            if market_type == "ML" or market_type == "Spread":
                ai_prob = vertex_home_prob if selection == "home" else (1.0 - vertex_home_prob)
        
        if ai_prob is None:
            # Heuristic fallback
            base_prob = self._derive_home_probability(feats)
            if base_prob is not None:
                ai_prob = base_prob if selection == "home" else (1.0 - base_prob)
                if market_type == "Total": ai_prob = 0.5 # Simple heuristic for totals

        if ai_prob is None: return None
        ai_prob = max(0.0, min(1.0, float(ai_prob)))
        
        # Edge calc
        edge = ai_prob - market_prob
        dec_odds = american_to_decimal(odds)
        ev = (ai_prob * (dec_odds - 1.0) - (1.0 - ai_prob)) if dec_odds else 0.0

        # Kalshi Edge
        kalshi_prob = feats.get("kalshi_prob")
        edge_vs_kalshi = None
        if kalshi_prob is not None and market_type in ("ML", "Spread"):
            k_pick_prob = kalshi_prob if selection == "home" else (1.0 - kalshi_prob)
            edge_vs_kalshi = ai_prob - k_pick_prob

        # Format Pick
        pick_text = selection
        if market_type == "ML":
            pick_text = f"{feats.get('home_team')} ML" if selection == "home" else f"{feats.get('away_team')} ML"
        elif market_type == "Spread" and line is not None:
            team = feats.get('home_team') if selection == "home" else feats.get('away_team')
            pick_text = f"{team} {line:+.1f}"

        return {
            "league": game_league,
            "game": f"{feats.get('away_team')} @ {feats.get('home_team')}",
            "home_team": feats.get("home_team"),
            "away_team": feats.get("away_team"),
            "the_pick": pick_text,
            "pick_market_type": market_type,
            "pick_line": line,
            "pick_odds": odds,
            "win_prob": ai_prob,
            "market_prob": market_prob,
            "edge_vs_market": edge,
            "edge_vs_kalshi": edge_vs_kalshi,
            "ev": ev,
            "kalshi_prob": kalshi_prob if selection == "home" else (1.0 - (kalshi_prob or 0.0)) if kalshi_prob else None,
            "kalshi_match_debug": feats.get("kalshi_match_debug", ""),
            "kalshi_available": feats.get("kalshi_available", False),
            "kalshi_label": feats.get("kalshi_label"),
            "game_time": feats.get("game_time"),
        }

    def analyze_all_games(self, games: List[Dict[str, Any]], league: str = "NBA") -> pd.DataFrame:
        if not games: return pd.DataFrame()
        vertex_enabled = is_vertex_prediction_configured()
        rows = []
        progress = st.progress(0)

        for idx, game in enumerate(games):
            try:
                # League detection
                skey = game.get("sport_key", "").lower()
                league_map = {
                    "nba": "NBA",
                    "basketball_nba": "NBA",
                    "nfl": "NFL",
                    "americanfootball_nfl": "NFL",
                    "ncaab": "NCAAB",
                    "basketball_ncaab": "NCAAB",
                    "ncaaf": "NCAAF",
                    "americanfootball_ncaaf": "NCAAF",
                    "nhl": "NHL",
                    "icehockey_nhl": "NHL",
                    "mlb": "MLB",
                    "baseball_mlb": "MLB",
                }
                game_league = league_map.get(skey, league)

                # Prefetch Kalshi
                kalshi_info = None
                if getattr(self, "kalshi", None) and getattr(self, "use_kalshi", True):
                    try:
                        # Add this RIGHT BEFORE line 684 (before the kalshi_info = match_game_to_kalshi call)

                        # Debug: Check what we're trying to match
                        logger.info(f"\n{'='*80}")
                        logger.info(f"KALSHI DEBUG - Attempting to match:")
                        logger.info(f"  Home: {game.get('home_team', '')}")
                        logger.info(f"  Away: {game.get('away_team', '')}")
                        logger.info(f"  League: {game_league}")
                        logger.info(f"  Integrator available: {integrator is not None}")
                        logger.info(f"{'='*80}")
                        
                        # Also let's check what markets are available (do this once at the start)
                        if integrator:
                            test_markets = integrator.get_todays_events("KXNBA")
                            logger.info(f"\nKALSHI: Found {len(test_markets)} NBA markets")
                            if test_markets:
                                sample = test_markets[0]
                                logger.info(f"Sample market structure:")
                                logger.info(f"  event_ticker: {sample.get('event_ticker')}")
                                logger.info(f"  event_title: {sample.get('event_title')}")
                                logger.info(f"  title: {sample.get('title')}")
                                logger.info(f"  subtitle: {sample.get('subtitle')}")
                        kalshi_info = match_game_to_kalshi(
                            game_league, game.get("home_team",""), game.get("away_team",""), game.get("commence_time"), integrator=self.kalshi
                        )
                    except Exception as e:
                        logger.warning(f"Kalshi prefetch fail: {e}")

                feats = self.build_comprehensive_features(game, game_league, kalshi_info)
                vec = self.build_vertex_feature_vector(feats)
                feat_row = self.build_vertex_feature_row(feats)

                # Vertex Prediction
                vertex_home_prob = None
                if vertex_enabled:
                    try:
                        probs = predict_win_probabilities(pd.DataFrame([feat_row]), VERTEX_FEATURE_COLUMNS)
                        if probs and len(probs) > 0: vertex_home_prob = float(probs[0])
                    except: pass

                candidates = []
                # ML
                for sel, odds in [("home", feats.get("home_ml_odds")), ("away", feats.get("away_ml_odds"))]:
                    c = self._evaluate_candidate(feats, "ML", sel, None, odds, game_league, vertex_enabled, vec, vertex_home_prob)
                    if c: candidates.append(c)
                # Spread
                for sel, line, odds in [("home", feats.get("home_spread"), feats.get("home_spread_odds")), 
                                      ("away", feats.get("away_spread"), feats.get("away_spread_odds"))]:
                    c = self._evaluate_candidate(feats, "Spread", sel, line, odds, game_league, vertex_enabled, vec, vertex_home_prob)
                    if c: candidates.append(c)

                if candidates:
                    # Sort by edge
                    best = sorted(candidates, key=lambda x: x.get("edge_vs_market", -99), reverse=True)[0]
                    rows.append(best)

            except Exception as e:
                logger.error(f"Error analyzing game {idx}: {e}")
            progress.progress((idx + 1) / len(games))

        return pd.DataFrame(rows)

def show_vertex_master_analysis(results_df: pd.DataFrame) -> None:
    """Render the Vertex AI Master Analysis in Streamlit."""
    if results_df is None or results_df.empty:
        st.info("No games to analyze.")
        return

    st.subheader("🏆 Vertex AI Master Analysis")
    
    display_df = results_df.copy()
    if "win_prob" in display_df.columns:
        display_df["Win %"] = (display_df["win_prob"] * 100).round(1)
    if "edge_vs_market" in display_df.columns:
        display_df["Edge %"] = (display_df["edge_vs_market"] * 100).round(1)
    
    def fmt_kalshi(row):
        if not row.get("kalshi_available") or row.get("kalshi_prob") is None:
            return "No Match"
        return f"{row.get('kalshi_label','Match')} ({float(row.get('kalshi_prob'))*100:.1f}%)"
    
    display_df["Kalshi Info"] = display_df.apply(fmt_kalshi, axis=1)
    
    cols = ["game", "the_pick", "pick_odds", "Win %", "Edge %", "ev", "Kalshi Info"]
    valid_cols = [c for c in cols if c in display_df.columns]
    
    st.dataframe(display_df[valid_cols], use_container_width=True)
