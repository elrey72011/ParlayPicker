
"""
Vertex AI Master Analyzer
Consolidates ALL data sources for ultimate best bet recommendations.
Rewritten clean version – no Anthropic, uses Google Gemini (Vertex AI)
plus spread-derived and fallback logic.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ml_predictions import get_vertex_ai_prediction, is_vertex_ai_enabled

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.odds_api = odds_api_client
        self.sportsdata = sportsdata_clients or {}
        self.apisports = apisports_clients or {}
        self.sentiment = sentiment_analyzer
        self.local_ml = local_ml_predictor
        self.theover = theover_data or {}
        self.kalshi = kalshi_integrator

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def build_comprehensive_features(self, game: Dict[str, Any], league: str) -> Dict[str, Any]:
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
        features.update(self._get_kalshi_features(game))

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
            return {"local_ml_prob": 0.5, "local_ml_confidence": 0.0}

        try:
            pred = self.local_ml.predict_game_outcome(
                home_team=game.get("home_team"),
                away_team=game.get("away_team"),
                sport_key=game.get("sport_key"),
            )
            return {
                "local_ml_prob": float(pred.get("home_win_prob", 0.5)),
                "local_ml_confidence": float(pred.get("confidence", 0.0)),
            }
        except Exception as e:
            logger.warning(f"Local ML prediction error: {e}")
            return {"local_ml_prob": 0.5, "local_ml_confidence": 0.0}

    # ---- TheOver.ai + Kalshi -----------------------------------------

    def _get_theover_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        def sf(val: Any, default: float) -> float:
            try:
                if val is None or pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        return {
            "theover_has_pick": 1 if game.get("theover_probability") not in (None, 0.5) else 0,
            "theover_pick": game.get("theover_pick") or "",
            "theover_probability": sf(game.get("theover_probability", 0.5), 0.5),
            "theover_spread": sf(game.get("theover_spread", 0.0), 0.0),
            "theover_total": sf(game.get("theover_total", 0.0), 0.0),
            "theover_total_pick": game.get("theover_total_pick") or "",
            "theover_total_probability": sf(game.get("theover_total_probability", 0.5), 0.5),
        }

    def _get_kalshi_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        feats = {
            "kalshi_available": False,
            "kalshi_prob": 0.5,
            "kalshi_alignment": 0.5,
        }
        if not self.kalshi:
            return feats

        try:
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            sport_key = (game.get("sport_key") or "").lower()

            if "nba" in sport_key:
                sport = "NBA"
            elif "nfl" in sport_key:
                sport = "NFL"
            elif "ncaab" in sport_key:
                sport = "NCAAB"
            elif "ncaaf" in sport_key:
                sport = "NCAAF"
            else:
                sport = "NBA"

            kalshi_data = None
            if hasattr(self.kalshi, "get_game_market"):
                kalshi_data = self.kalshi.get_game_market(
                    home_team=home_team, away_team=away_team, sport=sport
                )

            if kalshi_data and kalshi_data.get("kalshi_available"):
                prob = float(kalshi_data.get("kalshi_prob", 0.5))
            else:
                # Synthetic: very light-weight blend of implied + theover
                implied = game.get("implied_home_prob", 0.5) or 0.5
                theo = game.get("theover_probability", implied) or implied
                prob = float(0.7 * theo + 0.3 * implied)

            prob = max(0.15, min(0.85, prob))
            feats["kalshi_available"] = True
            feats["kalshi_prob"] = prob

            implied = game.get("implied_home_prob", 0.5) or 0.5
            feats["kalshi_alignment"] = 1.0 - abs(prob - implied)
        except Exception as e:
            logger.warning(f"Kalshi feature error: {e}")

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
        if implied_home is None:
            implied_home = 0.5
        d["implied_home_prob"] = float(implied_home)

        # Consensus blend
        probs = [
            d["implied_home_prob"],
            sg("local_ml_prob", 0.5),
            sg("theover_probability", 0.5),
        ]
        if feats.get("kalshi_available"):
            probs.append(sg("kalshi_prob", 0.5))
            probs.append(sg("kalshi_prob", 0.5))  # double weight Kalshi
        valid = [p for p in probs if p is not None and not pd.isna(p)]
        d["consensus_prob"] = float(np.mean(valid)) if valid else 0.5

        d["kalshi_validation_score"] = (
            1.0 - abs(sg("kalshi_prob", 0.5) - d["consensus_prob"])
            if feats.get("kalshi_available")
            else 0.5
        )

        return d

    # ------------------------------------------------------------------
    # Vertex AI / Gemini analysis
    # ------------------------------------------------------------------

    def build_vertex_feature_vector(self, feats: Dict[str, Any]) -> List[float]:
        """Map comprehensive features -> numeric vector for Vertex."""

        def sg(key: str, default: float) -> float:
            val = feats.get(key, default)
            try:
                if val is None or pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        vec = [
            sg("home_win_pct", 0.5),
            sg("away_win_pct", 0.5),
            sg("win_pct_diff", 0.0),
            sg("home_avg_points", 100.0) / 100.0,
            sg("away_avg_points", 100.0) / 100.0,
            sg("home_avg_points_allowed", 100.0) / 100.0,
            sg("away_avg_points_allowed", 100.0) / 100.0,
            sg("form_momentum_diff", 0.0) / 5.0,
            sg("implied_home_prob", 0.5),
            sg("home_spread", 0.0) / 20.0,
            sg("total_line", 200.0) / 200.0,
            sg("sentiment_diff", 0.0),
            sg("local_ml_prob", 0.5),
            sg("theover_probability", 0.5),
            sg("consensus_prob", 0.5),
            sg("theover_total", 200.0) / 200.0,
            sg("theover_total_probability", 0.5),
            1.0 if feats.get("kalshi_available") else 0.0,
            sg("kalshi_prob", 0.5),
            sg("kalshi_alignment", 0.5),
        ]

        cleaned: List[float] = []
        for v in vec:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    cleaned.append(0.0)
                else:
                    cleaned.append(float(v))
            except Exception:
                cleaned.append(0.0)
        return cleaned

    def analyze_all_games(self, games: List[Dict[str, Any]], league: str = "NBA") -> pd.DataFrame:
        """Run Vertex / Gemini + fallbacks across all games.

        Returns a DataFrame with one row per game.
        """
        if not games:
            return pd.DataFrame()

        vertex_enabled = is_vertex_ai_enabled()

        ml_sources_used = {
            "gcp_vertex": 0,
            "spread_derived": 0,
            "fallback_heuristic": 0,
        }

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

                feats = self.build_comprehensive_features(game, game_league)
                vec = self.build_vertex_feature_vector(feats)

                home_team = feats.get("home_team")
                away_team = feats.get("away_team")

                # ------------------------------------------------------------------
                # 1) Try Vertex / Gemini
                # ------------------------------------------------------------------
                ml_source = "spread_derived"
                home_win_prob: Optional[float] = None

                if vertex_enabled:
                    try:
                        feature_payload = {
                            "league": game_league,
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_ml_odds": feats.get("home_ml_odds"),
                            "away_ml_odds": feats.get("away_ml_odds"),
                            "home_spread": feats.get("home_spread"),
                            "total_line": feats.get("total_line"),
                            "implied_home_prob": feats.get("implied_home_prob"),
                            "theover_probability": feats.get("theover_probability"),
                            "theover_spread": feats.get("theover_spread"),
                            "theover_total": feats.get("theover_total"),
                            "theover_total_probability": feats.get("theover_total_probability"),
                            "kalshi_prob": feats.get("kalshi_prob"),
                            "local_ml_prob": feats.get("local_ml_prob"),
                            "consensus_prob": feats.get("consensus_prob"),
                            "numeric_features": vec,
                        }

                        context = (
                            f"{game_league} game: {away_team} at {home_team}. "
                            f"Sportsbook spread (home): {feats.get('home_spread')}. "
                            f"TheOver pick: {game.get('theover_pick')} "
                            f"with line {game.get('theover_spread')}."
                        )

                        home_win_prob = get_vertex_ai_prediction(feature_payload, context)
                        if home_win_prob is not None:
                            ml_source = "gcp_vertex"
                    except Exception as e:
                        logger.warning(
                            "Vertex AI prediction failed for %s vs %s: %s",
                            away_team,
                            home_team,
                            e,
                        )

                # ------------------------------------------------------------------
                # 2) Fallback: spread-derived 2.8% per point formula
                # ------------------------------------------------------------------
                if home_win_prob is None:
                    spread = feats.get("home_spread")
                    implied = feats.get("implied_home_prob", 0.5) or 0.5
                    try:
                        if spread is not None and not pd.isna(spread):
                            spread = float(spread)
                            # Negative spread (favorite) -> > 0.5
                            home_win_prob = max(0.15, min(0.85, 0.5 - spread * 0.028))
                        else:
                            home_win_prob = float(implied)
                    except Exception:
                        home_win_prob = float(implied)
                    ml_source = "spread_derived"

                # 3) Final fallback: straight consensus if still None
                if home_win_prob is None:
                    home_win_prob = float(feats.get("consensus_prob", 0.5) or 0.5)
                    ml_source = "fallback_heuristic"

                # Track source usage
                if ml_source in ml_sources_used:
                    ml_sources_used[ml_source] += 1

                # ------------------------------------------------------------------
                # Recommended pick - use AI probability to pick the team
                # DON'T use TheOver.ai's pick (it can be wrong!)
                # ------------------------------------------------------------------
                # Pick the team with higher AI win probability
                if home_win_prob >= 0.5:
                    pick_team = home_team
                else:
                    pick_team = away_team

                # Get TheOver.ai pick for consensus/comparison only
                theover_pick = feats.get("theover_pick") or ""

                # ------------------------------------------------------------------
                # Spread value for picked team
                # ALWAYS use home_spread from TheOddsAPI (reliable)
                # DON'T use theover_spread (unreliable signs)
                # ------------------------------------------------------------------
                home_spread = feats.get("home_spread") or 0.0
                try:
                    home_spread = float(home_spread)
                except Exception:
                    home_spread = 0.0

                # Calculate picked team's spread based on home_spread
                if pick_team == home_team:
                    pick_spread = home_spread  # Home picked, use home spread as-is
                else:
                    pick_spread = -home_spread  # Away picked, flip the sign

                try:
                    pick_spread = float(pick_spread)
                except Exception:
                    pick_spread = 0.0

                # Determine which side is actual favorite based on moneylines
                home_ml = feats.get("home_ml_odds")
                away_ml = feats.get("away_ml_odds")
                home_ip = implied_prob_from_american(home_ml) or 0.5
                away_ip = implied_prob_from_american(away_ml) or 0.5

                home_is_favorite = home_ip >= away_ip
                pick_is_favorite = home_is_favorite if pick_team == home_team else (not home_is_favorite)

                # Probability for the picked team
                away_win_prob = 1.0 - home_win_prob
                pick_win_prob = home_win_prob if pick_team == home_team else away_win_prob

                # EV: assume stake = 1 unit using picked team's ML
                pick_ml = home_ml if pick_team == home_team else away_ml
                dec = american_to_decimal(pick_ml)
                if dec is None:
                    ev = 0.0
                else:
                    ev = pick_win_prob * (dec - 1.0) - (1.0 - pick_win_prob)

                rows.append(
                    {
                        "league": game_league,
                        "game": f"{away_team} @ {home_team}",
                        "home_team": home_team,
                        "away_team": away_team,
                        "the_pick": f"{pick_team} {pick_spread:+.1f}",
                        "pick_team": pick_team,
                        "pick_spread": pick_spread,
                        "win_prob": pick_win_prob,
                        "home_win_prob": home_win_prob,
                        "ml_source": ml_source,
                        "is_favorite": pick_is_favorite,
                        "home_ml_odds": home_ml,
                        "away_ml_odds": away_ml,
                        "kalshi_prob": feats.get("kalshi_prob"),
                        "kalshi_alignment": feats.get("kalshi_alignment"),
                        "sentiment_diff": feats.get("sentiment_diff"),
                        "ev": ev,
                        # Game time and TheOver.ai consensus
                        "game_time": game.get("commence_time"),
                        "theover_pick": theover_pick,  # TheOver.ai's pick for consensus
                        # Novig odds
                        "novig_home_spread": feats.get("novig_home_spread"),
                        "novig_away_spread": feats.get("novig_away_spread"),
                        "novig_home_spread_odds": feats.get("novig_home_spread_odds"),
                        "novig_away_spread_odds": feats.get("novig_away_spread_odds"),
                    }
                )
            except Exception as e:
                logger.error(f"Error analyzing game: {e}", exc_info=True)

            progress.progress((idx + 1) / len(games))

        df = pd.DataFrame(rows)
        df.attrs["ml_sources_used"] = ml_sources_used
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
    vertex_games = int(ml_counts.get("gcp_vertex", 0))
    spread_games = int(ml_counts.get("spread_derived", 0))
    fallback_games = int(ml_counts.get("fallback_heuristic", 0))

    st.subheader("🧠 ML Prediction Source Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Google Gemini (Vertex AI)", f"{vertex_games} games")
    with c2:
        st.metric("📊 Spread-Derived", f"{spread_games} games")
    with c3:
        st.metric("🪫 Fallback Heuristics", f"{fallback_games} games")

    if vertex_games == 0 and spread_games > 0:
        st.info("Using spread × 2.8% formula for all games (Vertex AI not used).")

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
    display_df = display_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
    display_df["Rank"] = display_df.index + 1
    display_df["Win %"] = (display_df["win_prob"] * 100).round(1)
    display_df["Favorite"] = display_df["is_favorite"].apply(lambda x: "✅" if bool(x) else "🚨 Underdog")
    display_df["Sentiment"] = display_df["sentiment_diff"].apply(
        lambda x: "✅" if x is not None and x > 0 else "❌"
    )
    display_df["Kalshi"] = display_df["kalshi_alignment"].apply(
        lambda x: "✅" if x is not None and x > 0.5 else "❌"
    )
    display_df["Kalshi %"] = (display_df["kalshi_prob"] * 100).round(0)
    display_df["EV"] = display_df["ev"].map(lambda x: f"${x:.2f}")
    
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
    
    # NEW: TheOver.ai Consensus column
    def format_consensus(row):
        our_pick = row.get("pick_team", "")
        theover_pick = row.get("theover_pick", "")
        
        if not theover_pick or pd.isna(theover_pick):
            return "—"  # No TheOver.ai data
        
        # Check if picks match (case insensitive, partial match)
        if our_pick and theover_pick:
            if our_pick.lower() in theover_pick.lower() or theover_pick.lower() in our_pick.lower():
                return "✅ Agree"
            else:
                return "❌ Disagree"
        return "—"
    
    display_df["TheOver"] = display_df.apply(format_consensus, axis=1)
    
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

    cols = [
        "Rank",
        "league",
        "game",
        "Game Time",
        "the_pick",
        "Win %",
        "Favorite",
        "Novig",     # NEW - Novig odds for picked team
        "TheOver",
        "Sentiment",
        "Kalshi",
        "Kalshi %",
        "EV",
    ]

    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
    )

    csv = display_df[cols].to_csv(index=False)
    st.download_button(
        "📥 Download Single Best Pick Per Game (CSV)",
        data=csv,
        file_name=f"vertex_single_best_pick_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
