"""
Vertex AI Master Analyzer
Location: vertex_master_analyzer.py (ROOT DIRECTORY)
Consolidates ALL data sources for ultimate best bet recommendations.
Updated to use US/Eastern Time for matching logic.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict

import pandas as pd
import streamlit as st
import pytz  # Added pytz for timezone conversion

logger = logging.getLogger(__name__)


def _kalshi_prefix_counts(markets: List[Dict[str, Any]]) -> Dict[str, int]:
    tickers = [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in markets]
    return {
        "count_prefix_KXNBATOTAL": len([t for t in tickers if t.startswith("KXNBATOTAL")]),
        "count_prefix_KXNBASPREAD": len([t for t in tickers if t.startswith("KXNBASPREAD")]),
        "count_prefix_KXNBAGAME": len([t for t in tickers if t.startswith("KXNBAGAME")]),
    }


def _filter_markets_for_league(league: str, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not markets:
        return []
    league_key = (league or "").upper()
    tickers = [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in markets]
    if league_key != "NBA":
        return markets
    nba_markets = [m for m, t in zip(markets, tickers) if t.startswith("KXNBA") or "KXNBA" in t]
    if not nba_markets:
        return markets
    preferred = [
        m
        for m, t in zip(nba_markets, [str(m.get("ticker") or m.get("event_ticker") or "").upper() for m in nba_markets])
        if any(tag in t for tag in ["TOTAL", "SPREAD", "GAME", "WIN", "H2H"])
    ]
    return preferred or nba_markets


def _title_match_score(title: str, home_team: str, away_team: str) -> float:
    text = (title or "").lower()
    if not text:
        return 0.0
    home_norm = re.sub(r"[^a-z0-9\s]", "", home_team.lower())
    away_norm = re.sub(r"[^a-z0-9\s]", "", away_team.lower())
    if all(token in text for token in [home_norm.split()[0], away_norm.split()[0]]):
        return 1.0
    if home_norm.split()[0] in text and away_norm.split()[-1] in text:
        return 0.8
    return 0.0


def _fallback_match_by_title(
    markets: List[Dict[str, Any]],
    home_team: str,
    away_team: str,
    game_time: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    if not markets:
        return None
    best = None
    best_score = 0.0
    for m in markets:
        title = m.get("title") or ""
        score_team = _title_match_score(title, home_team, away_team)
        if score_team <= 0:
            continue
        score_time = 0.0
        if game_time and m.get("close_time"):
            try:
                dt = datetime.fromisoformat(str(m.get("close_time")).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt)
                delta_hours = abs((dt - game_time).total_seconds()) / 3600.0
                score_time = max(0.0, 1.0 - min(delta_hours / 72.0, 1.0))
            except Exception:
                score_time = 0.0
        final = 0.8 * score_team + 0.2 * score_time
        if final > best_score:
            best_score = final
            best = m
    return best

# --- IMPORTS FROM APP_CORE ---
try:
    from app_core.team_name_matcher import TeamNameMatcher
    from app_core.kalshi_integrator import (
        KalshiMatchResult,
        match_game_to_kalshi,
    )
    from app_core.vertex_ai_endpoint import (
        VERTEX_FEATURE_COLUMNS,
        is_vertex_prediction_configured,
        predict_win_probabilities,
    )

    # Optional LLM assistant (Gemini via Vertex AI) — MUST NOT crash app if broken
    # Optional LLM assistant (Gemini via Vertex AI) — MUST NOT crash app if broken
    try:
        from app_core.llm_assistant import analyze_kalshi_context_with_llm
        LLM_ASSISTANT_AVAILABLE = True
    except Exception as e:  # <-- IMPORTANT (catches SyntaxError too)
        logger.warning(f"LLM assistant not available: {e}")
        analyze_kalshi_context_with_llm = lambda *args, **kwargs: []
        LLM_ASSISTANT_AVAILABLE = False

except ImportError as e:
    logging.warning(f"VertexMasterAnalyzer import warning: {e}")
    # Fallbacks to prevent crash if app_core is missing
    TeamNameMatcher = None
    KalshiMatchResult = dict  # type: ignore
    match_game_to_kalshi = lambda *args, **kwargs: {}  # type: ignore
    VERTEX_FEATURE_COLUMNS = []  # type: ignore
    is_vertex_prediction_configured = lambda: False  # type: ignore
    predict_win_probabilities = lambda *args, **kwargs: []  # type: ignore
    analyze_kalshi_context_with_llm = lambda *args, **kwargs: []  # type: ignore
    LLM_ASSISTANT_AVAILABLE = False

# -------------------------------
# Normalization Helpers
# -------------------------------

def implied_prob_from_american(odds: Optional[float]) -> Optional[float]:
    if odds is None or odds == 0 or pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds: Optional[float]) -> Optional[float]:
    if odds is None or odds == 0 or pd.isna(odds):
        return None
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)

def _clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    try:
        p = float(p)
    except Exception:
        return 0.5
    return max(lo, min(hi, p))

def blended_win_prob(
    *,
    market_prob: Optional[float],
    vertex_prob: Optional[float],
    theover_prob: Optional[float],
    kalshi_prob: Optional[float],
    sentiment_diff: Optional[float],
    selection: str,  # "home" or "away"
    w_vertex: float = 0.40,
    w_theover: float = 0.25,
    w_kalshi: float = 0.20,
    w_sentiment: float = 0.15,
) -> float:

    """
    Produces a final win probability for the selected side ("home"/"away").

    - Uses Vertex/TheOver/Kalshi when available
    - Falls back to market_prob if sources are missing
    - Applies a small sentiment adjustment (bounded)
    """

    # Base fallback
    base_home = market_prob if market_prob is not None else 0.5
    base_home = _clamp(base_home)

    # Normalize inputs (home-side)
    v = _clamp(vertex_prob) if vertex_prob is not None else None
    t = _clamp(theover_prob) if theover_prob is not None else None
    k = _clamp(kalshi_prob) if kalshi_prob is not None else None

    # Sentiment -> convert to a small probability tweak on HOME side
    sd = float(sentiment_diff or 0.0)
    sent_adj = max(-0.08, min(0.08, sd * 0.08))  # cap impact to ±8%
    s = _clamp(base_home + sent_adj)

    # Dynamic reweight (only weight sources that exist)
    weights = []
    parts = []

    if v is not None:
        weights.append(w_vertex); parts.append(v)
    if t is not None:
        weights.append(w_theover); parts.append(t)
    if k is not None:
        weights.append(w_kalshi); parts.append(k)

    # Always include sentiment term (based on base/market)
    weights.append(w_sentiment); parts.append(s)

    denom = sum(weights) if sum(weights) > 0 else 1.0
    blended_home = sum(w * p for w, p in zip(weights, parts)) / denom
    blended_home = _clamp(blended_home)

    if selection == "home":
        return blended_home
    return _clamp(1.0 - blended_home)

# -------------------------------
# MASTER ANALYZER CLASS
# -------------------------------

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
        self.use_kalshi = bool(
            use_kalshi and st.session_state.get("kalshi_enabled", True)
        )
        self._kalshi_league_cache: Dict[str, List[Dict[str, Any]]] = {}

    def filter_kalshi_markets(self, league: str, markets: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.kalshi or not self.use_kalshi:
            return {
                "markets": [],
                "counts": {},
                "single_game_candidates": [],
                "multivariate_bundles": [],
                "fetch_debug": {"error": "kalshi_disabled"},
            }

        self.kalshi.assert_available()
        cache_key = (league or "").upper()
        if cache_key in self._kalshi_league_cache:
            league_markets = self._kalshi_league_cache[cache_key]
        else:
            league_markets = self.kalshi.get_league_markets(
                league, min_prefix_hits=200, max_pages=25
            )
            self._kalshi_league_cache[cache_key] = league_markets
        split = self.kalshi.split_market_kinds(league_markets, league)
        counts = _kalshi_prefix_counts(split.get("single_game_candidates", []))
        meta = getattr(self.kalshi, "last_fetch_meta", {}) or {}
        fetch_debug = {
            "pages": meta.get("pages"),
            "total_markets": meta.get("total_markets"),
            "prefix_hits": meta.get("prefix_hits"),
            "prefix": meta.get("prefix"),
            "single_game_candidates": len(split.get("single_game_candidates", [])),
            "multivariate_bundles": len(split.get("multivariate_bundles", [])),
            "league_markets": len(league_markets),
        }
        if not split.get("single_game_candidates"):
            raise RuntimeError(
                "Kalshi is required but no single-game markets were found for NBA after deep pagination. This is likely an API filtering/pagination issue."
            )
        return {
            "markets": league_markets,
            "counts": counts,
            "single_game_candidates": split.get("single_game_candidates", []),
            "multivariate_bundles": split.get("multivariate_bundles", []),
            "fetch_debug": fetch_debug,
        }

    # -------------------------------
    # MAIN ENTRYPOINT
    # -------------------------------

    def analyze_all_games(
        self, games: List[Dict[str, Any]], league: str = "NBA"
    ) -> pd.DataFrame:
        if not games:
            return pd.DataFrame()

        vertex_enabled = is_vertex_prediction_configured()
        rows: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "games_in": len(games),
            "rows_out": 0,
            "dropped_missing_teams": 0,
            "dropped_no_markets": 0,
            "h2h_found_count": 0,
            "vertex_calls": 0,
            "vertex_failures": 0,
            "vertex_none_count": 0,
            "exceptions_count": 0,
        }
        kalshi_markets: List[Dict[str, Any]] = []
        if self.use_kalshi and self.kalshi:
            try:
                kalshi_info = self.filter_kalshi_markets(league, games)
                kalshi_markets = kalshi_info.get("single_game_candidates", [])
                st.session_state["vertex_kalshi_debug"] = kalshi_info
            except Exception as e:
                st.session_state["vertex_kalshi_debug"] = {"error": str(e)}
                if st.session_state.get("kalshi_required", True):
                    raise
        progress = st.progress(0)

        def _safe_commence_iso(game_dict: Dict[str, Any]) -> Optional[str]:
            ts = (
                game_dict.get("commence_time_iso")
                or game_dict.get("commence_dt")
                or game_dict.get("commence_time")
            )
            if isinstance(ts, datetime):
                try:
                    return ts.isoformat()
                except Exception:
                    return None
            if ts:
                try:
                    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).isoformat()
                except Exception:
                    return None
            return None

        def _extract_primary_odds(game_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Pull primary ML odds from bookmaker payload for baseline rows."""

            home_team = game_dict.get("home_team")
            away_team = game_dict.get("away_team")
            home_ml = game_dict.get("home_ml_odds")
            away_ml = game_dict.get("away_ml_odds")
            best_book = None

            for bm in game_dict.get("bookmakers", []) or []:
                for market in bm.get("markets", []) or []:
                    key = (market.get("key") or market.get("outcome_type") or "").lower()
                    outcomes = market.get("outcomes") or []
                    if key == "h2h" and outcomes:
                        if best_book is None:
                            best_book = bm.get("title") or bm.get("key")
                        for outcome in outcomes:
                            name = outcome.get("name")
                            price = outcome.get("price")
                            if home_team and name == home_team and home_ml is None:
                                home_ml = price
                            elif away_team and name == away_team and away_ml is None:
                                away_ml = price
                        if home_ml is None and away_ml is None and len(outcomes) == 2:
                            prices = [outcomes[0].get("price"), outcomes[1].get("price")]
                            home_ml, away_ml = prices[0], prices[1]
                        break

            implied_home_prob = implied_prob_from_american(home_ml)
            implied_away_prob = implied_prob_from_american(away_ml)
            if implied_home_prob is None and implied_away_prob is not None:
                implied_home_prob = 1 - implied_away_prob
            return {
                "home_ml_odds": home_ml,
                "away_ml_odds": away_ml,
                "implied_home_prob": implied_home_prob,
                "implied_away_prob": implied_away_prob,
                "book": best_book,
            }

        # --- CORRECTED MASTER ANALYSIS LOOP ---
        for idx, g in enumerate(games):
            warnings: List[str] = list(g.get("warnings") or [])
            league_name = g.get("league")
            home = g.get("home_team")
            away = g.get("away_team")
        
            # Define codes immediately
            h_code = nba_abbrev(home)
            a_code = nba_abbrev(away)
        
            commence_iso = g.get("commence_time_iso_utc") or safe_iso(g.get("commence_time_iso"))
            commence_local = fmt_local_time(g.get("commence_time_local"))
            commence_date_local = g.get("commence_date_local") or ""
            
            # 1. CALCULATE EXTERNAL DATA FIRST (Fixes NameErrors)
            vertex_prob_home = get_vertex_prob(g)
            
            home_sent = sentiment_map.get(home, 0.0)
            away_sent = sentiment_map.get(away, 0.0)
            sentiment_diff = home_sent - away_sent
        
            # Kalshi Filtering & Matching
            filtered_markets = filter_kalshi_game_markets(
                kalshi_markets,
                g.get("commence_time_utc"),
                league_name,
                home,
                away,
                h_code,
                a_code,
            )
            
            deduped = {m.get("event_ticker") or m.get("ticker"): m for m in filtered_markets}
            winner_reason_override = "winner_not_in_fetched_markets" if not filtered_markets else None
            
            kalshi_matches, candidate_debug = match_kalshi_market(g, list(deduped.values()), winner_reason_override)
            
            kalshi_winner = kalshi_matches.get("winner", {})
            kalshi_spread = kalshi_matches.get("spread", {})
            kalshi_total = kalshi_matches.get("total", {})

            # 2. PROBABILITY HELPERS
            home_ml = g.get("home_ml_price")
            away_ml = g.get("away_ml_price")
            implied_home = american_to_implied_prob(home_ml)
            implied_away = american_to_implied_prob(away_ml)
            
            # Baseline probability
            market_home_prob = implied_home if implied_home is not None else (1.0 - implied_away if implied_away else 0.5)

            def blended_for_selection(selection_team: str, m_prob_home: Optional[float]) -> float:
                selection_flag = "home" if selection_team == home else "away"
                return blended_win_prob(
                    market_prob=m_prob_home, vertex_prob=vertex_prob_home,
                    theover_prob=None, kalshi_prob=kalshi_winner.get("kalshi_prob"),
                    sentiment_diff=sentiment_diff, selection=selection_flag
                )

            # 3. GENERATE OUTPUT ROWS
            
            # FALLBACK: IF NO ODDS EXIST
            if not (home_ml or g.get("home_spread_point") or g.get("total_point")):
                warnings = list(dict.fromkeys(warnings + ["no_markets"]))
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Commence (UTC)": commence_iso, "Commence (Local)": commence_local,
                    "Local Date": commence_date_local, "Market": "None",
                    "Pick": None, "Implied_Prob": None, "AI_Prob": vertex_prob_home,
                    "Warnings": ";".join(warnings),
                    "kalshi_available": kalshi_winner.get("kalshi_available"),
                    "kalshi_matched": kalshi_winner.get("kalshi_matched"),
                    "kalshi_prob": kalshi_winner.get("kalshi_prob"),
                    "Home_Sentiment": home_sent, "Away_Sentiment": away_sent, "Sentiment_Diff": sentiment_diff,
                })
                master_stats["market_rows_out"] += 1
                continue

            # MONEYLINE ROW
            if home_ml is not None or away_ml is not None:
                pick = home if (implied_home or 0) >= (implied_away or 0) else away
                implied_pick = implied_home if pick == home else implied_away
                ai_prob_row = blended_for_selection(pick, market_home_prob)
                
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Commence (UTC)": commence_iso, "Commence (Local)": commence_local,
                    "Local Date": commence_date_local, "Market": "Moneyline",
                    "Book": g.get("best_ml_book"), "Home_ML": home_ml, "Away_ML": away_ml,
                    "Pick": pick, "Implied_Prob": implied_pick, "AI_Prob": ai_prob_row,
                    "Home_Sentiment": home_sent, "Away_Sentiment": away_sent, "Sentiment_Diff": sentiment_diff,
                    "kalshi_available": kalshi_winner.get("kalshi_available"),
                    "kalshi_matched": kalshi_winner.get("kalshi_matched"),
                    "kalshi_prob": kalshi_winner.get("kalshi_prob"),
                    "kalshi_event_ticker": kalshi_winner.get("kalshi_event_ticker"),
                    "Warnings": ";".join(warnings),
                })
                master_stats["h2h_found"] += 1
                master_stats["market_rows_out"] += 1

            # SPREAD ROW
            if g.get("home_spread_point") is not None:
                spread_pick = home if (g.get("home_spread_price") or 0) >= (g.get("away_spread_price") or 0) else away
                ai_prob_row = blended_for_selection(spread_pick, market_home_prob)
                
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Commence (UTC)": commence_iso, "Commence (Local)": commence_local,
                    "Market": "Spread", "Book": g.get("best_spread_book"),
                    "Pick": spread_pick, "AI_Prob": ai_prob_row,
                    "kalshi_matched": kalshi_spread.get("kalshi_matched"),
                    "kalshi_prob": kalshi_spread.get("kalshi_prob"),
                    "Sentiment_Diff": sentiment_diff,
                })
                master_stats["market_rows_out"] += 1

            # TOTAL ROW
            if g.get("total_point") is not None:
                ai_prob_row = blended_for_selection(home, market_home_prob)
                
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Commence (UTC)": commence_iso, "Commence (Local)": commence_local,
                    "Market": "Total", "Book": g.get("best_total_book"),
                    "Pick": "Over", "AI_Prob": ai_prob_row,
                    "kalshi_matched": kalshi_total.get("kalshi_matched"),
                    "kalshi_prob": kalshi_total.get("kalshi_prob"),
                    "Sentiment_Diff": sentiment_diff,
                })
                master_stats["market_rows_out"] += 1

    # -------------------------------
    # LLM ASSISTANT HELPERS
    # -------------------------------

    def _run_llm_assistant(
            self,
            feats: Dict[str, Any],
            kalshi_info: Optional[Dict[str, Any]],
            league: str,
    ):
        # ---- HARD GUARD: assistant is optional and must never crash ----
        if (not LLM_ASSISTANT_AVAILABLE) or (analyze_kalshi_context_with_llm is None):
            return [], None, None, None

        if not kalshi_info or not kalshi_info.get("kalshi_available"):
            return [], None, None, None

        try:
            context_md = self._build_kalshi_context_for_llm(
                feats, kalshi_info, league
            )
            if not context_md.strip():
                return [], None, None, None

            contracts = analyze_kalshi_context_with_llm(context_md)
            if not contracts:
                return [], None, None, None

            best = max(contracts, key=lambda c: c.get("confidence", 0))
            return (
                contracts,
                best.get("side"),
                best.get("confidence"),
                best.get("reason"),
            )

        except Exception as e:
            logger.warning(f"LLM assistant failed safely: {e}")
            return [], None, None, None

    def _build_kalshi_context_for_llm(
        self,
        feats: Dict[str, Any],
        kalshi_info: Optional[Dict[str, Any]],
        league: str,
    ) -> str:
        if not kalshi_info:
            return ""

        home = feats.get("home_team") or ""
        away = feats.get("away_team") or ""
        kalshi_label = kalshi_info.get("label") or ""
        kalshi_prob = kalshi_info.get("probability")
        
        lines = [
            f"# {league} Game Kalshi Context",
            f"Game: {away} @ {home}",
            f"Kalshi Label: {kalshi_label}",
            f"Kalshi Prob: {kalshi_prob}",
        ]
        return "\n".join(lines)

    # -------------------------------
    # FEATURE BUILDING
    # -------------------------------

    def _get_kalshi_features(
        self,
        game: Dict[str, Any],
        league: str,
        prefetch_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch Kalshi probability and metadata."""
        feats = {
            "kalshi_available": False,
            "kalshi_prob": None,
            "kalshi_label": None,
            "kalshi_status": "no_match",
            "kalshi_ticker": None,
            "kalshi_date": None,
        }

        if prefetch_info and isinstance(prefetch_info, dict):
            matched = prefetch_info.get("matched", False)
            if matched:
                feats["kalshi_available"] = True
                feats["kalshi_prob"] = prefetch_info.get("probability")
                feats["kalshi_label"] = prefetch_info.get("label")
                feats["kalshi_status"] = "matched"
                feats["kalshi_ticker"] = prefetch_info.get("raw_event_id")
                
                k_date = prefetch_info.get("game_date")
                if k_date:
                    feats["kalshi_date"] = (
                        k_date.strftime("%Y-%m-%d")
                        if isinstance(k_date, datetime)
                        else str(k_date)
                    )
            else:
                feats["kalshi_status"] = prefetch_info.get("reason", "no_match")

        return feats

    def build_comprehensive_features(
        self,
        game: Dict[str, Any],
        league: str,
        kalshi_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble all features from game odds, sentiment, Kalshi, etc."""
        
        game_time = game.get("commence_time") or game.get("commence_dt")
        
        features: Dict[str, Any] = {
            "league": league,
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "game_time": game_time,
            "normalization_warnings": game.get("normalization_warnings", []),
        }

        features.update(
            {
                "home_ml_odds": game.get("home_ml_odds"),
                "away_ml_odds": game.get("away_ml_odds"),
                "home_spread": game.get("home_spread"),
                "away_spread": game.get("away_spread"),
                "home_spread_odds": game.get("home_spread_odds"),
                "away_spread_odds": game.get("away_spread_odds"),
                "implied_home_prob": game.get("implied_home_prob"),
            }
        )

        h_sent = float(game.get("home_sentiment", 0) or 0)
        a_sent = float(game.get("away_sentiment", 0) or 0)
        features["sentiment_diff"] = h_sent - a_sent
        features["home_sentiment"] = h_sent
        features["away_sentiment"] = a_sent

        k_feats = self._get_kalshi_features(game, league, kalshi_info)
        features.update(k_feats)

        return features

    def build_vertex_feature_row(self, feats: Dict[str, Any]) -> Dict[str, float]:
        """Flatten features for Vertex AI model."""
        def gv(k: str, d: float = 0.0) -> float:
            try:
                return float(feats.get(k, d) or d)
            except Exception:
                return d

        return {
            "implied_home_prob": gv("implied_home_prob", 0.5),
            "sentiment_diff": gv("sentiment_diff"),
            "kalshi_prob": gv("kalshi_prob", 0.5)
            if feats.get("kalshi_available")
            else 0.5,
        }

    # -------------------------------
    # CANDIDATE EVALUATION
    # -------------------------------

    def _evaluate_candidate(
        self,
        feats: Dict[str, Any],
        market_type: str,
        selection: str,
        line: Optional[float],
        odds: Optional[float],
        game_league: str,
        vertex_enabled: bool,
        vertex_home_prob: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single betting candidate (Home ML, Away Spread, etc).
        """
        if odds is None or pd.isna(odds):
            return None

        market_prob = implied_prob_from_american(odds)
        if market_prob is None:
            return None

        # --- Blended probability (uses Vertex + TheOver + Kalshi + Sentiment) ---
        # Market probability (from sportsbook odds) is our baseline fallback.
        # For ML sources, we treat them as HOME-side probabilities.
        market_home_prob = feats.get("implied_home_prob", None)
        try:
            market_home_prob = float(market_home_prob) if market_home_prob is not None else None
        except Exception:
            market_home_prob = None

        # Vertex model returns home win probability (if enabled)
        vertex_home = None
        if vertex_enabled and vertex_home_prob is not None:
            try:
                vertex_home = float(vertex_home_prob)
            except Exception:
                vertex_home = None

        # TheOver: if you store one, it should also be HOME-side.
        # If you don’t have this yet, it will just be None and get ignored.
        theover_home = feats.get("theover_home_prob", None)
        try:
            theover_home = float(theover_home) if theover_home is not None else None
        except Exception:
            theover_home = None

        # Kalshi prob: your feats["kalshi_prob"] should be HOME-side probability
        # (or at least "home favored" probability for wins markets). If it’s not,
        # keep it None until you align the meaning.
        kalshi_home = feats.get("kalshi_prob", None)
        try:
            kalshi_home = float(kalshi_home) if kalshi_home is not None else None
        except Exception:
            kalshi_home = None

        sentiment_diff = feats.get("sentiment_diff", 0.0)

        ai_prob = blended_win_prob(
            market_prob=market_home_prob,
            vertex_prob=vertex_home,
            theover_prob=theover_home,
            kalshi_prob=kalshi_home if feats.get("kalshi_available") else None,
            sentiment_diff=sentiment_diff,
            selection=selection,  # "home" or "away"
        )

        # Edge vs THIS market (ML or Spread) is computed vs the market implied prob for THIS candidate
        edge = ai_prob - market_prob

        dec_odds = american_to_decimal(odds)
        ev = (
            (ai_prob * (dec_odds - 1.0) - (1.0 - ai_prob))
            if dec_odds
            else 0.0
        )

        team = feats.get("home_team") if selection == "home" else feats.get("away_team")
        if market_type == "ML":
            pick_text = f"{team} ML"
        elif market_type == "Spread":
            pick_text = f"{team} {line:+.1f}"
        else:
            pick_text = f"{selection} {line}"
        
        g_time = feats.get("game_time")
        
        result: Dict[str, Any] = {
            "league": game_league,
            "game": f"{feats.get('away_team')} @ {feats.get('home_team')}",
            "game_time": g_time,
            "the_pick": pick_text,
            "pick_odds": odds,
            "win_prob": ai_prob,
            "market_prob": market_prob,
            "edge_vs_market": edge,
            "ev": ev,
            "market_type": market_type,
            "selection": selection,
            "line": line,
            # Kalshi fields
            "kalshi_available": bool(feats.get("kalshi_available", False)),
            "kalshi_prob": feats.get("kalshi_prob"),
            "kalshi_status": feats.get("kalshi_status", ""),
            "kalshi_ticker": feats.get("kalshi_ticker"),
            "kalshi_date": feats.get("kalshi_date"),
        }

        result["final_win_prob"] = ai_prob
        result["vertex_home_prob"] = vertex_home
        result["theover_home_prob"] = theover_home
        result["kalshi_home_prob"] = kalshi_home if feats.get("kalshi_available") else None
        result["market_home_prob"] = market_home_prob

        return result

    def _standardize_row(
        self,
        best: Dict[str, Any],
        feats: Dict[str, Any],
        game_league: str,
        warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize candidate dict into UI-friendly row with fallbacks."""

        g_time = feats.get("game_time")
        commence_utc = None
        if isinstance(g_time, datetime):
            if g_time.tzinfo is None:
                g_time = pytz.utc.localize(g_time)
            commence_utc = g_time.astimezone(pytz.utc)
        elif g_time:
            try:
                commence_utc = datetime.fromisoformat(str(g_time).replace("Z", "+00:00"))
            except Exception:
                commence_utc = None

        warn_parts = []
        if warning:
            warn_parts.append(warning)
        warn_parts.extend(feats.get("normalization_warnings", []))
        warn_text = "; ".join([p for p in warn_parts if p])
        row = {
            "League": best.get("league", game_league),
            "Home": feats.get("home_team"),
            "Away": feats.get("away_team"),
            "Commence (UTC)": commence_utc,
            "Market": best.get("market_type", "ML"),
            "Pick": best.get("the_pick"),
            "Implied_Prob": best.get("market_prob", feats.get("implied_home_prob")),
            "AI_Prob": best.get("win_prob"),
            "AI_Edge": best.get("edge_vs_market"),
            "Kalshi_Prob": best.get("kalshi_home_prob"),
            "Kalshi_Edge": None,
            "TheOver_Pick": None,
            "TheOver_Prob": best.get("theover_home_prob"),
            "Warnings": warn_text,
        }

        # Retain legacy keys for CSV/download consumers
        row.update(best)
        return row

    def _fallback_row(
        self,
        feats: Dict[str, Any],
        game_league: str,
        vertex_home_prob: Optional[float],
        warning: str,
    ) -> Dict[str, Any]:
        """Guarantee at least one row per game even when odds/candidates missing."""

        pick_side = "home" if (vertex_home_prob or 0.5) >= 0.5 else "away"
        pick_team = feats.get("home_team") if pick_side == "home" else feats.get("away_team")
        best = {
            "league": game_league,
            "game": f"{feats.get('away_team')} @ {feats.get('home_team')}",
            "game_time": feats.get("game_time"),
            "the_pick": f"{pick_team} (no odds)",
            "pick_odds": None,
            "win_prob": vertex_home_prob,
            "market_prob": None,
            "edge_vs_market": None,
            "ev": None,
            "market_type": "Info",
            "selection": pick_side,
            "line": None,
            "kalshi_available": bool(feats.get("kalshi_available", False)),
            "kalshi_prob": feats.get("kalshi_prob"),
            "kalshi_status": feats.get("kalshi_status", ""),
            "kalshi_ticker": feats.get("kalshi_ticker"),
            "kalshi_date": feats.get("kalshi_date"),
            "final_win_prob": vertex_home_prob,
            "vertex_home_prob": vertex_home_prob,
            "theover_home_prob": feats.get("theover_home_prob"),
            "kalshi_home_prob": feats.get("kalshi_prob")
            if feats.get("kalshi_available")
            else None,
            "market_home_prob": feats.get("implied_home_prob"),
        }
        return self._standardize_row(best, feats, game_league, warning)

# -------------------------------
# STREAMLIT DISPLAY
# -------------------------------

def show_vertex_master_analysis(results_df: pd.DataFrame) -> None:
    """
    Render the Vertex AI Master Analysis results.
    """
    if results_df is None or results_df.empty:
        st.info("No games to analyze.")
        return

    st.subheader("🏆 Vertex AI Master Analysis")

    display_df = results_df.copy()

    display_df["Win %"] = (display_df["win_prob"] * 100).round(1)
    display_df["Edge %"] = (display_df["edge_vs_market"] * 100).round(1)

    # Format Commence Time to show EST/EDT
    def _fmt_commence(x):
        if not x:
            return ""
        try:
            target_tz = pytz.timezone('US/Eastern')
            if isinstance(x, str):
                dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
                return dt.astimezone(target_tz).strftime("%Y-%m-%d %I:%M %p")
            if isinstance(x, datetime):
                # Ensure it's aware before converting
                if x.tzinfo is None:
                    x = pytz.utc.localize(x)
                return x.astimezone(target_tz).strftime("%Y-%m-%d %I:%M %p")
            return str(x)
        except Exception:
            return str(x)

    if "game_time" in display_df.columns:
        display_df["Commence (ET)"] = display_df["game_time"].apply(_fmt_commence)
    else:
        display_df["Commence (ET)"] = ""

    # Kalshi display column
    def fmt_kalshi(row: pd.Series) -> str:
        if not row.get("kalshi_available"):
            return "No Match"
        prob = row.get("kalshi_prob")
        if prob is None or pd.isna(prob):
            return "Matched"
        try:
            return f"Matched ({float(prob) * 100:.1f}%)"
        except Exception:
            return "Matched"

    display_df["Kalshi"] = display_df.apply(fmt_kalshi, axis=1)

    if "kalshi_status" in display_df.columns:
        display_df["Kalshi Match Debug"] = display_df["kalshi_status"]
    else:
        display_df["Kalshi Match Debug"] = ""

    cols = [
        "game",
        "Commence (ET)",
        "the_pick",
        "pick_odds",
        "Win %",
        "Edge %",
        "ev",
        "Kalshi",
        "Kalshi Match Debug",
    ]

    # Filter only columns that actually exist to avoid KeyError
    existing_cols = [c for c in cols if c in display_df.columns]
    st.dataframe(display_df[existing_cols], width="stretch")

    # CSV export
    export_cols = [
        "league", "game", "game_time", "the_pick", "pick_odds",
        "win_prob", "final_win_prob", "market_prob", "market_home_prob",
        "vertex_home_prob", "theover_home_prob", "kalshi_home_prob",
        "edge_vs_market", "ev",
        "kalshi_available", "kalshi_prob", "kalshi_status",
        "kalshi_ticker", "kalshi_date"
    ]

    export_df = results_df.copy()
    for c in export_cols:
        if c not in export_df.columns:
            export_df[c] = None

    csv_bytes = export_df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Vertex Master Analysis CSV",
        data=csv_bytes,
        file_name="vertex_master_analysis.csv",
        mime="text/csv",
        key="vertex_master_analysis_csv_btn",
    )
