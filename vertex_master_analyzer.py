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
    try:
        from app_core.llm_assistant import analyze_kalshi_context_with_llm

        LLM_ASSISTANT_AVAILABLE = True
    except Exception as e:  # catches ImportError + SyntaxError + anything else during import
        logger.warning(f"LLM assistant not available: {e}")
        analyze_kalshi_context_with_llm = lambda *args, **kwargs: []  # type: ignore
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
        progress = st.progress(0)

        kalshi_active = bool(self.kalshi and self.use_kalshi)
        
        # Define target timezone for consistent date matching
        target_tz = pytz.timezone('US/Eastern')

        for idx, game in enumerate(games):
            try:
                # 1. League Normalization
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

                # 2. Kalshi Prefetch with Timezone Conversion
                kalshi_info: Optional[Dict[str, Any]] = None
                if kalshi_active:
                    try:
                        g_time_raw = game.get("commence_time")
                        g_dt: Optional[datetime] = None
                        
                        if g_time_raw:
                            if isinstance(g_time_raw, str):
                                g_dt = datetime.fromisoformat(g_time_raw.replace("Z", "+00:00"))
                            elif isinstance(g_time_raw, datetime):
                                g_dt = g_time_raw
                            
                            # Convert to US/Eastern for better matching with Kalshi markets
                            if g_dt and g_dt.tzinfo:
                                g_dt = g_dt.astimezone(target_tz)
                            elif g_dt:
                                g_dt = pytz.utc.localize(g_dt).astimezone(target_tz)

                        # Use status=None to find markets even if closed/locked
                        raw_kalshi_result = match_game_to_kalshi(
                            game_league,
                            game.get("home_team", ""),
                            game.get("away_team", ""),
                            g_dt,
                            integrator=self.kalshi,
                            status=None,
                        )
                        kalshi_info = asdict(raw_kalshi_result) if raw_kalshi_result else None
                    
                    except Exception as e:
                        logger.warning(f"Kalshi prefetch error for {game.get('home_team')}: {e}")

                # 3. Build Features (including Kalshi flags/metadata)
                feats = self.build_comprehensive_features(
                    game, game_league, kalshi_info
                )

                # 4. Optional LLM Assistant
                (
                    assistant_contracts,
                    assistant_best_side,
                    assistant_confidence,
                    assistant_reason,
                ) = self._run_llm_assistant(feats, kalshi_info, game_league)

                feats["assistant_contracts"] = assistant_contracts
                feats["assistant_best_side"] = assistant_best_side
                feats["assistant_confidence"] = assistant_confidence
                feats["assistant_reason"] = assistant_reason

                # 5. Vertex Prediction
                vertex_home_prob: Optional[float] = None
                if vertex_enabled:
                    try:
                        feat_row = self.build_vertex_feature_row(feats)
                        probs = predict_win_probabilities(
                            pd.DataFrame([feat_row]), VERTEX_FEATURE_COLUMNS
                        )
                        if probs and len(probs) > 0:
                            vertex_home_prob = float(probs[0])
                    except Exception as e:
                        logger.warning(f"Vertex prediction failed: {e}")

                # 6. Evaluate Candidates (Moneyline & Spread)
                candidates: List[Dict[str, Any]] = []

                # Moneyline
                for sel, odds in [
                    ("home", feats.get("home_ml_odds")),
                    ("away", feats.get("away_ml_odds")),
                ]:
                    c = self._evaluate_candidate(
                        feats,
                        "ML",
                        sel,
                        None,
                        odds,
                        game_league,
                        vertex_enabled,
                        vertex_home_prob,
                    )
                    if c:
                        candidates.append(c)

                # Spread
                for sel, line, odds in [
                    ("home", feats.get("home_spread"), feats.get("home_spread_odds")),
                    ("away", feats.get("away_spread"), feats.get("away_spread_odds")),
                ]:
                    c = self._evaluate_candidate(
                        feats,
                        "Spread",
                        sel,
                        line,
                        odds,
                        game_league,
                        vertex_enabled,
                        vertex_home_prob,
                    )
                    if c:
                        candidates.append(c)

                if candidates:
                    best = sorted(
                        candidates,
                        key=lambda x: x.get("edge_vs_market", -99),
                        reverse=True,
                    )[0]
                    rows.append(best)

            except Exception as e:
                logger.error(f"Error analyzing game {idx}: {e}")

            progress.progress((idx + 1) / len(games))

        return pd.DataFrame(rows)

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
        if not x: return ""
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
        except:
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
        "Commence (ET)",  # Changed header to ET
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
