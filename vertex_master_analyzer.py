"""
Vertex AI Master Analyzer
Location: vertex_master_analyzer.py (ROOT DIRECTORY)
Consolidates ALL data sources for ultimate best bet recommendations.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

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
    # Optional LLM assistant (e.g., Gemini wrapper)
    try:
        from app_core.llm_assistant import analyze_kalshi_context_with_llm

        LLM_ASSISTANT_AVAILABLE = True
    except ImportError as _llm_import_err:
        logger.warning(f"LLM assistant not available: {_llm_import_err}")
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

        # Check Kalshi availability once
        kalshi_active = bool(self.kalshi and self.use_kalshi)

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

                # 2. Kalshi Prefetch (single, best match)
                kalshi_info: Optional[Dict[str, Any]] = None
                if kalshi_active:
                    try:
                        g_time = game.get("commence_time")
                        g_dt: Optional[datetime] = None
                        if g_time:
                            # Handle string or datetime
                            if isinstance(g_time, str):
                                g_dt = datetime.fromisoformat(
                                    str(g_time).replace("Z", "+00:00")
                                )
                            elif isinstance(g_time, datetime):
                                g_dt = g_time

                        # Use status=None to find markets even if closed/locked
                        kalshi_info = match_game_to_kalshi(
                            game_league,
                            game.get("home_team", ""),
                            game.get("away_team", ""),
                            g_dt,
                            integrator=self.kalshi,
                            status=None,
                        )
                    except Exception as e:
                        logger.warning(f"Kalshi prefetch error: {e}")

                # 3. Build Features (including Kalshi flags/metadata)
                feats = self.build_comprehensive_features(
                    game, game_league, kalshi_info
                )

                # 4. Optional LLM Assistant (placeholder AI reasoning)
                #    This does NOT replace Vertex; it adds commentary/extra columns.
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
                    # Pick best candidate by edge
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
        """
        Call the optional LLM assistant (e.g., Gemini wrapper) to get a
        second-opinion recommendation based on Kalshi info + basic odds.

        Returns:
            (contracts_list, best_side, best_confidence, best_reason)
        """
        if not LLM_ASSISTANT_AVAILABLE:
            return [], None, None, None

        # Only bother if we have a Kalshi match
        if not kalshi_info or not kalshi_info.get("kalshi_available"):
            return [], None, None, None

        try:
            context_md = self._build_kalshi_context_for_llm(feats, kalshi_info, league)
            if not context_md.strip():
                return [], None, None, None

            contracts = analyze_kalshi_context_with_llm(context_md)
            if not contracts:
                return [], None, None, None

            # Choose the highest-confidence suggestion
            best = max(contracts, key=lambda c: c.get("confidence", 0))
            best_side = best.get("side")
            best_conf = best.get("confidence")
            best_reason = best.get("reason")

            return contracts, best_side, best_conf, best_reason

        except Exception as e:
            logger.warning(f"LLM assistant failed for game {feats.get('game_time')}: {e}")
            return [], None, None, None

    def _build_kalshi_context_for_llm(
        self,
        feats: Dict[str, Any],
        kalshi_info: Optional[Dict[str, Any]],
        league: str,
    ) -> str:
        """
        Build a markdown context string for the LLM assistant, summarizing:
          - League, teams, game time
          - Sportsbook odds (ML + spreads)
          - Kalshi probability and label if available
        """
        if not kalshi_info:
            return ""

        home = feats.get("home_team") or ""
        away = feats.get("away_team") or ""
        game_time = feats.get("game_time") or ""
        kalshi_label = kalshi_info.get("label") or ""
        kalshi_prob = kalshi_info.get("probability")
        kalshi_status = kalshi_info.get("reason") or ""
        kalshi_ticker = kalshi_info.get("raw_event_id") or ""

        lines: List[str] = []
        lines.append(f"# {league} Game Kalshi Context")
        lines.append("")
        lines.append(f"Game: {away} @ {home}")
        lines.append(f"Game Time (UTC/ISO): {game_time}")
        lines.append("")

        # Sportsbook odds snapshot
        lines.append("## Sportsbook Odds")
        lines.append(f"- Home ML Odds: {feats.get('home_ml_odds')}")
        lines.append(f"- Away ML Odds: {feats.get('away_ml_odds')}")
        lines.append(
            f"- Home Spread: {feats.get('home_spread')} "
            f"(odds {feats.get('home_spread_odds')})"
        )
        lines.append(
            f"- Away Spread: {feats.get('away_spread')} "
            f"(odds {feats.get('away_spread_odds')})"
        )
        lines.append("")

        # Kalshi info
        lines.append("## Kalshi Market")
        lines.append(f"- Ticker: {kalshi_ticker}")
        lines.append(f"- Label: {kalshi_label}")
        if kalshi_prob is not None:
            try:
                pct = float(kalshi_prob) * 100
                lines.append(f"- Kalshi Implied Probability: {pct:.1f}%")
            except Exception:
                lines.append(f"- Kalshi Implied Probability: {kalshi_prob}")
        else:
            lines.append("- Kalshi Implied Probability: None")
        lines.append(f"- Kalshi Status: {kalshi_status}")
        lines.append("")

        # Simple instructions for the assistant (high-level)
        lines.append(
            "Given the sportsbook odds and the Kalshi market info above, "
            "identify whether the home side or away side appears underpriced "
            "and explain why. Return your answer as JSON with contracts, "
            "each containing: ticker, side ('home'/'away' or 'yes'/'no'), "
            "bid_price (0-100), reason, and confidence (0-100)."
        )

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

        # If we passed prefetch_info, use it directly
        if prefetch_info and isinstance(prefetch_info, dict):
            matched = prefetch_info.get("matched", False)
            if matched:
                feats["kalshi_available"] = True
                feats["kalshi_prob"] = prefetch_info.get("probability")
                feats["kalshi_label"] = prefetch_info.get("label")
                feats["kalshi_status"] = "matched"
                feats["kalshi_ticker"] = prefetch_info.get("raw_event_id")

                # Format date if available
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
        features: Dict[str, Any] = {
            "league": league,
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "game_time": game.get("commence_time"),
        }

        # Odds
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

        # Sentiment (mock or real)
        h_sent = float(game.get("home_sentiment", 0) or 0)
        a_sent = float(game.get("away_sentiment", 0) or 0)
        features["sentiment_diff"] = h_sent - a_sent
        features["home_sentiment"] = h_sent
        features["away_sentiment"] = a_sent

        # Kalshi
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
        Computes:
          - Market probability
          - AI probability (Vertex or heuristic)
          - Edge vs market
          - EV
        And attaches Kalshi + LLM assistant metadata.
        """
        if odds is None or pd.isna(odds):
            return None

        market_prob = implied_prob_from_american(odds)
        if not market_prob:
            return None

        # Determine AI probability
        ai_prob: Optional[float] = None
        if vertex_enabled and vertex_home_prob is not None:
            ai_prob = vertex_home_prob if selection == "home" else (1.0 - vertex_home_prob)
        else:
            # Fallback heuristic if Vertex isn't configured/available
            base = feats.get("implied_home_prob", 0.5)
            # Adjust by sentiment
            sent_adj = (feats.get("sentiment_diff", 0) or 0) * 0.1
            adj_base = base + sent_adj
            ai_prob = adj_base if selection == "home" else (1.0 - adj_base)

        ai_prob = max(0.01, min(0.99, ai_prob))
        edge = ai_prob - market_prob

        # EV Calc
        dec_odds = american_to_decimal(odds)
        ev = (
            (ai_prob * (dec_odds - 1.0) - (1.0 - ai_prob))
            if dec_odds
            else 0.0
        )

        # Construct Label
        team = feats.get("home_team") if selection == "home" else feats.get("away_team")
        if market_type == "ML":
            pick_text = f"{team} ML"
        elif market_type == "Spread":
            pick_text = f"{team} {line:+.1f}"
        else:
            pick_text = f"{selection} {line}"

        # Base result row
        result: Dict[str, Any] = {
            "league": game_league,
            "game": f"{feats.get('away_team')} @ {feats.get('home_team')}",
            "the_pick": pick_text,
            "pick_odds": odds,
            "win_prob": ai_prob,
            "market_prob": market_prob,
            "edge_vs_market": edge,
            "ev": ev,
            "kalshi_available": bool(feats.get("kalshi_available", False)),
            "kalshi_prob": feats.get("kalshi_prob"),          # 0–1 float if matched
            "kalshi_status": feats.get("kalshi_status", ""),  # human-readable reason
        }) 
            # Kalshi Metadata for Export
            "kalshi_available": feats.get("kalshi_available"),
            "kalshi_prob": feats.get("kalshi_prob"),
            "kalshi_ticker": feats.get("kalshi_ticker"),
            "kalshi_date": feats.get("kalshi_date"),
            "kalshi_status": feats.get("kalshi_status"),
        }

        # Attach LLM assistant metadata (game-level)
        result["assistant_contracts"] = feats.get("assistant_contracts")
        result["assistant_best_side"] = feats.get("assistant_best_side")
        result["assistant_confidence"] = feats.get("assistant_confidence")
        result["assistant_reason"] = feats.get("assistant_reason")

        return result


# -------------------------------
# STREAMLIT DISPLAY
# -------------------------------

def show_vertex_master_analysis(results_df: pd.DataFrame) -> None:
    """
    Render the Vertex AI Master Analysis results, including Kalshi metadata
    and a CSV export with debug columns.
    """
    if results_df is None or results_df.empty:
        st.info("No games to analyze.")
        return

    st.subheader("🏆 Vertex AI Master Analysis")

    display_df = results_df.copy()

    # Friendly display columns
    display_df["Win %"] = (display_df["win_prob"] * 100).round(1)
    display_df["Edge %"] = (display_df["edge_vs_market"] * 100).round(1)

    # --- Kalshi display column ------------------------------------------
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

    # --- Debug column exposes the matcher reason/status -----------------
    if "kalshi_status" in display_df.columns:
        display_df["Kalshi Match Debug"] = display_df["kalshi_status"]
    else:
        display_df["Kalshi Match Debug"] = ""

    # Columns for the on-screen grid
    cols = [
        "game",
        "the_pick",
        "pick_odds",
        "Win %",
        "Edge %",
        "ev",
        "Kalshi",
        "Kalshi Match Debug",
    ]
    st.dataframe(display_df[cols], use_container_width=True)

    # --- CSV export with raw Kalshi fields ------------------------------
    export_cols = [
        "league",
        "game",
        "the_pick",
        "pick_odds",
        "win_prob",
        "market_prob",
        "edge_vs_market",
        "ev",
        "kalshi_available",
        "kalshi_prob",
        "kalshi_status",
        "kalshi_ticker",
        "kalshi_date",
    ]

    export_df = results_df.copy()
    # Ensure all export columns exist
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
