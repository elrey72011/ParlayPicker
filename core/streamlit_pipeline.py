from __future__ import annotations

import logging
import os
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.bankroll_simulator import simulate_bankroll
from core.kelly_optimizer import add_kelly_bet_sizing
from core.parlay_engine import generate_parlays as generate_parlay_candidates
from core.probability_calibration import calibrate_probabilities
from core.probability_engine import american_to_prob, normalize_probability_components, remove_vig
from core.schema.base_schema import ensure_base_schema
from core.team_mapper import normalize_team_name

logger = logging.getLogger(__name__)


try:
    from complete_workflow_implementation import run_ml_predictions
except Exception:  # pragma: no cover
    run_ml_predictions = None


MERGE_KEYS = ["league", "home_team", "away_team", "game_date"]
MODEL_PATH = "models/sports_model_latest.joblib"
SPORT_ALIASES = {
    "NBA": "NBA",
    "NHL": "NHL",
    "NCAAM": "NCAAB",
    "NCAA MEN'S BASKETBALL": "NCAAB",
    "NCAA MENS BASKETBALL": "NCAAB",
    "NCAA BASKETBALL": "NCAAB",
    "COLLEGE BASKETBALL": "NCAAB",
}
BEST_PICK_COLUMNS = [
    "league",
    "home_team",
    "away_team",
    "game_date",
    "best_pick",
    "calibrated_probability",
    "expected_value",
    "edge",
    "odds_american",
    "market_probability",
    "ml_probability",
]


def _infer_market_type(row: pd.Series) -> str:
    allowed_market_types = {
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    }

    existing_market_type = str(row.get("market_type") or "").strip().lower()
    if existing_market_type in allowed_market_types:
        return existing_market_type

    market_hint = " ".join(
        [
            str(row.get("market") or ""),
            str(row.get("bet_type") or ""),
            str(row.get("wager_type") or ""),
            str(row.get("pick_type") or ""),
            str(row.get("pick") or ""),
            str(row.get("side") or ""),
            str(row.get("over_under") or ""),
            str(row.get("selection") or ""),
        ]
    ).lower()

    spread_candidates = [row.get("spread"), row.get("spread_line"), row.get("line")]
    total_candidates = [row.get("total"), row.get("total_line"), row.get("total_points"), row.get("points")]
    spread_val = pd.Series(spread_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    total_val = pd.Series(total_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    spread_num = spread_val.iloc[0] if not spread_val.empty else np.nan
    total_num = total_val.iloc[0] if not total_val.empty else np.nan

    pick_team = str(row.get("team") or row.get("selection") or row.get("pick") or "").strip().lower()
    home_team = str(row.get("home_team") or "").strip().lower()
    away_team = str(row.get("away_team") or "").strip().lower()
    is_home_pick = bool(row.get("is_home_pick", False))
    if pick_team and home_team:
        is_home_pick = pick_team == home_team
    elif pick_team and away_team:
        is_home_pick = pick_team != away_team

    has_over_under_text = any(token in market_hint for token in ["over", "under", "o/u", "ou"])
    has_total_text = any(token in market_hint for token in ["total", "over", "under", "o/u", "points"])
    has_spread_text = any(token in market_hint for token in ["spread", "ats", "handicap"])

    if "under" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_under"
    if "over" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_over"

    if has_spread_text or (pd.notna(spread_num) and not has_over_under_text):
        return "spread_home" if is_home_pick else "spread_away"

    if pd.notna(total_num) and has_over_under_text:
        return "total_under" if "under" in market_hint else "total_over"

    if has_total_text or pd.notna(total_num):
        return "total_over"

    if pd.notna(spread_num):
        return "spread_home" if is_home_pick else "spread_away"

    return "unknown"


def format_pick(row: pd.Series) -> str:
    def _format_signed_spread(value: float, invert_sign: bool = False) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return ""
        if invert_sign:
            numeric = -numeric
        return f"{numeric:+.1f}"

    def _format_total(value: float) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return ""
        return f"{numeric:.1f}"

    if row["market_type"] == "spread_home":
        spread_display = _format_signed_spread(row.get("spread"))
        return f"{row['home_team']} {spread_display}".strip()

    if row["market_type"] == "spread_away":
        spread_display = _format_signed_spread(row.get("spread"), invert_sign=True)
        return f"{row['away_team']} {spread_display}".strip()

    if row["market_type"] == "total_over":
        return f"Over {_format_total(row.get('total'))}".strip()

    if row["market_type"] == "total_under":
        return f"Under {_format_total(row.get('total'))}".strip()

    return ""


def _build_best_picks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "market_type" not in df.columns:
        df["market_type"] = ""

    if "spread" not in df.columns:
        df["spread"] = pd.to_numeric(df.get("spread_line"), errors="coerce")
    else:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")

    if "total" not in df.columns:
        df["total"] = pd.to_numeric(df.get("total_line"), errors="coerce")
    else:
        df["total"] = pd.to_numeric(df["total"], errors="coerce")

    df["inferred_market_type"] = df.apply(_infer_market_type, axis=1)
    df["market_type"] = df["inferred_market_type"]
    df["source_has_spread"] = df[[c for c in ["spread", "spread_line", "line"] if c in df.columns]].apply(
        lambda row: row.apply(pd.to_numeric, errors="coerce").notna().any(), axis=1
    ) if any(c in df.columns for c in ["spread", "spread_line", "line"]) else False
    df["source_has_total"] = df[[c for c in ["total", "total_line", "total_points", "points"] if c in df.columns]].apply(
        lambda row: row.apply(pd.to_numeric, errors="coerce").notna().any(), axis=1
    ) if any(c in df.columns for c in ["total", "total_line", "total_points", "points"]) else False

    logger.info(
        "Best-pick inference debug: inferred_market_type_counts=%s source_has_spread=%s source_has_total=%s",
        df["inferred_market_type"].value_counts(dropna=False).to_dict(),
        int(pd.to_numeric(df["source_has_spread"], errors="coerce").fillna(False).astype(bool).sum()),
        int(pd.to_numeric(df["source_has_total"], errors="coerce").fillna(False).astype(bool).sum()),
    )

    allowed_market_types = {"spread_home", "spread_away", "total_over", "total_under"}
    df = df[df["market_type"].isin(allowed_market_types)].copy()
    if df.empty:
        best_picks = pd.DataFrame(columns=BEST_PICK_COLUMNS)
        return best_picks

    group_keys = ["league", "home_team", "away_team", "game_date"]
    available_group_keys = [k for k in group_keys if k in df.columns]
    if not available_group_keys:
        available_group_keys = ["home_team", "away_team"]

    sort_col = "expected_value" if "expected_value" in df.columns else "edge"
    best_picks = (
        df.sort_values(sort_col, ascending=False)
        .groupby(available_group_keys)
        .first()
        .reset_index()
    )
    best_picks["best_pick"] = best_picks.apply(format_pick, axis=1)
    best_picks = best_picks[best_picks["best_pick"].astype(str).str.len() > 0].copy()

    for col in BEST_PICK_COLUMNS:
        if col not in best_picks.columns:
            best_picks[col] = pd.NA

    return best_picks[BEST_PICK_COLUMNS]


def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Build one spread/total best-pick row per game from a raw analysis dataframe."""
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)
    return _build_best_picks(analysis_df)


def normalize_merge_keys(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    df = df.copy()

    if "league" in df.columns:
        df["league"] = df["league"].astype(str)

    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].astype(str).str.strip().str.lower()

    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype(str).str.strip().str.lower()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    return df


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("ML model not found, using market probabilities.")
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print("Model load failed:", e)
        return None


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1


def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["home_team", "away_team", "team"]:
        if col in df.columns:
            df.loc[:, col] = df[col].apply(normalize_team_name)
    return df


def _normalize_league_value(value: str | object) -> str:
    if pd.isna(value):
        return ""
    normalized = str(value).strip().upper()
    return SPORT_ALIASES.get(normalized, normalized)


def _normalize_sports_filter(sports: Iterable[str] | None) -> list[str]:
    if not sports:
        return []
    return [_normalize_league_value(sport) for sport in sports]


def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {"sport": "league", "date": "game_date", "commence_time": "game_date"}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "game_date" not in df.columns:
        df["game_date"] = pd.NaT

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    if "league" not in df.columns:
        df["league"] = ""
    df["league"] = df["league"].apply(_normalize_league_value)
    return _normalize_teams(df)


def _infer_uploaded_league_row(row: pd.Series, selected_sports: list[str] | None = None) -> str:
    current = _normalize_league_value(row.get("league"))
    if current:
        return current

    selected_set = set(selected_sports or [])
    context_text = " ".join(
        [
            str(row.get("market") or ""),
            str(row.get("bet_type") or ""),
            str(row.get("wager_type") or ""),
            str(row.get("pick_type") or ""),
            str(row.get("source") or ""),
            str(row.get("source_file") or ""),
            str(row.get("filename") or ""),
        ]
    ).upper()

    if "NHL" in context_text and (not selected_set or "NHL" in selected_set):
        return "NHL"
    if any(token in context_text for token in ["NCAAB", "NCAAM", "COLLEGE BASKETBALL", "NCAA"]) and (
        not selected_set or "NCAAB" in selected_set
    ):
        return "NCAAB"
    if "NBA" in context_text and (not selected_set or "NBA" in selected_set):
        return "NBA"

    if len(selected_set) == 1:
        return next(iter(selected_set))

    return ""


def _enrich_uploaded_league(df: pd.DataFrame, selected_sports: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    enriched = df.copy()
    if "league" not in enriched.columns:
        enriched["league"] = ""
    enriched["league"] = enriched.apply(lambda row: _infer_uploaded_league_row(row, selected_sports), axis=1)
    enriched["league"] = enriched["league"].apply(_normalize_league_value)
    return enriched


def _resolve_american_odds(row: pd.Series) -> float:
    for col in ["odds_american", "home_odds", "odds"]:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                continue
    return -110.0


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame | None, suffix: str) -> pd.DataFrame:
    if right is None or right.empty:
        return left

    left = normalize_merge_keys(_normalize_key_columns(left))
    right = normalize_merge_keys(_normalize_key_columns(right))

    keys = [k for k in MERGE_KEYS if k in left.columns and k in right.columns]
    if not keys:
        return left

    right = right.drop_duplicates(subset=keys)

    merged = left.merge(
        right,
        on=keys,
        how="left",
        suffixes=("", suffix),
    )

    return merged


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = ensure_base_schema(df)

    df["odds_american"] = df.apply(_resolve_american_odds, axis=1)

    # 1) Market probability from each row's odds
    df["market_prob"] = pd.to_numeric(df["odds_american"], errors="coerce").apply(american_to_prob)

    # 2) Remove vig when both sides are available
    if {"home_odds", "away_odds"}.issubset(df.columns):
        home_prob = pd.to_numeric(df["home_odds"], errors="coerce").apply(lambda x: american_to_prob(x) if pd.notna(x) else pd.NA)
        away_prob = pd.to_numeric(df["away_odds"], errors="coerce").apply(lambda x: american_to_prob(x) if pd.notna(x) else pd.NA)
        no_vig = pd.DataFrame(
            [remove_vig(h, a) if pd.notna(h) and pd.notna(a) else (pd.NA, pd.NA) for h, a in zip(home_prob, away_prob)]
        )

        if "is_home_pick" in df.columns:
            is_home_pick = df["is_home_pick"].fillna(False).astype(bool)
            df["market_prob"] = no_vig[1].where(is_home_pick, no_vig[0]).fillna(df["market_prob"])
        elif "team" in df.columns and {"home_team", "away_team"}.issubset(df.columns):
            is_home_pick = df["team"].astype(str).str.lower() == df["home_team"].astype(str).str.lower()
            df["market_prob"] = no_vig[1].where(is_home_pick, no_vig[0]).fillna(df["market_prob"])
        else:
            df["market_prob"] = no_vig[0].fillna(df["market_prob"])

    # 3) ML probability: use model when available, fallback to market probability + noise
    model = load_model()
    model_loaded = model is not None
    df["model_probability"] = pd.to_numeric(df["market_prob"], errors="coerce")
    if model is not None:
        try:
            if isinstance(model, dict) and {"model", "feature_names"}.issubset(model):
                estimator = model["model"]
                feature_names = model["feature_names"]
            else:
                estimator = model
                feature_names = []

            if feature_names and all(f in df.columns for f in feature_names):
                X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                df["model_probability"] = estimator.predict_proba(X)[:, 1]
            elif hasattr(estimator, "predict_proba"):
                numeric_df = df.select_dtypes(include=["number"]).fillna(0.0)
                if not numeric_df.empty:
                    df["model_probability"] = estimator.predict_proba(numeric_df)[:, 1]
                else:
                    model_loaded = False
        except Exception as exc:
            model_loaded = False
            logger.warning("ML model unavailable for predict_proba; falling back to market_probability: %s", exc)

    if not model_loaded:
        market_prob = pd.to_numeric(df["market_prob"], errors="coerce").fillna(0.5)
        df["model_probability"] = (market_prob + np.random.normal(0, 0.015, size=len(df))).clip(0.01, 0.99)

    df["ml_prob"] = pd.to_numeric(df["model_probability"], errors="coerce").fillna(df["market_prob"])
    df["ai_prob"] = pd.to_numeric(df.get("ai_probability", pd.NA), errors="coerce")

    # 4) Weighted consensus probability
    df = normalize_probability_components(df)
    df["market_probability"] = df["market_prob"].clip(lower=0.0, upper=1.0)
    df["ml_probability"] = df["ml_prob"].clip(lower=0.0, upper=1.0)
    df["ai_probability"] = df["ai_prob"].clip(lower=0.0, upper=1.0)

    # 5) Probability calibration, EV calculation, and edge
    df["decimal_odds"] = pd.to_numeric(df["odds_american"], errors="coerce").apply(american_to_decimal)
    df = calibrate_probabilities(df)
    prob_for_ev = pd.to_numeric(df.get("calibrated_probability"), errors="coerce").fillna(df["model_probability"])
    df["expected_value"] = (
        prob_for_ev * (df["decimal_odds"] - 1)
        - (1 - prob_for_ev)
    )
    df["edge"] = prob_for_ev - df["market_probability"]
    df["debug_merge_keys"] = ", ".join([k for k in MERGE_KEYS if k in df.columns])
    df["debug_model_loaded"] = bool(model_loaded)

    if "team" not in df.columns:
        df["team"] = df.get("away_team", "")

    df = df.sort_values("edge", ascending=False).reset_index(drop=True)

    # 9) Debug output for verification in logs
    debug_cols = [
        "home_team",
        "away_team",
        "odds_american",
        "market_probability",
        "ml_probability",
        "calibrated_probability",
        "consensus_prob",
        "expected_value",
        "edge",
    ]
    available_debug_cols = [c for c in debug_cols if c in df.columns]
    if available_debug_cols:
        logger.info("Analysis probability debug sample:\n%s", df[available_debug_cols].head(25).to_string(index=False))

    return df


@st.cache_data(ttl=300)
def load_base_data() -> pd.DataFrame:
    df = pd.read_csv("data/master_all_sports.csv")
    return _normalize_key_columns(df)


@st.cache_data(ttl=180)
def run_analysis_pipeline(
    sports: Iterable[str],
    max_rows: int,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base_df = load_base_data().copy()
    if "league" in base_df.columns:
        base_df["league"] = base_df["league"].apply(_normalize_league_value)

    selected_sports = _normalize_sports_filter(sports)

    available_base_leagues = sorted(base_df["league"].dropna().astype(str).str.upper().unique().tolist()) if "league" in base_df.columns else []
    logger.info("Selected sports (normalized): %s", selected_sports)
    logger.info("Available base leagues (normalized): %s", available_base_leagues)

    has_theover_data = (spreads_df is not None and not spreads_df.empty) or (totals_df is not None and not totals_df.empty)

    if selected_sports and "league" in base_df.columns:
        filtered = base_df[base_df["league"].isin(selected_sports)].copy()
    else:
        filtered = base_df.copy()

    logger.info("Rows after sports filter: %s", len(filtered))
    if filtered.empty:
        if selected_sports and "league" in base_df.columns:
            logger.warning(
                "No base rows matched selected leagues. Possible league label mismatch. selected=%s available=%s",
                selected_sports,
                available_base_leagues,
            )
        else:
            logger.warning("No base rows available before merge stage.")

    filtered = filtered.head(max_rows)

    if use_ml and run_ml_predictions and not filtered.empty:
        ml_df = _normalize_key_columns(run_ml_predictions(filtered))
        filtered = _safe_merge(filtered, ml_df, "_ml")

    merged_theover = None
    if spreads_df is not None and not spreads_df.empty:
        logger.info("Uploaded spreads_df columns: %s", spreads_df.columns.tolist())
        merged_theover = _enrich_uploaded_league(_normalize_key_columns(spreads_df), selected_sports)
    if totals_df is not None and not totals_df.empty:
        logger.info("Uploaded totals_df columns: %s", totals_df.columns.tolist())
        totals_norm = _normalize_key_columns(totals_df)
        totals_norm = _enrich_uploaded_league(totals_norm, selected_sports)
        merged_theover = pd.concat([merged_theover, totals_norm], ignore_index=True) if merged_theover is not None else totals_norm

    if merged_theover is not None and not merged_theover.empty:
        logger.info("Merged TheOver columns: %s", merged_theover.columns.tolist())
        unique_leagues = sorted(merged_theover["league"].fillna("").astype(str).unique().tolist()) if "league" in merged_theover.columns else []
        logger.info("Unique merged TheOver league values: %s", unique_leagues)

    if filtered.empty and has_theover_data and merged_theover is not None and not merged_theover.empty:
        fallback_df = merged_theover.copy()
        logger.info("Fallback row count before sports filtering: %s", len(fallback_df))
        if selected_sports and "league" in fallback_df.columns:
            non_blank_leagues = fallback_df["league"].fillna("").astype(str).str.strip()
            if non_blank_leagues.ne("").any():
                fallback_df = fallback_df[fallback_df["league"].isin(selected_sports)].copy()
            else:
                logger.info(
                    "Skipping fallback sports filter: uploaded TheOver data had no usable league labels. selected=%s",
                    selected_sports,
                )
        logger.info("Fallback row count after sports filtering: %s", len(fallback_df))
        filtered = fallback_df.head(max_rows)
        logger.info(
            "Fallback activated: using uploaded TheOver data as base. fallback_rows=%s selected=%s",
            len(filtered),
            selected_sports,
        )

    filtered = _safe_merge(filtered, merged_theover, "_theover")

    analyzed = _apply_analysis_calculations(filtered)
    logger.info("Analyzed row count: %s", len(analyzed))
    if analyzed.empty:
        return analyzed

    return analyzed


def generate_parlays(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()
    parlays_df = generate_parlay_candidates(analysis_df)
    print("Total games:", len(analysis_df))
    print("Positive EV bets:", len(analysis_df[analysis_df["expected_value"] > 0]) if "expected_value" in analysis_df.columns else 0)
    print("Top edge:", analysis_df["edge"].max() if "edge" in analysis_df.columns else 0)
    print("Parlays generated:", len(parlays_df))
    return parlays_df


def build_realtime_edges(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()

    edge_cols = [
        c
        for c in [
            "league",
            "home_team",
            "away_team",
            "calibrated_probability",
            "market_probability",
            "decimal_odds",
            "expected_value",
            "edge",
        ]
        if c in analysis_df.columns
    ]
    if edge_cols:
        edges_df = analysis_df[edge_cols]
        if "edge" in edges_df.columns:
            return edges_df.sort_values("edge", ascending=False).head(25)
        if "expected_value" in edges_df.columns:
            return edges_df.sort_values("expected_value", ascending=False).head(25)
        return edges_df.head(25)

    return analysis_df.head(25)


def optimize_portfolio_allocation(analysis_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    edges = build_realtime_edges(analysis_df)
    if edges.empty:
        return edges

    portfolio = add_kelly_bet_sizing(edges, bankroll=bankroll, fraction=0.25)
    recommended_total = portfolio["recommended_bet"].sum() if "recommended_bet" in portfolio.columns else 0.0
    if recommended_total > 0:
        portfolio["allocation_pct"] = ((portfolio["recommended_bet"] / recommended_total) * 100).round(2)
    else:
        portfolio["allocation_pct"] = 0.0
    return portfolio.sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=1000, simulations=1000)
