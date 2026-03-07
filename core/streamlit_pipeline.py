from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _StreamlitShim:
        @staticmethod
        def cache_data(*_args: Any, **_kwargs: Any):
            def _decorator(func):
                return func

            return _decorator

    st = _StreamlitShim()

from core.bankroll_simulator import simulate_bankroll
from core.kelly_optimizer import add_kelly_bet_sizing
from core.probability_engine import american_to_prob
from core.schema.base_schema import ensure_base_schema
from core.team_mapper import normalize_team_name

logger = logging.getLogger(__name__)

VALID_MARKETS = {"spread_home", "spread_away", "total_over", "total_under"}
DATE_ALIASES = ["game_date", "commence_time", "start_time", "time", "date", "event_date"]
LEAGUE_ALIASES = {"NCAAM": "NCAAB", "NCAA MEN'S BASKETBALL": "NCAAB", "NCAA MENS BASKETBALL": "NCAAB"}

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
    "kalshi_probability",
    "kalshi_match_status",
    "kalshi_match_reason",
]

CANONICAL_BET_COLUMNS = [
    "league",
    "home_team",
    "away_team",
    "game_date",
    "game_key",
    "market_type",
    "spread_line",
    "total_line",
    "theover_probability",
    "odds_american",
    "market_probability",
    "ml_probability",
    "calibrated_probability",
    "expected_value",
    "edge",
    "best_pick",
]


def _string_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="string")
    if col in df.columns:
        return df[col].fillna(default).astype("string")
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def _numeric_series(df: pd.DataFrame, col: str, default: float | int | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if col in df.columns:
        out = pd.to_numeric(df[col], errors="coerce")
    else:
        out = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    if default is not None:
        out = out.fillna(default)
    return out


def _game_dates(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    out = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")
    for col in DATE_ALIASES:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            out = out.where(out.notna(), parsed)
    return out


def _normalize_upload(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out["league"] = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
    out["home_team"] = _string_series(out, "home_team").map(normalize_team_name)
    out["away_team"] = _string_series(out, "away_team").map(normalize_team_name)
    out["game_date"] = _game_dates(out)
    return out


def _mk_game_key(df: pd.DataFrame) -> pd.Series:
    return (
        _string_series(df, "league").str.upper()
        + "|"
        + _string_series(df, "home_team").str.upper()
        + "|"
        + _string_series(df, "away_team").str.upper()
    )


@st.cache_data(ttl=300)
def load_base_data() -> pd.DataFrame:
    try:
        base_df = pd.read_csv("data/master_all_sports.csv")
    except Exception:
        return pd.DataFrame()
    base_df = ensure_base_schema(base_df)
    base_df.columns = [str(c).strip().lower() for c in base_df.columns]
    base_df["league"] = _string_series(base_df, "league").str.upper().replace(LEAGUE_ALIASES)
    base_df["home_team"] = _string_series(base_df, "home_team").map(normalize_team_name)
    base_df["away_team"] = _string_series(base_df, "away_team").map(normalize_team_name)
    base_df["game_date"] = _game_dates(base_df)
    base_df["game_key"] = _mk_game_key(base_df)
    return base_df


def _first_existing_numeric(df: pd.DataFrame, candidates: list[str], default: float | int | None = None) -> pd.Series:
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    for col in candidates:
        if col in df.columns:
            out = out.where(out.notna(), pd.to_numeric(df[col], errors="coerce"))
    if default is not None:
        out = out.fillna(default)
    return out


def american_to_decimal(odds: Any) -> float:
    v = pd.to_numeric(odds, errors="coerce")
    if pd.isna(v):
        return 1.9091
    v = float(v)
    if v > 0:
        return 1 + (v / 100.0)
    if v < 0:
        return 1 + (100.0 / abs(v))
    return 1.0


def _format_best_pick(row: pd.Series) -> str:
    market = str(row.get("market_type") or "")
    if market == "spread_home":
        line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        return f"{row.get('home_team', '')} {line:+.1f}" if pd.notna(line) else str(row.get("home_team") or "")
    if market == "spread_away":
        line = pd.to_numeric(row.get("spread_line"), errors="coerce")
        return f"{row.get('away_team', '')} {line:+.1f}" if pd.notna(line) else str(row.get("away_team") or "")
    if market == "total_over":
        line = pd.to_numeric(row.get("total_line"), errors="coerce")
        return f"Over {line:.1f}" if pd.notna(line) else "Over"
    if market == "total_under":
        line = pd.to_numeric(row.get("total_line"), errors="coerce")
        return f"Under {line:.1f}" if pd.notna(line) else "Under"
    return ""


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["odds_american"] = _numeric_series(out, "odds_american", -110.0)
    out["market_probability"] = out["odds_american"].apply(american_to_prob)

    theover = _numeric_series(out, "theover_probability")
    theover = theover.where(theover <= 1, theover / 100.0)
    ml = _numeric_series(out, "ml_probability")
    calibrated = theover.where(theover.notna(), ml)
    calibrated = calibrated.where(calibrated.notna(), out["market_probability"])  # fallback chain
    calibrated = calibrated.clip(0.01, 0.99)

    out["theover_probability"] = theover
    out["ml_probability"] = ml
    out["calibrated_probability"] = calibrated
    out["decimal_odds"] = out["odds_american"].apply(american_to_decimal)
    out["expected_value"] = calibrated * (out["decimal_odds"] - 1) - (1 - calibrated)
    out["edge"] = calibrated - out["market_probability"]
    out["best_pick"] = out.apply(_format_best_pick, axis=1)
    return out


def build_theover_bet_rows(
    spreads_df: pd.DataFrame | None,
    totals_df: pd.DataFrame | None,
    selected_sports: list[str] | None,
) -> pd.DataFrame:
    spreads = _normalize_upload(spreads_df)
    totals = _normalize_upload(totals_df)
    pieces: list[pd.DataFrame] = []

    if not spreads.empty:
        spread_line = _first_existing_numeric(spreads, ["spread_line", "spread", "line", "points"])
        spread_prob = _first_existing_numeric(spreads, ["theover_probability", "winprobability", "win_probability", "probability"])
        spread_odds = _first_existing_numeric(spreads, ["odds_american", "american_odds", "odds"], default=-110.0)

        base = spreads[["league", "home_team", "away_team", "game_date"]].copy()
        spread_home = base.copy()
        spread_home["market_type"] = "spread_home"
        spread_home["spread_line"] = spread_line
        spread_home["total_line"] = pd.NA
        spread_home["theover_probability"] = spread_prob
        spread_home["odds_american"] = spread_odds

        spread_away = base.copy()
        spread_away["market_type"] = "spread_away"
        spread_away["spread_line"] = -spread_line
        spread_away["total_line"] = pd.NA
        spread_away["theover_probability"] = (1 - spread_prob).where(spread_prob.notna(), pd.NA)
        spread_away["odds_american"] = spread_odds
        pieces.extend([spread_home, spread_away])

    if not totals.empty:
        total_line = _first_existing_numeric(totals, ["total_line", "total", "line", "points"])
        total_prob = _first_existing_numeric(totals, ["theover_probability", "winprobability", "win_probability", "probability"])
        total_odds = _first_existing_numeric(totals, ["odds_american", "american_odds", "odds"], default=-110.0)

        base = totals[["league", "home_team", "away_team", "game_date"]].copy()
        total_over = base.copy()
        total_over["market_type"] = "total_over"
        total_over["spread_line"] = pd.NA
        total_over["total_line"] = total_line
        total_over["theover_probability"] = total_prob
        total_over["odds_american"] = total_odds

        total_under = base.copy()
        total_under["market_type"] = "total_under"
        total_under["spread_line"] = pd.NA
        total_under["total_line"] = total_line
        total_under["theover_probability"] = (1 - total_prob).where(total_prob.notna(), pd.NA)
        total_under["odds_american"] = total_odds
        pieces.extend([total_over, total_under])

    valid_pieces: list[pd.DataFrame] = []
    for p in pieces:
        if p is None:
            continue
        if not isinstance(p, pd.DataFrame):
            continue
        if p.empty:
            continue
        if p.dropna(how="all").empty:
            continue
        cleaned_piece = p.dropna(axis=1, how="all")
        if cleaned_piece.empty:
            continue
        valid_pieces.append(cleaned_piece)

    out = pd.concat(valid_pieces, ignore_index=True) if valid_pieces else pd.DataFrame(columns=CANONICAL_BET_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_BET_COLUMNS)
    out["league"] = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
    out["home_team"] = _string_series(out, "home_team").map(normalize_team_name)
    out["away_team"] = _string_series(out, "away_team").map(normalize_team_name)
    out["market_type"] = _string_series(out, "market_type")
    out["game_date"] = _game_dates(out)
    out["spread_line"] = pd.to_numeric(out.get("spread_line"), errors="coerce")
    out["total_line"] = pd.to_numeric(out.get("total_line"), errors="coerce")
    out["theover_probability"] = pd.to_numeric(out.get("theover_probability"), errors="coerce")
    out["odds_american"] = pd.to_numeric(out.get("odds_american"), errors="coerce")
    if selected_sports:
        selected = {str(s).upper() for s in selected_sports}
        out = out[_string_series(out, "league").isin(selected)].copy()
    out["game_key"] = _mk_game_key(out)
    out = _apply_analysis_calculations(out)
    for col in CANONICAL_BET_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CANONICAL_BET_COLUMNS]


def _fill_missing_game_dates_from_base(bet_rows_df: pd.DataFrame, base_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = bet_rows_df.copy()
    out["game_date"] = _game_dates(out)
    missing_before = out["game_date"].isna()

    if base_df is not None and not base_df.empty and missing_before.any():
        base = base_df.copy()
        base["league"] = _string_series(base, "league").str.upper().replace(LEAGUE_ALIASES)
        base["home_team"] = _string_series(base, "home_team").map(normalize_team_name)
        base["away_team"] = _string_series(base, "away_team").map(normalize_team_name)
        base["game_date"] = _game_dates(base)
        base = base[base["game_date"].notna()].copy()

        schedule = (
            base.sort_values("game_date")
            .drop_duplicates(["league", "home_team", "away_team"], keep="last")
            [["league", "home_team", "away_team", "game_date"]]
        )
        out = out.merge(
            schedule,
            on=["league", "home_team", "away_team"],
            how="left",
            suffixes=("", "_base"),
        )
        out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_base"])
        out = out.drop(columns=["game_date_base"])

    filled = int((missing_before & out["game_date"].notna()).sum())
    missing_after = int(out["game_date"].isna().sum())
    return out, {
        "date_fill_total_rows": int(missing_before.sum()),
        "date_fill_success_rows": filled,
        "date_fill_success_rate": float(filled / max(int(missing_before.sum()), 1)),
        "missing_game_date_rows": missing_after,
    }


def is_stale_schedule(base_df: pd.DataFrame, bet_rows_df: pd.DataFrame) -> bool:
    if base_df is None or base_df.empty or bet_rows_df is None or bet_rows_df.empty:
        return False
    base_dates = pd.to_datetime(base_df.get("game_date"), errors="coerce", utc=True)
    bet_dates = pd.to_datetime(bet_rows_df.get("game_date"), errors="coerce", utc=True)
    if base_dates.notna().sum() == 0 or bet_dates.notna().sum() == 0:
        return False
    return bool(base_dates.max() < bet_dates.max())


def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)
    if "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")

    df = analysis_df[_string_series(analysis_df, "market_type").isin(VALID_MARKETS)].copy()
    if df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)

    df["expected_value"] = _numeric_series(df, "expected_value", 0.0)
    df["edge"] = _numeric_series(df, "edge", 0.0)
    df["best_pick"] = df.apply(_format_best_pick, axis=1)
    df = df.sort_values(["expected_value", "edge"], ascending=[False, False])

    pick_df = (
        df.groupby(["league", "home_team", "away_team", "game_date"], dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    for col in BEST_PICK_COLUMNS:
        if col not in pick_df.columns:
            pick_df[col] = pd.NA
    return pick_df[BEST_PICK_COLUMNS]


def run_analysis_pipeline(
    sports: list[str] | None = None,
    max_rows: int = 1000,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_df = load_base_data()
    odds_schedule_loaded = not base_df.empty

    bet_rows = build_theover_bet_rows(spreads_df, totals_df, sports)
    bet_rows, date_stats = _fill_missing_game_dates_from_base(bet_rows, base_df)

    merge_keys = ["league", "home_team", "away_team"]
    merged = bet_rows.copy()
    if not base_df.empty:
        base_merge_cols = merge_keys + [c for c in ["game_date", "odds_american", "ml_probability"] if c in base_df.columns]
        merged = merged.merge(base_df[base_merge_cols].drop_duplicates(merge_keys), on=merge_keys, how="left", suffixes=("", "_base"))
        if "game_date_base" in merged.columns:
            merged["game_date"] = pd.to_datetime(merged["game_date"], errors="coerce", utc=True)
            merged["game_date_base"] = pd.to_datetime(merged["game_date_base"], errors="coerce", utc=True)
            merged["game_date"] = merged["game_date"].where(merged["game_date"].notna(), merged["game_date_base"])
            merged = merged.drop(columns=["game_date_base"])
        if "odds_american_base" in merged.columns:
            merged["odds_american"] = _numeric_series(merged, "odds_american").where(
                _numeric_series(merged, "odds_american").notna(),
                _numeric_series(merged, "odds_american_base"),
            ).fillna(-110.0)
            merged = merged.drop(columns=["odds_american_base"])
        if "ml_probability_base" in merged.columns:
            merged["ml_probability"] = _numeric_series(merged, "ml_probability").where(
                _numeric_series(merged, "ml_probability").notna(),
                _numeric_series(merged, "ml_probability_base"),
            )
            merged = merged.drop(columns=["ml_probability_base"])

    merged["game_date"] = _game_dates(merged)

    analysis_df = _apply_analysis_calculations(merged).head(max_rows).copy()
    if not analysis_df.empty and "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")

    best_picks_df = build_best_picks_df(analysis_df)

    if not analysis_df.empty and _string_series(analysis_df, "market_type").isin(VALID_MARKETS).any() and best_picks_df.empty:
        logger.warning("best_picks_df empty while analysis_df has spread/total rows")

    stale = is_stale_schedule(base_df, analysis_df)
    base_coverage = float(pd.to_datetime(base_df.get("game_date"), errors="coerce", utc=True).notna().mean()) if not base_df.empty else 0.0

    diagnostics = {
        "total_rows": int(len(analysis_df)),
        "rows_with_game_date": int(pd.to_datetime(analysis_df.get("game_date"), errors="coerce", utc=True).notna().sum()) if not analysis_df.empty else 0,
        "total_games": int(analysis_df[["league", "home_team", "away_team"]].drop_duplicates().shape[0]) if not analysis_df.empty else 0,
        "bet_rows": int(len(analysis_df)),
        "best_picks": int(len(best_picks_df)),
        "kalshi_attempted": 0,
        "kalshi_matches": 0,
        "kalshi_match_rate": 0.0,
        "match_rate": 0.0,
        "theover_totals_games": int(analysis_df[_string_series(analysis_df, "market_type").str.startswith("total")]["game_key"].nunique()) if not analysis_df.empty else 0,
        "theover_spreads_games": int(analysis_df[_string_series(analysis_df, "market_type").str.startswith("spread")]["game_key"].nunique()) if not analysis_df.empty else 0,
        "date_fill_total_rows": int(date_stats["date_fill_total_rows"]),
        "date_fill_success_rows": int(date_stats["date_fill_success_rows"]),
        "date_fill_success_rate": float(date_stats["date_fill_success_rate"]),
        "missing_game_date_rows": int(date_stats["missing_game_date_rows"]),
        "positive_ev_picks": int((_numeric_series(best_picks_df, "expected_value", 0.0) > 0).sum()) if not best_picks_df.empty else 0,
        "market_type_counts": _string_series(analysis_df, "market_type").value_counts(dropna=False).to_dict() if not analysis_df.empty else {},
        "allowed_market_type_rows": int(_string_series(analysis_df, "market_type").isin(VALID_MARKETS).sum()) if not analysis_df.empty else 0,
        "positive_ev_rows": int((_numeric_series(analysis_df, "expected_value", 0.0) > 0).sum()) if not analysis_df.empty else 0,
        "best_pick_nonempty_rows": int(_string_series(best_picks_df, "best_pick").str.strip().str.len().gt(0).sum()) if not best_picks_df.empty else 0,
        "best_picks_count": int(len(best_picks_df)),
        "odds_schedule_loaded": odds_schedule_loaded,
        "base_rows_loaded": int(len(base_df)),
        "merge_keys_used": merge_keys,
        "stale_base_schedule": stale,
        "base_date_coverage": base_coverage,
        "has_normalized_bet_rows": not analysis_df.empty,
    }

    default_odds_ratio = float((_numeric_series(analysis_df, "odds_american") == -110).mean()) if not analysis_df.empty else 1.0
    diagnostics["odds_fallback_only"] = bool(default_odds_ratio >= 0.99)
    if diagnostics["odds_fallback_only"] and not analysis_df.empty:
        diagnostics["diagnostic_warning"] = "odds_american mostly fallback -110"

    result = (analysis_df, best_picks_df, diagnostics)
    return result


def generate_parlays(best_picks_df: pd.DataFrame, max_legs: int = 5) -> pd.DataFrame:
    cols = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame(columns=cols)
    df = best_picks_df.copy()
    df = df[_string_series(df, "best_pick").str.strip().str.len() > 0].copy()
    if len(df) < 2:
        return pd.DataFrame(columns=cols)

    df["calibrated_probability"] = _numeric_series(df, "calibrated_probability", 0.5).clip(0.01, 0.99)
    df["decimal_odds"] = _numeric_series(df, "decimal_odds").fillna(_numeric_series(df, "odds_american", -110.0).apply(american_to_decimal))

    records: list[dict[str, Any]] = []
    for leg_count in range(2, min(max_legs, len(df)) + 1):
        for combo in combinations(df.index.tolist(), leg_count):
            legs_df = df.loc[list(combo)]
            prob = float(legs_df["calibrated_probability"].prod())
            odds = float(legs_df["decimal_odds"].prod())
            ev = prob * (odds - 1) - (1 - prob)
            legs_str = " | ".join(legs_df["best_pick"].astype(str).tolist())
            records.append(
                {
                    "parlay_legs": legs_str,
                    "combined_probability": prob,
                    "combined_decimal_odds": odds,
                    "parlay_ev": ev,
                    "legs": leg_count,
                }
            )

    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records)[cols].sort_values("parlay_ev", ascending=False).reset_index(drop=True)


def optimize_portfolio_allocation(best_picks_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    portfolio = best_picks_df.copy()
    portfolio = portfolio[_string_series(portfolio, "best_pick").str.strip().str.len() > 0].copy()
    if portfolio.empty:
        return pd.DataFrame()

    portfolio["decimal_odds"] = _numeric_series(portfolio, "decimal_odds").fillna(_numeric_series(portfolio, "odds_american", -110.0).apply(american_to_decimal))
    portfolio = add_kelly_bet_sizing(portfolio, bankroll=bankroll, fraction=0.25)
    if "recommended_bet" not in portfolio.columns:
        portfolio["recommended_bet"] = 0.0

    cols = [
        "league",
        "home_team",
        "away_team",
        "best_pick",
        "calibrated_probability",
        "expected_value",
        "edge",
        "decimal_odds",
        "recommended_bet",
    ]
    for col in cols:
        if col not in portfolio.columns:
            portfolio[col] = pd.NA
    return portfolio[cols].sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=1000, simulations=1000)
