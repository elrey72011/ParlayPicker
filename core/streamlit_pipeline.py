from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import pandas as pd
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

CANONICAL_BET_COLUMNS = [
    "league",
    "home_team",
    "away_team",
    "game_date",
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
    "game_key",
]

LEAGUE_ALIASES = {"NCAAM": "NCAAB", "NCAA MEN'S BASKETBALL": "NCAAB", "NCAA MENS BASKETBALL": "NCAAB"}
VALID_MARKETS = {"spread_home", "spread_away", "total_over", "total_under"}
DATE_ALIASES = ["game_date", "commence_time", "start_time", "time", "date", "event_date"]
ODDS_ALIASES = ["odds_american", "american_odds", "odds", "line_odds"]
PROB_ALIASES = ["theover_probability", "winprobability", "win_probability", "probability", "model_probability"]


def _numeric_series(df: pd.DataFrame, col: str, default: float | int | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    if default is not None:
        s = s.fillna(default)
    return s


def _string_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="string")
    if col in df.columns:
        return df[col].fillna(default).astype("string")
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def _to_game_date(df: pd.DataFrame) -> pd.Series:
    for c in DATE_ALIASES:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().any():
                return dt
    return pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")


def _first_numeric(df: pd.DataFrame, aliases: list[str], default: float | int | None = None) -> pd.Series:
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    for col in aliases:
        if col in df.columns:
            out = out.where(out.notna(), pd.to_numeric(df[col], errors="coerce"))
    if default is not None:
        out = out.fillna(default)
    return out


def _normalize_upload(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for col in ["league", "home_team", "away_team", "pick", "selection", "pick_team", "pickteam"]:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    out["league"] = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
    out["home_team"] = _string_series(out, "home_team").map(normalize_team_name)
    out["away_team"] = _string_series(out, "away_team").map(normalize_team_name)
    out["game_date"] = _to_game_date(out)
    return out


def _mk_game_key(df: pd.DataFrame) -> pd.Series:
    return (
        _string_series(df, "league").str.upper()
        + "|"
        + _string_series(df, "home_team").str.upper()
        + "|"
        + _string_series(df, "away_team").str.upper()
    )


def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["league", "home_team", "away_team", "game_date", "game_key"])
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out["league"] = _string_series(out, "league").str.upper().replace(LEAGUE_ALIASES)
    out["home_team"] = _string_series(out, "home_team").map(normalize_team_name)
    out["away_team"] = _string_series(out, "away_team").map(normalize_team_name)
    out["game_date"] = _to_game_date(out)
    out["game_key"] = _mk_game_key(out)
    return out


@st.cache_data(ttl=300)
def load_base_data() -> pd.DataFrame:
    try:
        base_df = pd.read_csv("data/master_all_sports.csv")
    except Exception:
        return pd.DataFrame()
    base_df = ensure_base_schema(base_df)
    base_df = _normalize_key_columns(base_df)
    base_df.attrs["loaded_from"] = "data/master_all_sports.csv"
    return base_df


def normalize_theover_df(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_upload(df)


def _spread_rows(spreads_df: pd.DataFrame) -> pd.DataFrame:
    if spreads_df.empty:
        return pd.DataFrame()
    line = _first_numeric(spreads_df, ["spread_line", "spread", "line", "points"])
    prob = _first_numeric(spreads_df, PROB_ALIASES)
    odds = _first_numeric(spreads_df, ODDS_ALIASES, default=-110.0)

    base = pd.DataFrame(
        {
            "league": _string_series(spreads_df, "league").str.upper(),
            "home_team": _string_series(spreads_df, "home_team").map(normalize_team_name),
            "away_team": _string_series(spreads_df, "away_team").map(normalize_team_name),
            "game_date": _to_game_date(spreads_df),
        }
    )
    home = base.copy()
    away = base.copy()
    home["market_type"] = "spread_home"
    away["market_type"] = "spread_away"
    home["spread_line"] = line
    away["spread_line"] = -line
    home["total_line"] = pd.NA
    away["total_line"] = pd.NA
    home["theover_probability"] = prob
    away["theover_probability"] = (1 - prob).where(prob.notna(), pd.NA)
    home["odds_american"] = odds
    away["odds_american"] = odds
    return pd.concat([home, away], ignore_index=True)


def _total_rows(totals_df: pd.DataFrame) -> pd.DataFrame:
    if totals_df.empty:
        return pd.DataFrame()
    line = _first_numeric(totals_df, ["total_line", "total", "line", "points"])
    prob = _first_numeric(totals_df, PROB_ALIASES)
    odds = _first_numeric(totals_df, ODDS_ALIASES, default=-110.0)

    base = pd.DataFrame(
        {
            "league": _string_series(totals_df, "league").str.upper(),
            "home_team": _string_series(totals_df, "home_team").map(normalize_team_name),
            "away_team": _string_series(totals_df, "away_team").map(normalize_team_name),
            "game_date": _to_game_date(totals_df),
        }
    )
    over = base.copy()
    under = base.copy()
    over["market_type"] = "total_over"
    under["market_type"] = "total_under"
    over["spread_line"] = pd.NA
    under["spread_line"] = pd.NA
    over["total_line"] = line
    under["total_line"] = line
    over["theover_probability"] = prob
    under["theover_probability"] = (1 - prob).where(prob.notna(), pd.NA)
    over["odds_american"] = odds
    under["odds_american"] = odds
    return pd.concat([over, under], ignore_index=True)


def build_theover_bet_rows(spreads_df: pd.DataFrame | None, totals_df: pd.DataFrame | None, selected_sports: list[str] | None = None) -> pd.DataFrame:
    spreads = _normalize_upload(spreads_df) if isinstance(spreads_df, pd.DataFrame) else pd.DataFrame()
    totals = _normalize_upload(totals_df) if isinstance(totals_df, pd.DataFrame) else pd.DataFrame()
    rows = pd.concat([_spread_rows(spreads), _total_rows(totals)], ignore_index=True)
    if rows.empty:
        return pd.DataFrame(columns=CANONICAL_BET_COLUMNS)

    if selected_sports:
        sports = {str(s).upper() for s in selected_sports}
        rows = rows[_string_series(rows, "league").isin(sports)].copy()

    rows["league"] = _string_series(rows, "league").str.upper().replace(LEAGUE_ALIASES)
    rows["game_key"] = _mk_game_key(rows)
    rows = _apply_analysis_calculations(rows)
    for col in CANONICAL_BET_COLUMNS:
        if col not in rows.columns:
            rows[col] = pd.NA
    return rows[CANONICAL_BET_COLUMNS]


def _fill_missing_game_dates_from_base(bet_rows_df: pd.DataFrame, base_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = bet_rows_df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce", utc=True)
    out["game_key"] = _mk_game_key(out)
    missing = out["game_date"].isna()
    fill_total = int(missing.sum())

    base = base_df.copy()
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce", utc=True)
    base = base[base["game_date"].notna()].copy()
    base["game_key"] = _mk_game_key(base)
    by_key = base.sort_values("game_date").drop_duplicates("game_key", keep="last")[["game_key", "game_date"]]
    out = out.merge(by_key, on="game_key", how="left", suffixes=("", "_base"))
    out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_base"])
    out = out.drop(columns=["game_date_base"])

    still = out["game_date"].isna()
    if still.any() and not base.empty:
        fallback = base.sort_values("game_date").drop_duplicates(["league", "home_team", "away_team"], keep="last")
        out = out.merge(
            fallback[["league", "home_team", "away_team", "game_date"]],
            on=["league", "home_team", "away_team"],
            how="left",
            suffixes=("", "_fallback"),
        )
        out["game_date"] = out["game_date"].where(out["game_date"].notna(), out["game_date_fallback"])
        out = out.drop(columns=["game_date_fallback"])

    filled = int((missing & out["game_date"].notna()).sum())
    out["date_missing_after_fill"] = out["game_date"].isna()
    return out, {
        "date_fill_total_rows": fill_total,
        "date_fill_success_rows": filled,
        "date_fill_success_rate": float(filled / max(fill_total, 1)),
    }


def is_stale_schedule(base_df: pd.DataFrame, bet_rows_df: pd.DataFrame) -> bool:
    if base_df is None or base_df.empty or bet_rows_df is None or bet_rows_df.empty:
        return False
    base_dates = pd.to_datetime(base_df.get("game_date"), errors="coerce", utc=True)
    bet_dates = pd.to_datetime(bet_rows_df.get("game_date"), errors="coerce", utc=True)
    if base_dates.notna().sum() == 0 or bet_dates.notna().sum() == 0:
        return False
    return bool(base_dates.max() < bet_dates.max())


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
    mt = str(row.get("market_type") or "")
    if mt == "spread_home":
        val = pd.to_numeric(row.get("spread_line", row.get("spread")), errors="coerce")
        return f"{row.get('home_team','')} {val:+.1f}" if pd.notna(val) else str(row.get("home_team") or "")
    if mt == "spread_away":
        val = pd.to_numeric(row.get("spread_line", row.get("spread")), errors="coerce")
        return f"{row.get('away_team','')} {val:+.1f}" if pd.notna(val) else str(row.get("away_team") or "")
    if mt == "total_over":
        val = pd.to_numeric(row.get("total_line", row.get("total")), errors="coerce")
        return f"Over {val:.1f}" if pd.notna(val) else "Over"
    if mt == "total_under":
        val = pd.to_numeric(row.get("total_line", row.get("total")), errors="coerce")
        return f"Under {val:.1f}" if pd.notna(val) else "Under"
    return ""


def _apply_analysis_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()

    odds = _numeric_series(out, "odds_american", -110.0)
    out["odds_american"] = odds
    out["decimal_odds"] = odds.apply(american_to_decimal)
    out["market_probability"] = odds.apply(american_to_prob)

    theover = _numeric_series(out, "theover_probability")
    theover = theover.where(theover <= 1, theover / 100.0)
    ml = _numeric_series(out, "ml_probability")
    ai = _numeric_series(out, "ai_probability")
    model = _numeric_series(out, "model_probability")
    if ml.isna().all():
        ml = model.where(model.notna(), ai)
    out["ml_probability"] = ml

    calibrated = theover.where(theover.notna(), ml)
    calibrated = calibrated.where(calibrated.notna(), out["market_probability"])
    calibrated = calibrated.clip(0.01, 0.99)
    out["theover_probability"] = theover
    out["calibrated_probability"] = calibrated

    out["expected_value"] = calibrated * (out["decimal_odds"] - 1) - (1 - calibrated)
    out["edge"] = calibrated - out["market_probability"]
    kalshi = _numeric_series(out, "kalshi_probability")
    out["consensus_prob"] = pd.concat([theover, ml, ai, kalshi], axis=1).mean(axis=1, skipna=True)

    if "market_type" in out.columns:
        out["best_pick"] = out.apply(_format_best_pick, axis=1)
    else:
        out["best_pick"] = _string_series(out, "away_team") + " spread"

    return out


def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)
    if "market_type" not in analysis_df.columns:
        raise ValueError("analysis_df missing market_type before best-pick construction")
    df = analysis_df.copy()
    df = df[_string_series(df, "market_type").isin(VALID_MARKETS)].copy()
    if df.empty:
        return pd.DataFrame(columns=BEST_PICK_COLUMNS)

    df["best_pick"] = df.apply(_format_best_pick, axis=1)
    df["expected_value"] = _numeric_series(df, "expected_value", 0.0)
    df["edge"] = _numeric_series(df, "edge", 0.0)
    df = df.sort_values(["expected_value", "edge"], ascending=[False, False])

    group_cols = [c for c in ["league", "home_team", "away_team", "game_date"] if c in df.columns]
    picked = df.groupby(group_cols, as_index=False, sort=False).head(1).copy()

    for c in BEST_PICK_COLUMNS:
        if c not in picked.columns:
            picked[c] = pd.NA
    return picked[BEST_PICK_COLUMNS].reset_index(drop=True)


def generate_parlays_table(analysis_df: pd.DataFrame, min_ev: float = 0.02) -> pd.DataFrame:
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=["away_team", "expected_value"])
    out = analysis_df.copy()
    out["expected_value"] = _numeric_series(out, "expected_value", 0.0)
    out = out[out["expected_value"] >= min_ev].sort_values("expected_value", ascending=False)
    return out


def run_analysis_pipeline(
    sports: list[str] | None = None,
    max_rows: int = 1000,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # a) load base odds/schedule
    base_df = load_base_data()
    odds_schedule_loaded = bool(base_df.attrs.get("loaded_from")) and not base_df.empty

    # b/c) normalize TheOver uploads and build canonical spread/total bet rows
    bet_rows_df = build_theover_bet_rows(spreads_df, totals_df, sports)

    # d) fill game_date deterministically
    bet_rows_df, date_stats = _fill_missing_game_dates_from_base(bet_rows_df, base_df)

    # e) merge odds/base fields
    merge_keys = ["league", "home_team", "away_team"]
    merge_cols = merge_keys + [c for c in ["game_date", "odds_american", "market_probability", "decimal_odds", "ml_probability"] if c in base_df.columns]
    enriched = bet_rows_df.merge(base_df[merge_cols].drop_duplicates(merge_keys), on=merge_keys, how="left", suffixes=("", "_base"))

    for col in ["odds_american", "market_probability", "decimal_odds", "ml_probability"]:
        base_col = f"{col}_base"
        if base_col in enriched.columns:
            if col == "odds_american":
                left = _numeric_series(enriched, col)
                right = _numeric_series(enriched, base_col)
                enriched[col] = left.where(left.notna(), right).fillna(-110.0)
            else:
                left = _numeric_series(enriched, col)
                right = _numeric_series(enriched, base_col)
                enriched[col] = left.where(left.notna(), right)
            enriched = enriched.drop(columns=[base_col])

    # f) compute probabilities / EV / edge
    analyzed = _apply_analysis_calculations(enriched)
    analyzed = analyzed.head(max_rows).copy()

    # g) build best_picks_df from spread/total rows only
    best_picks_df = build_best_picks_df(analyzed)

    market_type_counts = (
        _string_series(analyzed, "market_type").value_counts(dropna=False).to_dict()
        if not analyzed.empty
        else {}
    )
    allowed_market_rows = int(_string_series(analyzed, "market_type").isin(VALID_MARKETS).sum()) if not analyzed.empty else 0
    positive_ev_rows = int((_numeric_series(analyzed, "expected_value", 0.0) > 0).sum()) if not analyzed.empty else 0
    best_pick_nonempty_rows = int(_string_series(best_picks_df, "best_pick").str.strip().str.len().gt(0).sum()) if not best_picks_df.empty else 0

    total_games = int(analyzed[["league", "home_team", "away_team"]].drop_duplicates().shape[0]) if not analyzed.empty else 0
    spread_games = int(analyzed[_string_series(analyzed, "market_type").str.startswith("spread")]["game_key"].nunique()) if not analyzed.empty else 0
    totals_games = int(analyzed[_string_series(analyzed, "market_type").str.startswith("total")]["game_key"].nunique()) if not analyzed.empty else 0
    kalshi_matches = 0
    positive_ev = int((_numeric_series(best_picks_df, "expected_value", 0.0) > 0).sum()) if not best_picks_df.empty else 0

    stale_status = is_stale_schedule(base_df, analyzed)

    diagnostics = {
        "total_games": total_games,
        "bet_rows": int(len(analyzed)),
        "best_picks": int(len(best_picks_df)),
        "kalshi_attempted": 0,
        "kalshi_matches": kalshi_matches,
        "kalshi_match_rate": float(kalshi_matches / max(len(best_picks_df), 1)),
        "match_rate": float(kalshi_matches / max(len(best_picks_df), 1)),
        "theover_totals_games": totals_games,
        "theover_spreads_games": spread_games,
        "date_fill_total_rows": int(date_stats["date_fill_total_rows"]),
        "date_fill_success_rows": int(date_stats["date_fill_success_rows"]),
        "date_fill_success_rate": float(date_stats["date_fill_success_rate"]),
        "positive_ev_picks": positive_ev,
        "market_type_counts": market_type_counts,
        "allowed_market_type_rows": allowed_market_rows,
        "positive_ev_rows": positive_ev_rows,
        "best_pick_nonempty_rows": best_pick_nonempty_rows,
        "merge_keys_used": merge_keys,
        "base_stale": stale_status,
        "odds_schedule_loaded": odds_schedule_loaded,
        "base_rows_loaded": int(len(base_df)),
        "has_normalized_bet_rows": not analyzed.empty,
    }
    # h) return analysis_df, best_picks_df, diagnostics
    return analyzed, best_picks_df, diagnostics


def generate_parlays(analysis_df: pd.DataFrame, max_legs: int = 5) -> pd.DataFrame:
    cols = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=cols)
    df = analysis_df.copy()
    df = df[_string_series(df, "best_pick").str.strip().str.len() > 0].copy()
    if len(df) < 2:
        return pd.DataFrame(columns=cols)
    df["calibrated_probability"] = _numeric_series(df, "calibrated_probability", 0.5).clip(0.01, 0.99)
    df["decimal_odds"] = _numeric_series(df, "decimal_odds").fillna(_numeric_series(df, "odds_american", -110.0).apply(american_to_decimal))
    records = []
    for leg_count in range(2, min(max_legs, len(df)) + 1):
        for combo in combinations(df.index.tolist(), leg_count):
            legs = df.loc[list(combo)]
            p = float(legs["calibrated_probability"].prod())
            o = float(legs["decimal_odds"].prod())
            ev = p * (o - 1) - (1 - p)
            leg_str = " | ".join([f"{r.home_team} vs {r.away_team}: {r.best_pick}" for r in legs.itertuples()])
            records.append({"parlay_legs": leg_str, "combined_probability": p, "combined_decimal_odds": o, "parlay_ev": ev, "legs": leg_count})
    return pd.DataFrame(records)[cols].sort_values("parlay_ev", ascending=False).reset_index(drop=True) if records else pd.DataFrame(columns=cols)


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
    need_cols = ["league", "home_team", "away_team", "best_pick", "calibrated_probability", "expected_value", "edge", "decimal_odds", "recommended_bet"]
    for c in need_cols:
        if c not in portfolio.columns:
            portfolio[c] = pd.NA
    return portfolio[need_cols].sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=1000, simulations=1000)
