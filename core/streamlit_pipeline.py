from __future__ import annotations

import logging
from itertools import combinations
import os
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.bankroll_simulator import simulate_bankroll
from core.kelly_optimizer import add_kelly_bet_sizing
from core.probability_calibration import calibrate_probabilities
from core.probability_engine import american_to_prob, normalize_probability_components, remove_vig
from core.schema.base_schema import ensure_base_schema
from core.team_mapper import normalize_team_name
from app_core.kalshi_integrator import enrich_with_kalshi_markets

logger = logging.getLogger(__name__)


try:
    from complete_workflow_implementation import run_ml_predictions
except Exception:  # pragma: no cover
    run_ml_predictions = None


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

THEOVER_COLUMN_ALIASES = {
    "league": ["league"],
    "home_team": ["home_team", "hometeam", "home", "home team", "team_home"],
    "away_team": ["away_team", "awayteam", "away", "away team", "team_away"],
    "game_date": ["game_date", "date", "commence_time", "time", "start_time"],
    "market": ["market", "bet_type", "market_type", "wager_type", "pick_type"],
    "pick": ["pick", "selection", "side", "o/u", "over_under"],
    "pickteam": ["pickteam", "pick_team", "team", "selection_team"],
    "line": ["line", "spread", "spread_line", "total", "total_line", "points", "number"],
    "winprobability": ["winprobability", "probability", "win_prob", "win_probability"],
}


def normalize_theover_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip().str.lower()

    rename_map: dict[str, str] = {}
    for canonical, aliases in THEOVER_COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.strip().lower()
            if key in normalized.columns and canonical not in normalized.columns:
                rename_map[key] = canonical
                break
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce")
    if "line" in normalized.columns:
        normalized["line"] = pd.to_numeric(normalized["line"], errors="coerce")
    if "winprobability" in normalized.columns:
        normalized["winprobability"] = pd.to_numeric(normalized["winprobability"], errors="coerce")

    return _normalize_key_columns(normalized)


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    base_keys = [k for k in ["league", "home_team", "away_team"] if k in left.columns and k in right.columns]
    if "game_date" not in left.columns or "game_date" not in right.columns:
        return base_keys

    left_dates = pd.to_datetime(left["game_date"], errors="coerce")
    right_dates = pd.to_datetime(right["game_date"], errors="coerce")
    left_coverage = left_dates.notna().mean() if len(left_dates) else 0.0
    right_coverage = right_dates.notna().mean() if len(right_dates) else 0.0

    if left_coverage < 0.5 or right_coverage < 0.5:
        return base_keys

    overlap = set(left_dates.dropna().dt.date.unique()) & set(right_dates.dropna().dt.date.unique())
    if not overlap:
        return base_keys

    return base_keys + ["game_date"]


def infer_market_type_and_lines(row: pd.Series) -> pd.Series:
    pick = str(row.get("pick") or "").lower()
    market = str(row.get("market") or "").lower()
    line = pd.to_numeric(row.get("line"), errors="coerce")
    pickteam = normalize_team_name(row.get("pickteam"))
    home_team = normalize_team_name(row.get("home_team"))

    market_type = ""
    spread_line = pd.NA
    total_line = pd.NA

    if "over" in pick or "over" in market:
        market_type = "total_over"
        total_line = line
    elif "under" in pick or "under" in market:
        market_type = "total_under"
        total_line = line
    else:
        is_home_pick = bool(pickteam and home_team and pickteam == home_team)
        market_type = "spread_home" if is_home_pick else "spread_away"
        spread_line = line if is_home_pick else (-line if pd.notna(line) else pd.NA)

    return pd.Series({"market_type": market_type, "spread_line": spread_line, "total_line": total_line})


def infer_market_type_from_row(row: pd.Series) -> str:
    """Infer canonical market type for an analysis row."""
    allowed_market_types = {
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
        "moneyline_home",
        "moneyline_away",
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
            str(row.get("best_pick") or ""),
            str(row.get("team") or ""),
        ]
    ).lower()

    spread_candidates = [row.get("spread"), row.get("spread_line"), row.get("line")]
    total_candidates = [row.get("total"), row.get("total_line"), row.get("total_points"), row.get("points")]
    spread_val = pd.Series(spread_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    total_val = pd.Series(total_candidates).apply(pd.to_numeric, errors="coerce").dropna()
    spread_num = spread_val.iloc[0] if not spread_val.empty else np.nan
    total_num = total_val.iloc[0] if not total_val.empty else np.nan

    pick_team = normalize_team_name(row.get("team") or row.get("selection") or row.get("pick") or row.get("pickteam"))
    home_team = normalize_team_name(row.get("home_team"))
    away_team = normalize_team_name(row.get("away_team"))

    is_home_pick = bool(row.get("is_home_pick", False))
    if pick_team and home_team and pick_team == home_team:
        is_home_pick = True
    elif pick_team and away_team and pick_team == away_team:
        is_home_pick = False

    has_moneyline_text = any(token in market_hint for token in ["moneyline", "ml", "to win", "winner"])
    has_total_text = any(token in market_hint for token in ["total", "over", "under", "o/u", "points"])
    has_spread_text = any(token in market_hint for token in ["spread", "ats", "handicap"])

    if "under" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_under"
    if "over" in market_hint and (has_total_text or pd.notna(total_num)):
        return "total_over"

    if has_spread_text or pd.notna(spread_num):
        return "spread_home" if is_home_pick else "spread_away"

    if has_moneyline_text:
        return "moneyline_home" if is_home_pick else "moneyline_away"

    # Fallback when team is explicit but market family is not.
    if pick_team and (pick_team == home_team or pick_team == away_team):
        return "moneyline_home" if pick_team == home_team else "moneyline_away"

    return "unknown"


def _infer_market_type(row: pd.Series) -> str:
    return infer_market_type_from_row(row)


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
        spread_display = _format_signed_spread(row.get("spread_line", row.get("spread")))
        return f"{row['home_team']} {spread_display}".strip()

    if row["market_type"] == "spread_away":
        spread_display = _format_signed_spread(row.get("spread_line", row.get("spread")))
        return f"{row['away_team']} {spread_display}".strip()

    if row["market_type"] == "total_over":
        return f"Over {_format_total(row.get('total_line', row.get('total')))}".strip()

    if row["market_type"] == "total_under":
        return f"Under {_format_total(row.get('total_line', row.get('total')))}".strip()

    return ""


def _ensure_best_pick_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    resolved = df.copy()
    if "market_type" not in resolved.columns:
        resolved["market_type"] = resolved.apply(_infer_market_type, axis=1)
    else:
        market_type_series = resolved["market_type"].astype(str).str.strip().str.lower()
        missing_market_type = market_type_series.eq("") | market_type_series.eq("nan")
        if missing_market_type.any():
            resolved.loc[missing_market_type, "market_type"] = resolved.loc[missing_market_type].apply(_infer_market_type, axis=1)

    if "spread_line" not in resolved.columns:
        resolved["spread_line"] = pd.to_numeric(resolved.get("spread"), errors="coerce")
    else:
        resolved["spread_line"] = pd.to_numeric(resolved["spread_line"], errors="coerce")
        fallback_spread = pd.to_numeric(resolved.get("spread"), errors="coerce")
        resolved["spread_line"] = resolved["spread_line"].where(resolved["spread_line"].notna(), fallback_spread)

    if "total_line" not in resolved.columns:
        resolved["total_line"] = pd.to_numeric(resolved.get("total"), errors="coerce")
    else:
        resolved["total_line"] = pd.to_numeric(resolved["total_line"], errors="coerce")
        fallback_total = pd.to_numeric(resolved.get("total"), errors="coerce")
        resolved["total_line"] = resolved["total_line"].where(resolved["total_line"].notna(), fallback_total)

    generated_best_pick = resolved.apply(format_pick, axis=1)
    if "best_pick" in resolved.columns:
        best_pick_series = resolved["best_pick"].fillna("").astype(str).str.strip()
        resolved["best_pick"] = best_pick_series.where(best_pick_series.str.len() > 0, generated_best_pick)
    else:
        resolved["best_pick"] = generated_best_pick

    resolved["best_pick"] = resolved["best_pick"].fillna("").astype(str).str.strip()
    return resolved


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

    if "expected_value" not in df.columns:
        df["expected_value"] = pd.NA
    if "edge" not in df.columns:
        df["edge"] = pd.NA

    df["expected_value"] = pd.to_numeric(df["expected_value"], errors="coerce")
    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")

    best_picks = (
        df.sort_values(["expected_value", "edge"], ascending=[False, False])
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




def _numeric_series(df, col, default=None):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return s.fillna(default) if default is not None else s


def _string_series(df, col, default=""):
    if col in df.columns:
        return df[col].fillna(default).astype("string")
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def _with_game_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["game_key"] = pd.Series(dtype="string")
        return out
    game_ts = pd.to_datetime(_string_series(out, "game_date"), errors="coerce", utc=True)
    game_day = game_ts.dt.strftime("%Y-%m-%d").fillna("")
    league = _string_series(out, "league").str.upper().str.strip()
    away = _string_series(out, "away_team").str.lower().str.strip()
    home = _string_series(out, "home_team").str.lower().str.strip()
    out["game_key"] = np.where(game_day.ne(""), league + "|" + game_day + "|" + away + "|" + home, league + "|" + away + "|" + home)
    return out


def is_stale_schedule(base_df: pd.DataFrame, theover_df: pd.DataFrame, now_utc: pd.Timestamp, max_age_days: int = 14) -> bool:  # type: ignore[override]
    now_ts = pd.to_datetime(now_utc, errors="coerce", utc=True)
    if pd.isna(now_ts):
        return False
    base_dates = pd.to_datetime(_string_series(base_df, "game_date"), errors="coerce", utc=True)
    theover_dates = pd.to_datetime(_string_series(theover_df, "game_date"), errors="coerce", utc=True)
    base_max = base_dates.max() if base_dates.notna().any() else pd.NaT
    upload_max = theover_dates.max() if theover_dates.notna().any() else pd.NaT
    stale_by_age = pd.notna(base_max) and base_max < (now_ts - pd.Timedelta(days=max_age_days))
    stale_vs_upload = pd.notna(base_max) and pd.notna(upload_max) and base_max < (upload_max - pd.Timedelta(days=1))
    missing_base_with_upload = pd.isna(base_max) and pd.notna(upload_max)
    return bool(stale_by_age or stale_vs_upload or missing_base_with_upload)


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:  # type: ignore[override]
    base_keys = [k for k in ["league", "home_team", "away_team"] if k in left.columns and k in right.columns]
    if "game_date" not in left.columns or "game_date" not in right.columns:
        return base_keys
    l = pd.to_datetime(_string_series(left, "game_date"), errors="coerce", utc=True)
    r = pd.to_datetime(_string_series(right, "game_date"), errors="coerce", utc=True)
    if l.notna().sum() == 0 or r.notna().sum() == 0:
        return base_keys
    overlap = set(l.dropna().dt.date.unique()) & set(r.dropna().dt.date.unique())
    return base_keys + ["game_date"] if overlap else base_keys


def _fill_missing_game_dates_from_base(bet_rows_df: pd.DataFrame, base_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    stats = {"date_fill_attempted": 0, "date_fill_filled": 0, "date_fill_rate": 0.0}
    if bet_rows_df.empty or base_df.empty or "game_date" not in bet_rows_df.columns:
        return bet_rows_df, stats
    out = bet_rows_df.copy()
    row_dates = pd.to_datetime(_string_series(out, "game_date"), errors="coerce", utc=True)
    missing_mask = row_dates.isna()
    attempted = int(missing_mask.sum())
    stats["date_fill_attempted"] = attempted
    if attempted == 0:
        return out, stats
    base_keys = [k for k in ["league", "home_team", "away_team"] if k in out.columns and k in base_df.columns]
    if not base_keys:
        return out, stats
    schedule = base_df[base_keys + ["game_date"]].copy()
    schedule["game_date"] = pd.to_datetime(_string_series(schedule, "game_date"), errors="coerce", utc=True)
    schedule = schedule.dropna(subset=["game_date"]).drop_duplicates(subset=base_keys).rename(columns={"game_date": "schedule_game_date"})
    if schedule.empty:
        return out, stats
    merged = out.merge(schedule, on=base_keys, how="left")
    schedule_dates = pd.to_datetime(_string_series(merged, "schedule_game_date"), errors="coerce", utc=True)
    can_fill = missing_mask & schedule_dates.notna()
    merged.loc[can_fill, "game_date"] = schedule_dates.loc[can_fill]
    merged = merged.drop(columns=["schedule_game_date"])
    filled = int(can_fill.sum())
    stats["date_fill_filled"] = filled
    stats["date_fill_rate"] = float(filled / max(attempted, 1))
    return merged, stats


def build_bet_rows(spreads_df: pd.DataFrame | None, totals_df: pd.DataFrame | None, selected_sports: list[str]) -> pd.DataFrame:
    schema = ['league','home_team','away_team','game_date','game_key','market_type','spread_line','total_line','theover_probability','odds_american','market_probability','ml_probability','expected_value','edge','best_pick']
    selected = {_normalize_league_value(s) for s in (selected_sports or [])}

    def prep(df: pd.DataFrame | None) -> pd.DataFrame:
        norm = normalize_theover_df(df)
        if norm.empty:
            return norm
        out = _normalize_key_columns(norm)
        out = _enrich_uploaded_league(out, list(selected))
        out['league'] = _string_series(out, 'league').apply(_normalize_league_value)
        out['game_date'] = pd.to_datetime(_string_series(out, 'game_date'), errors='coerce', utc=True)
        out['spread_line'] = _numeric_series(out, 'spread_line').where(_numeric_series(out, 'spread_line').notna(), _numeric_series(out, 'line'))
        out['total_line'] = _numeric_series(out, 'total_line').where(_numeric_series(out, 'total_line').notna(), _numeric_series(out, 'line'))
        out['theover_probability'] = _numeric_series(out, 'winprobability').clip(0.0, 1.0)
        out['odds_american'] = _numeric_series(out, 'odds_american').where(_numeric_series(out, 'odds_american').notna(), _numeric_series(out, 'american_odds')).fillna(-110.0)
        out['home_team'] = _string_series(out, 'home_team')
        out['away_team'] = _string_series(out, 'away_team')
        return out

    spreads = prep(spreads_df)
    totals = prep(totals_df)

    spread_rows = pd.DataFrame()
    if not spreads.empty:
        base = spreads[spreads['spread_line'].notna()].copy()
        home_rows = base.copy()
        home_rows['market_type'] = 'spread_home'
        home_rows['spread_line'] = _numeric_series(home_rows, 'spread_line')
        away_rows = base.copy()
        away_rows['market_type'] = 'spread_away'
        away_rows['spread_line'] = -_numeric_series(away_rows, 'spread_line')
        spread_rows = pd.concat([home_rows, away_rows], ignore_index=True)
        spread_rows['total_line'] = pd.NA

    total_rows = pd.DataFrame()
    if not totals.empty:
        base = totals[totals['total_line'].notna()].copy()
        over_rows = base.copy()
        over_rows['market_type'] = 'total_over'
        under_rows = base.copy()
        under_rows['market_type'] = 'total_under'
        total_rows = pd.concat([over_rows, under_rows], ignore_index=True)
        total_rows['spread_line'] = pd.NA

    bet_rows = pd.concat([spread_rows, total_rows], ignore_index=True)
    if bet_rows.empty:
        return pd.DataFrame(columns=schema)
    if selected:
        bet_rows = bet_rows[bet_rows['league'].isin(selected)].copy()

    bet_rows = _with_game_key(bet_rows)
    bet_rows['market_probability'] = bet_rows['odds_american'].apply(american_to_prob)
    bet_rows['ml_probability'] = bet_rows['theover_probability']
    for c in schema:
        if c not in bet_rows.columns:
            bet_rows[c] = pd.NA
    return bet_rows[schema]


def build_theover_bet_rows(spreads_df: pd.DataFrame | None, totals_df: pd.DataFrame | None, selected_sports: list[str]) -> pd.DataFrame:  # type: ignore[override]
    return build_bet_rows(spreads_df, totals_df, selected_sports)
def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
    if analysis_df is None or analysis_df.empty:
        return _empty_best_picks_df()
    df = analysis_df.copy()
    allowed = {'spread_home', 'spread_away', 'total_over', 'total_under'}
    df['market_type'] = _string_series(df, 'market_type').str.lower().str.strip()
    df = df[df['market_type'].isin(allowed)].copy()
    if df.empty:
        return _empty_best_picks_df()

    df['expected_value'] = _numeric_series(df, 'expected_value')
    df['edge'] = _numeric_series(df, 'edge')
    df['spread_line'] = _numeric_series(df, 'spread_line')
    df['total_line'] = _numeric_series(df, 'total_line').where(_numeric_series(df, 'total_line').notna(), _numeric_series(df, 'total'))

    group_cols = ['league', 'home_team', 'away_team', 'game_date']
    available = [c for c in group_cols if c in df.columns]
    best = df.sort_values(['expected_value', 'edge'], ascending=[False, False]).groupby(available, dropna=False).head(1).copy()

    home = _string_series(best, 'home_team')
    away = _string_series(best, 'away_team')
    mt = _string_series(best, 'market_type').str.lower()
    best['best_pick'] = np.where(
        mt.eq('spread_home'), home + ' ' + best['spread_line'].map(lambda v: f"{v:+.1f}" if pd.notna(v) else ''),
        np.where(
            mt.eq('spread_away'), away + ' ' + best['spread_line'].map(lambda v: f"{v:+.1f}" if pd.notna(v) else ''),
            np.where(mt.eq('total_over'), 'Over ' + best['total_line'].map(lambda v: f"{v:.1f}" if pd.notna(v) else ''), 'Under ' + best['total_line'].map(lambda v: f"{v:.1f}" if pd.notna(v) else '')),
        ),
    )
    for c in BEST_PICK_COLUMNS:
        if c not in best.columns:
            best[c] = pd.NA
    return best[BEST_PICK_COLUMNS].reset_index(drop=True)


@st.cache_data(ttl=180)
def run_analysis_pipeline(  # type: ignore[override]
    sports: Iterable[str],
    max_rows: int,
    use_ml: bool = True,
    spreads_df: pd.DataFrame | None = None,
    totals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str | None]]:
    now_utc = pd.to_datetime(pd.Timestamp.utcnow(), errors='coerce', utc=True)
    selected_sports = _normalize_sports_filter(sports)

    base_df = load_base_data().copy()
    base_df['game_date'] = pd.to_datetime(_string_series(base_df, 'game_date'), errors='coerce', utc=True)
    if selected_sports and 'league' in base_df.columns:
        base_df = base_df[base_df['league'].isin(selected_sports)].copy()

    spreads_norm = normalize_theover_df(spreads_df)
    totals_norm = normalize_theover_df(totals_df)
    spreads_games = _with_game_key(_normalize_key_columns(spreads_norm.copy())) if not spreads_norm.empty else pd.DataFrame(columns=['game_key'])
    totals_games = _with_game_key(_normalize_key_columns(totals_norm.copy())) if not totals_norm.empty else pd.DataFrame(columns=['game_key'])

    bet_rows_df = build_bet_rows(spreads_df, totals_df, selected_sports)
    base_stale = is_stale_schedule(base_df, bet_rows_df, now_utc)
    bet_rows_df, date_fill_stats = _fill_missing_game_dates_from_base(bet_rows_df, base_df)

    if not bet_rows_df.empty:
        analysis_input = bet_rows_df.head(max_rows).copy()
        merge_keys_used: list[str] = []
        if not base_stale and not base_df.empty:
            merge_keys = choose_merge_keys(analysis_input, base_df)
            merge_keys_used = merge_keys
            extra_cols = [c for c in base_df.columns if c not in analysis_input.columns and c not in merge_keys]
            if merge_keys and extra_cols:
                left_before = len(analysis_input)
                merged = analysis_input.merge(base_df[merge_keys + extra_cols], on=merge_keys, how='left')
                if len(merged) == left_before:
                    analysis_input = merged
    else:
        merge_keys_used = []
        analysis_input = base_df.head(max_rows).copy()

    analyzed = _apply_analysis_calculations(analysis_input)
    if analyzed.empty:
        di = {
            'total_games': 0, 'bet_rows': 0, 'best_picks': 0,
            'theover_spreads_rows': int(len(spreads_norm)), 'theover_totals_rows': int(len(totals_norm)),
            'theover_spreads_games': int(spreads_games['game_key'].nunique() if 'game_key' in spreads_games else 0),
            'theover_totals_games': int(totals_games['game_key'].nunique() if 'game_key' in totals_games else 0),
            'theover_spreads_bet_games': 0, 'theover_totals_bet_games': 0,
            'market_type_counts': {}, 'merge_keys_used': merge_keys_used, 'base_stale': bool(base_stale),
            'base_max_date': None, 'theover_max_date': None, 'bet_rows_max_date': None, 'now_utc': None if pd.isna(now_utc) else str(now_utc),
            'kalshi_attempted': 0, 'kalshi_matches': 0, 'kalshi_match_rate': 0.0, 'match_rate': 0.0,
            **date_fill_stats,
        }
        return analyzed, _empty_best_picks_df(), di

    analyzed['market_type'] = _string_series(analyzed, 'market_type').str.lower().str.strip()
    analyzed['model_probability'] = _numeric_series(analyzed, 'theover_probability').fillna(_numeric_series(analyzed, 'model_probability')).fillna(0.5).clip(0.01, 0.99)
    analyzed = calibrate_probabilities(analyzed)
    analyzed['calibrated_probability'] = _numeric_series(analyzed, 'calibrated_probability').fillna(analyzed['model_probability']).clip(0.01, 0.99)
    analyzed['odds_american'] = _numeric_series(analyzed, 'odds_american', -110.0)
    analyzed['market_probability'] = _numeric_series(analyzed, 'market_probability').fillna(analyzed['odds_american'].apply(american_to_prob))
    analyzed['ml_probability'] = _numeric_series(analyzed, 'ml_probability').fillna(analyzed['model_probability'])
    analyzed['decimal_odds'] = analyzed['odds_american'].apply(american_to_decimal)
    analyzed['expected_value'] = analyzed['calibrated_probability'] * (analyzed['decimal_odds'] - 1) - (1 - analyzed['calibrated_probability'])
    analyzed['edge'] = analyzed['calibrated_probability'] - analyzed['market_probability']

    best_picks_df = build_best_picks_df(analyzed)
    if not best_picks_df.empty:
        best_picks_df = enrich_with_kalshi_markets(best_picks_df)

    market_counts = _string_series(analyzed, 'market_type').value_counts(dropna=False).to_dict()
    if not bet_rows_df.empty and 'game_key' not in bet_rows_df.columns:
        bet_rows_df = _with_game_key(bet_rows_df)
    spread_games = int(bet_rows_df[_string_series(bet_rows_df, 'market_type').str.startswith('spread', na=False)]['game_key'].nunique()) if (not bet_rows_df.empty and 'game_key' in bet_rows_df.columns) else 0
    total_games = int(bet_rows_df[_string_series(bet_rows_df, 'market_type').str.startswith('total', na=False)]['game_key'].nunique()) if (not bet_rows_df.empty and 'game_key' in bet_rows_df.columns) else 0
    kalshi_matches = int(_string_series(best_picks_df, 'kalshi_match_status').str.lower().eq('matched').sum()) if not best_picks_df.empty else 0
    base_max_date = pd.to_datetime(_string_series(base_df, 'game_date'), errors='coerce', utc=True).max() if not base_df.empty else pd.NaT
    theover_max_date = pd.to_datetime(_string_series(bet_rows_df, 'game_date'), errors='coerce', utc=True).max() if not bet_rows_df.empty else pd.NaT
    di = {
        'total_games': int(analyzed[['league','home_team','away_team','game_date']].drop_duplicates().shape[0]) if set(['league','home_team','away_team','game_date']).issubset(analyzed.columns) else int(len(analyzed)),
        'bet_rows': int(len(bet_rows_df)),
        'best_picks': int(len(best_picks_df)),
        'theover_spreads_rows': int(len(spreads_norm)),
        'theover_totals_rows': int(len(totals_norm)),
        'theover_spreads_games': int(spreads_games['game_key'].nunique() if 'game_key' in spreads_games else 0),
        'theover_totals_games': int(totals_games['game_key'].nunique() if 'game_key' in totals_games else 0),
        'theover_spreads_bet_games': spread_games,
        'theover_totals_bet_games': total_games,
        'market_type_counts': market_counts,
        'base_max_date': None if pd.isna(base_max_date) else str(base_max_date),
        'theover_max_date': None if pd.isna(theover_max_date) else str(theover_max_date),
        'bet_rows_max_date': None if pd.isna(theover_max_date) else str(theover_max_date),
        'merge_keys_used': merge_keys_used,
        'base_stale': bool(base_stale),
        'now_utc': None if pd.isna(now_utc) else str(now_utc),
        'kalshi_attempted': int(len(best_picks_df)),
        'kalshi_matches': kalshi_matches,
        'kalshi_match_rate': float(kalshi_matches / max(len(best_picks_df), 1)),
        'match_rate': float(kalshi_matches / max(len(best_picks_df), 1)),
        **date_fill_stats,
    }
    return analyzed, best_picks_df, di


def generate_parlays(analysis_df: pd.DataFrame, max_legs: int = 5) -> pd.DataFrame:
    columns = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=columns)
    df = analysis_df.copy()
    if "best_pick" not in df.columns:
        df = _ensure_best_pick_column(df)
    df = df[_string_series(df, "best_pick").str.len() > 0].copy()
    if len(df) < 2:
        return pd.DataFrame(columns=columns)
    df["calibrated_probability"] = _numeric_series(df, "calibrated_probability", 0.5).clip(0.01, 0.99)
    decimal = _numeric_series(df, "decimal_odds")
    fallback_decimal = _numeric_series(df, "odds_american", -110.0).apply(american_to_decimal)
    df["decimal_odds"] = decimal.where(decimal.notna(), fallback_decimal).fillna(1.9091)

    records: list[dict[str, float | int | str]] = []
    for leg_count in range(2, min(max_legs, len(df)) + 1):
        for combo in combinations(df.index.tolist(), leg_count):
            legs = df.loc[list(combo)]
            combined_probability = float(legs["calibrated_probability"].prod())
            combined_decimal_odds = float(legs["decimal_odds"].prod())
            parlay_ev = combined_probability * (combined_decimal_odds - 1) - (1 - combined_probability)
            labels = [f"{r.home_team} vs {r.away_team}: {r.best_pick}" for r in legs.itertuples()]
            records.append(
                {
                    "parlay_legs": " | ".join(labels),
                    "combined_probability": combined_probability,
                    "combined_decimal_odds": combined_decimal_odds,
                    "parlay_ev": parlay_ev,
                    "legs": leg_count,
                }
            )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)[columns].sort_values("parlay_ev", ascending=False).reset_index(drop=True)


def build_realtime_edges(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame()
    df = _ensure_best_pick_column(analysis_df.copy())
    keep = [c for c in ["league", "home_team", "away_team", "best_pick", "market_type", "calibrated_probability", "market_probability", "decimal_odds", "expected_value", "edge"] if c in df.columns]
    return df[keep].sort_values("edge", ascending=False).head(25) if keep else df.head(25)


def optimize_portfolio_allocation(analysis_df: pd.DataFrame, bankroll: float = 1000.0) -> pd.DataFrame:
    edges = build_realtime_edges(analysis_df)
    if edges.empty:
        return edges
    if "decimal_odds" not in edges.columns:
        edges["decimal_odds"] = _numeric_series(edges, "odds_american", -110.0).apply(american_to_decimal)
    edges["decimal_odds"] = _numeric_series(edges, "decimal_odds").fillna(1.9091)
    portfolio = add_kelly_bet_sizing(edges, bankroll=bankroll, fraction=0.25)
    if "best_pick" not in portfolio.columns:
        portfolio = _ensure_best_pick_column(portfolio)
    if "recommended_bet" not in portfolio.columns:
        portfolio["recommended_bet"] = 0.0
    return portfolio.sort_values("edge", ascending=False).reset_index(drop=True)


def run_bankroll_simulation(portfolio_df: pd.DataFrame, bankroll: float) -> dict[str, float | list[list[float]]]:
    return simulate_bankroll(portfolio_df=portfolio_df, starting_bankroll=bankroll, days=1000, simulations=1000)
