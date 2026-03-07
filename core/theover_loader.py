from __future__ import annotations

import pandas as pd

THEOVER_COLUMN_ALIASES = {
    "league": ["league", "sport", "competition"],
    "home_team": ["home_team", "hometeam", "home", "home team"],
    "away_team": ["away_team", "awayteam", "away", "away team"],
    "game_date": ["game_date", "commence_time", "start_time", "time", "date"],
    "market": ["market", "bet_type", "wager_type", "pick_type", "market_type"],
    "pick": ["pick", "selection", "side", "over_under", "ou"],
    "pickteam": ["pickteam", "pick_team", "team", "selection_team"],
    "line": ["line", "spread", "spread_line", "total", "total_line", "points"],
    "winprobability": ["winprobability", "win_probability", "probability", "win_prob"],
    "odds_american": ["odds_american", "odds", "american_odds"],
}


def normalize_theover_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    normalized = df.copy()
    normalized.columns = (
        normalized.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    rename_map: dict[str, str] = {}
    for canonical, aliases in THEOVER_COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = alias.strip().lower().replace(" ", "_")
            if alias_key in normalized.columns and canonical not in normalized.columns:
                rename_map[alias_key] = canonical
                break
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce", utc=True)

    if "line" in normalized.columns:
        normalized["line"] = pd.to_numeric(normalized["line"], errors="coerce")

    if "winprobability" in normalized.columns:
        probs = pd.to_numeric(normalized["winprobability"], errors="coerce")
        probs = probs.where(~((probs > 1.0) & (probs <= 100.0)), probs / 100.0)
        normalized["winprobability"] = probs.clip(0.0, 1.0)

    if "odds_american" in normalized.columns:
        normalized["odds_american"] = pd.to_numeric(normalized["odds_american"], errors="coerce")

    return normalized


def _ensure_required_theover_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for date_col in ["commence_time", "start_time", "date"]:
        if date_col not in out.columns:
            out[date_col] = out.get("game_date", pd.NaT)
        out[f"{date_col}_as_game_date"] = pd.to_datetime(out[date_col], errors="coerce", utc=True)

    if "spread_line" not in out.columns:
        out["spread_line"] = pd.to_numeric(out.get("line"), errors="coerce")
    if "total_line" not in out.columns:
        out["total_line"] = pd.to_numeric(out.get("line"), errors="coerce")

    prob = pd.to_numeric(out.get("winprobability"), errors="coerce")
    out["theover_probability_spread"] = pd.to_numeric(out.get("theover_probability_spread", prob), errors="coerce")
    out["theover_probability_total"] = pd.to_numeric(out.get("theover_probability_total", prob), errors="coerce")

    required = [
        "league", "home_team", "away_team", "spread_line", "total_line",
        "commence_time_as_game_date", "start_time_as_game_date", "date_as_game_date",
        "theover_probability_spread", "theover_probability_total",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def load_theover_csv(uploaded_file) -> tuple[pd.DataFrame, str | None]:
    """Load and normalize a TheOver CSV upload.

    Returns (dataframe, error_message). error_message is None on success.
    Intentionally has NO st.* calls so it is safe to call before any
    Streamlit delta is emitted (avoids triggering spurious reruns).
    """
    if uploaded_file is None:
        return pd.DataFrame(), None

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.empty:
            return pd.DataFrame(), "Uploaded TheOver CSV is empty."
        normalized = normalize_theover_df(df)
        return _ensure_required_theover_columns(normalized), None
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "Uploaded CSV contains no readable data."
    except Exception as e:  # pragma: no cover
        return pd.DataFrame(), f"CSV loading error: {e}"
