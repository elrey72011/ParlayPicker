import pandas as pd
import streamlit as st

THEOVER_COLUMN_ALIASES = {
    "league": ["league", "sport", "competition"],
    "home_team": ["home_team", "hometeam", "home"],
    "away_team": ["away_team", "awayteam", "away"],
    "game_date": ["game_date", "commence_time", "start_time", "time", "date"],
    "market": ["market", "bet_type", "wager_type", "pick_type"],
    "pick": ["pick", "selection", "side", "over_under"],
    "pickteam": ["pickteam", "pick_team", "team", "selection_team"],
    "line": ["line", "spread", "spread_line", "total", "total_line", "points"],
    "winprobability": ["winprobability", "win_probability", "probability", "win_prob"],
}


def normalize_theover_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()

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
            key = alias.strip().lower().replace(" ", "_")
            if key in normalized.columns and canonical not in normalized.columns:
                rename_map[key] = canonical
                break
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce")

    if "winprobability" in normalized.columns:
        probs = pd.to_numeric(normalized["winprobability"], errors="coerce")
        if probs.dropna().gt(1.0).any():
            probs = probs / 100.0
        normalized["winprobability"] = probs.clip(0.0, 1.0)

    return normalized


def load_theover_csv(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("Uploaded TheOver CSV is empty.")
            return pd.DataFrame()
        return normalize_theover_df(df)
    except pd.errors.EmptyDataError:
        st.warning("Uploaded CSV contains no readable data.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"CSV loading error: {e}")
        return pd.DataFrame()
