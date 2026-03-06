import pandas as pd
import streamlit as st

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


def load_theover_csv(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("Uploaded TheOver CSV is empty.")
            return pd.DataFrame()
        # Keep loader permissive; downstream pipeline normalization handles schema mapping.
        return df
    except pd.errors.EmptyDataError:
        st.warning("Uploaded CSV contains no readable data.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"CSV loading error: {e}")
        return pd.DataFrame()
