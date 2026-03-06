import pandas as pd
import streamlit as st


def load_theover_csv(uploaded_file):

    if uploaded_file is None:
        return pd.DataFrame()

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("Uploaded TheOver CSV is empty.")
            return pd.DataFrame()

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )

        canonical_map = {
            "hometeam": "home_team",
            "home": "home_team",
            "awayteam": "away_team",
            "away": "away_team",
            "sport": "league",
            "bet_type": "market",
            "commence_time": "game_date",
            "date": "game_date",
            "time": "game_date",
            "start_time": "game_date",
            "win_probability": "winprobability",
            "probability": "winprobability",
            "market": "market",
            "win_prob": "winprobability",
            "spread": "line",
            "total": "line",
            "spread_line": "line",
            "total_line": "line",
            "pick_team": "pickteam",
        }
        for src, dst in canonical_map.items():
            if src in df.columns and dst not in df.columns:
                df = df.rename(columns={src: dst})

        if "winprobability" in df.columns:
            df["winprobability"] = pd.to_numeric(df["winprobability"], errors="coerce")
            if df["winprobability"].dropna().gt(1.0).mean() > 0.8:
                df["winprobability"] = df["winprobability"] / 100.0
            df["winprobability"] = df["winprobability"].clip(0.0, 1.0)

        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce", utc=True)

        return df

    except pd.errors.EmptyDataError:

        st.warning("Uploaded CSV contains no readable data.")
        return pd.DataFrame()

    except Exception as e:

        st.error(f"CSV loading error: {e}")
        return pd.DataFrame()
