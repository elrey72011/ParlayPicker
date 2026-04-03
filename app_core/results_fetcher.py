import pandas as pd
import logging
import streamlit as st
from datetime import datetime, timedelta
from typing import List
from app_core.espn_results import fetch_espn_results

logger = logging.getLogger(__name__)

def fetch_yesterdays_results(leagues: List[str]) -> pd.DataFrame:
    logger.info(f"Routing result fetch to ESPN for leagues: {leagues}")

    try:
        if "restricted_leagues" in st.session_state:
            st.session_state["restricted_leagues"] = set()
    except Exception:
        pass

    df = fetch_espn_results(leagues)
    return df
