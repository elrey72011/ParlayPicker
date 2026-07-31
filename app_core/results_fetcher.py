import pandas as pd
import logging
from datetime import date
from typing import List
from app_core.espn_results import fetch_espn_results

logger = logging.getLogger(__name__)

def fetch_yesterdays_results(
    leagues: List[str],
    target_date: date | None = None,
    attempts: int = 2,
) -> pd.DataFrame:
    logger.info(f"Routing result fetch to ESPN for leagues: {leagues}")

    try:
        import streamlit as st

        if "restricted_leagues" in st.session_state:
            st.session_state["restricted_leagues"] = set()
    except Exception:
        pass

    df = fetch_espn_results(leagues, target_date=target_date, attempts=attempts)

    unsupported = set(df.attrs.get("unsupported_leagues", []))
    try:
        import streamlit as st

        st.session_state["unsupported_result_leagues"] = unsupported
    except Exception:
        pass
    return df
