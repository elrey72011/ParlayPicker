import pandas as pd
import streamlit as st


def render_portfolio(portfolio_df: pd.DataFrame) -> None:
    st.subheader("Portfolio Allocation")
    st.dataframe(portfolio_df, use_container_width=True)
