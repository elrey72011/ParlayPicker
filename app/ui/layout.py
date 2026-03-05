import streamlit as st


def setup_page() -> None:
    st.set_page_config(
        page_title="ParlayPicker",
        layout="wide",
        page_icon="📈",
    )
    st.title("ParlayPicker Sports Betting Analytics")
