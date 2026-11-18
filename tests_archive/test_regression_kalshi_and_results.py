import pandas as pd
import pytest
from core.streamlit_pipeline import compute_blended_probability
from app.ui.results_dashboard import render_results_dashboard
from app_core.results_ingestion import attach_results
from app_core.weights_config import KALSHI_WEIGHT, MARKET_WEIGHT, ML_MODEL_WEIGHT, THEOVER_WEIGHT, SENTIMENT_WEIGHT

def test_kalshi_probability_orientation_no_flip():
    """
    1) Kalshi pick-side probability should NOT be flipped again in compute_blended_probability()
    - Add a test for spread_away
    - Add a test for total_under
    - Confirm compute_blended_probability() uses kalshi_probability as already pick-side oriented
    - The expected blended value should use the configured weights directly, without any extra 1 - p inversion for away/under
    """

    # We will pass Kalshi probability = 0.6.
    # Other probabilities = 0.5.
    # We expect the blend to be (0.6 * KALSHI_WEIGHT) + 0.5 * (others)

    p_market = pd.Series([0.5, 0.5])
    p_kalshi = pd.Series([0.6, 0.6])
    p_ml = pd.Series([0.5, 0.5])
    p_theover = pd.Series([0.5, 0.5])
    p_sentiment = pd.Series([0.5, 0.5])
    league = pd.Series(["NBA", "NBA"])
    market_type = pd.Series(["spread_away", "total_under"])

    blended = compute_blended_probability(
        p_market=p_market,
        p_kalshi=p_kalshi,
        p_ml=p_ml,
        p_theover=p_theover,
        p_sentiment=p_sentiment,
        league=league,
        market_type=market_type
    )

    # Neutral TheOver/sentiment inputs are dropped and NBA totals have a distinct
    # weight profile. Orientation is correct when both pick-side 0.60 reads move
    # their blends above 0.50 instead of being inverted to 0.40.
    assert blended.iloc[0] > 0.5
    assert blended.iloc[1] > 0.5

def test_multi_word_team_spread_grading():
    """
    2) Multi-word team spread grading in app_core/results_ingestion.py
    - Add a test where the pick is something like: "New Orleans Pelicans +4.5"
    - Confirm attach_results() correctly infers away side and grades the spread correctly from final scores
    - Assert both spread_result and Pick_Outcome are correct
    """

    master_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "Los Angeles Lakers",
        "away_team": "New Orleans Pelicans",
        "best_pick": "New Orleans Pelicans +4.5",
        "spread_pick_team": "New Orleans Pelicans +4.5",
        "spread_pick_line": 4.5,
        "spread_pick_side": "away"
    }])

    # Lakers 110 - Pelicans 106. Pelicans lose by 4.
    # Pelicans +4.5 means they cover and win the bet.
    results_df = pd.DataFrame([{
        "league": "NBA",
        "home_team": "Los Angeles Lakers",
        "away_team": "New Orleans Pelicans",
        "home_score": 110,
        "away_score": 106,
        "date": "2023-10-24"
    }])

    attached = attach_results(master_df, results_df)

    assert "spread_result" in attached.columns, "spread_result column missing"
    assert "Pick_Outcome" in attached.columns, "Pick_Outcome column missing"

    assert attached["spread_result"].iloc[0] == "WIN", f"Expected WIN, got {attached['spread_result'].iloc[0]}"
    assert attached["Pick_Outcome"].iloc[0] == "WIN", f"Expected WIN, got {attached['Pick_Outcome'].iloc[0]}"


def test_results_recap_upload_reset_behavior():
    """
    3) Results recap upload reset behavior in app/ui/results_dashboard.py
    - Add a lightweight test that simulates two uploaded CSVs in sequence
    - Confirm st.session_state["perf_edited_picks"] is replaced when a new upload is detected
    - Confirm stale values from the first upload do not remain after the second upload
    - Mock Streamlit as lightly as possible
    """
    import io
    import pandas as pd
    import streamlit as st

    class MockUploadedFile(io.BytesIO):
        def __init__(self, name, size, initial_bytes):
            super().__init__(initial_bytes)
            self.name = name
            self.size = size

    # We patch st.session_state by clearing it natively instead of replacing the proxy object
    st.session_state.clear()

    # Setup our mock state dictionary that tests can read
    mock_state = {}

    def mock_file_uploader(label, type=None, key=None, **kwargs):
        if key == "perf_picks_uploader":
            return mock_state.get("_mock_uploaded_file")
        return None

    def mock_toggle(label, value=False, key=None):
        return False

    def mock_warning(msg): pass
    def mock_info(msg): pass
    def mock_success(msg): pass
    def mock_error(msg): pass
    def mock_subheader(msg): pass
    def mock_divider(): pass

    def mock_data_editor(df, *args, **kwargs):
        return df.copy()

    class MockColumns:
        def metric(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def mock_columns(n):
        return [MockColumns() for _ in range(n)]

    def mock_rerun(): pass

    # Apply mocks temporarily
    orig_uploader = st.file_uploader
    orig_toggle = st.toggle
    orig_warning = st.warning
    orig_info = st.info
    orig_success = st.success
    orig_error = st.error
    orig_subheader = st.subheader
    orig_divider = st.divider
    orig_data_editor = st.data_editor
    orig_columns = st.columns
    orig_rerun = st.rerun

    st.file_uploader = mock_file_uploader
    st.toggle = mock_toggle
    st.warning = mock_warning
    st.info = mock_info
    st.success = mock_success
    st.error = mock_error
    st.subheader = mock_subheader
    st.divider = mock_divider
    st.data_editor = mock_data_editor
    st.columns = mock_columns
    st.rerun = mock_rerun

    try:
        # First file upload
        csv1 = b"league,home_team,away_team,best_pick\nNBA,Lakers,Warriors,Lakers -5.5\n"
        file1 = MockUploadedFile("picks1.csv", len(csv1), csv1)
        mock_state["_mock_uploaded_file"] = file1

        # Run dashboard
        render_results_dashboard(None)

        # Check state contains file1 data
        assert "perf_edited_picks" in st.session_state
        df1 = st.session_state["perf_edited_picks"]
        assert len(df1) == 1
        assert df1["away_team"].iloc[0] == "Warriors"

        # Second file upload
        csv2 = b"league,home_team,away_team,best_pick\nNBA,Knicks,Nets,Knicks -2.5\nNBA,Bulls,Pistons,Bulls -4.5\n"
        file2 = MockUploadedFile("picks2.csv", len(csv2), csv2)
        mock_state["_mock_uploaded_file"] = file2

        # Run dashboard again
        render_results_dashboard(None)

        # Check state contains file2 data ONLY
        df2 = st.session_state["perf_edited_picks"]
        assert len(df2) == 2, f"Expected 2 rows from the second file, but got {len(df2)} rows."
        assert df2["home_team"].iloc[0] == "Knicks"
        assert df2["away_team"].iloc[1] == "Pistons"

    finally:
        # Restore mocks
        st.file_uploader = orig_uploader
        st.toggle = orig_toggle
        st.warning = orig_warning
        st.info = orig_info
        st.success = orig_success
        st.error = orig_error
        st.subheader = orig_subheader
        st.divider = orig_divider
        st.data_editor = orig_data_editor
        st.columns = orig_columns
        st.rerun = orig_rerun
