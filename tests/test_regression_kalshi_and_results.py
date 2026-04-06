import contextlib
import io
import types

import pandas as pd
import pytest

from app_core.results_ingestion import attach_results
from app_core.weights_config import (
    KALSHI_WEIGHT,
    MARKET_WEIGHT,
    ML_MODEL_WEIGHT,
    SENTIMENT_WEIGHT,
    THEOVER_WEIGHT,
)
from core.streamlit_pipeline import compute_blended_probability


class FakeUpload(io.StringIO):
    def __init__(self, text: str, name: str):
        super().__init__(text)
        self.name = name
        self.size = len(text.encode("utf-8"))


class FakeColumn:
    def metric(self, *args, **kwargs):
        return None


@pytest.mark.parametrize("market_type", ["spread_away", "spread_home"])
def test_kalshi_pick_side_probability_is_not_flipped_for_spreads(market_type):
    market = pd.Series([0.52])
    kalshi = pd.Series([0.62])
    ml = pd.Series([0.57])
    theover = pd.Series([0.55])
    sentiment = pd.Series([0.50])

    blended = compute_blended_probability(
        p_market=market,
        p_kalshi=kalshi,
        p_ml=ml,
        p_theover=theover,
        p_sentiment=sentiment,
        league=pd.Series(["NBA"]),
        market_type=pd.Series([market_type]),
    )

    expected = (
        (0.62 * KALSHI_WEIGHT)
        + (0.52 * MARKET_WEIGHT)
        + (0.57 * ML_MODEL_WEIGHT)
        + (0.55 * THEOVER_WEIGHT)
        + (0.50 * SENTIMENT_WEIGHT)
    )

    assert blended.iloc[0] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("market_type", ["total_under", "total_over"])
def test_kalshi_pick_side_probability_is_not_flipped_for_totals(market_type):
    market = pd.Series([0.51])
    kalshi = pd.Series([0.61])
    ml = pd.Series([0.56])
    theover = pd.Series([0.54])
    sentiment = pd.Series([0.50])

    blended = compute_blended_probability(
        p_market=market,
        p_kalshi=kalshi,
        p_ml=ml,
        p_theover=theover,
        p_sentiment=sentiment,
        league=pd.Series(["NBA"]),
        market_type=pd.Series([market_type]),
    )

    expected = (
        (0.61 * KALSHI_WEIGHT)
        + (0.51 * MARKET_WEIGHT)
        + (0.56 * ML_MODEL_WEIGHT)
        + (0.54 * THEOVER_WEIGHT)
        + (0.50 * SENTIMENT_WEIGHT)
    )

    assert blended.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_attach_results_grades_multi_word_away_spread_correctly():
    master_df = pd.DataFrame(
        [
            {
                "league": "NBA",
                "home_team": "Boston Celtics",
                "away_team": "New Orleans Pelicans",
                "best_pick": "New Orleans Pelicans +4.5",
                "Pick": "New Orleans Pelicans +4.5",
            }
        ]
    )

    results_df = pd.DataFrame(
        [
            {
                "league": "NBA",
                "home_team": "Boston Celtics",
                "away_team": "New Orleans Pelicans",
                "home_score": 100,
                "away_score": 98,
                "date": "2026-03-17",
            }
        ]
    )

    out = attach_results(master_df, results_df)

    assert out.loc[0, "spread_result"] == "WIN"
    assert out.loc[0, "Pick_Outcome"] == "WIN"


def test_results_dashboard_resets_session_state_for_new_upload(monkeypatch):
    from app.ui import results_dashboard as rd

    csv_one = (
        "league,home_team,away_team,best_pick,Pick_Status,odds_american\n"
        "NBA,Boston Celtics,New Orleans Pelicans,New Orleans Pelicans +4.5,Actionable,-110\n"
    )
    csv_two = (
        "league,home_team,away_team,best_pick,Pick_Status,odds_american\n"
        "NBA,Los Angeles Lakers,Golden State Warriors,Golden State Warriors +3.5,Actionable,-110\n"
    )

    uploads = {
        "first": FakeUpload(csv_one, "first.csv"),
        "second": FakeUpload(csv_two, "second.csv"),
    }

    session_state = {}

    def install_fake_streamlit(current_upload):
        def file_uploader(label, type=None, key=None):
            if key == "perf_picks_uploader":
                return current_upload
            return None

        fake_st = types.SimpleNamespace(
            session_state=session_state,
            file_uploader=file_uploader,
            subheader=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            success=lambda *a, **k: None,
            error=lambda *a, **k: None,
            info=lambda *a, **k: None,
            toggle=lambda *a, **k: False,
            divider=lambda *a, **k: None,
            data_editor=lambda table_df, **k: table_df,
            columns=lambda n: tuple(FakeColumn() for _ in range(n)),
            rerun=lambda: None,
            spinner=contextlib.nullcontext,
            column_config=types.SimpleNamespace(
                NumberColumn=lambda *a, **k: None,
                SelectboxColumn=lambda *a, **k: None,
            ),
        )
        monkeypatch.setattr(rd, "st", fake_st)

    install_fake_streamlit(uploads["first"])
    rd.render_results_dashboard(None)
    assert session_state["perf_edited_picks"].iloc[0]["best_pick"] == "New Orleans Pelicans +4.5"

    session_state["perf_edited_picks"].loc[:, "best_pick"] = "STALE VALUE"

    install_fake_streamlit(uploads["second"])
    rd.render_results_dashboard(None)

    assert session_state["perf_picks_file_identifier"] == "second.csv_135"
    assert session_state["perf_edited_picks"].iloc[0]["best_pick"] == "Golden State Warriors +3.5"
    assert "STALE VALUE" not in session_state["perf_edited_picks"]["best_pick"].tolist()
