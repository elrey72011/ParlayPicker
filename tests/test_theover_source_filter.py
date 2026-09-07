import pandas as pd
from core.streamlit_pipeline import (
    _exclude_nonforecast_theover_values, _normalize_upload,
    _build_total_rows, _theover_upload_coverage,
)


def fixture():
    return pd.DataFrame({
        "League": ["MLB"] * 4,
        "HomeTeam": ["Boston", "Miami", "Atlanta", "Seattle"],
        "AwayTeam": ["Toronto", "Detroit", "Baltimore", "Texas"],
        "Market": ["Total"] * 4, "Line": [8.5] * 4,
        "WinProbability": [.5, .76, .61, .5],
        "WinProbSource": ["default_0.5", " PUBLIC_BETTING_PCT ", "model_probability", ""],
    })


def test_nonforecasts_do_not_enter_either_direction_or_coverage():
    raw = fixture()
    before = raw.copy(deep=True)
    normalized = _normalize_upload(raw)
    over, under = _build_total_rows(normalized)
    assert over.theover_probability.iloc[:2].isna().all()
    assert under.theover_probability.iloc[:2].isna().all()
    assert over.theover_probability.iloc[2:].tolist() == [.61, .5]
    assert len(over) == 4 and over.total_line.eq(8.5).all()
    assert _theover_upload_coverage(raw, "totals")["probability_game_count"] == 2
    pd.testing.assert_frame_equal(raw, before)


def test_probability_aliases_cannot_reintroduce_excluded_values():
    raw = pd.DataFrame({"win_prob_source": ["public_betting_pct"],
        "theover_probability": [.76], "winprobability": [.76],
        "win_probability": [.76], "probability": [.76]})
    filtered = _exclude_nonforecast_theover_values(raw)
    assert filtered.drop(columns="win_prob_source").isna().all().all()
    assert filtered.win_prob_source.iloc[0] == "public_betting_pct"
