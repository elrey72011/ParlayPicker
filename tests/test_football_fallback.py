import pandas as pd
from core.football_fallback import market_input, selection_sources


def test_real_pairs_only_and_other_sports_unchanged():
    frame = pd.DataFrame({"league": ["NFL", "NCAAF", "MLB", "NFL"],
        "odds_american": [-110, -115, -110, -120],
        "opposing_odds_american": [None, -105, None, 0]})
    result = market_input(frame, pd.Series([.51] * 4))
    assert pd.isna(result[0]) and pd.isna(result[3])
    assert abs(result[1] - (115/215)/(115/215+105/205)) < 1e-12
    assert result[2] == .51


def test_missing_model_label_does_not_promote_or_modify_probability():
    frame = pd.DataFrame({"league": ["NFL", "NCAAF", "MLB", "NFL"],
                          "ml_probability": [None, None, None, .6]})
    assert list(selection_sources(frame)) == [
        "football_research_blend_no_independent_model",
        "football_research_blend_no_independent_model",
        "calibrated_probability", "calibrated_probability"]


def test_nfl_context_model_has_explicit_probability_provenance():
    frame = pd.DataFrame({
        "league": ["NFL", "NFL"],
        "ml_probability": [0.57, None],
        "ml_probability_source": ["score-distribution-v1:nfl", ""],
    })
    assert list(selection_sources(frame)) == [
        "nfl_score_distribution_recent_form_injury",
        "football_research_blend_no_independent_model",
    ]


def test_missing_football_market_is_not_a_synthetic_neutral_vote():
    from core.streamlit_pipeline import compute_blended_probability
    result = compute_blended_probability(
        pd.Series([float("nan")]), pd.Series([.6]), pd.Series([float("nan")]),
        pd.Series([float("nan")]), pd.Series([.5]), pd.Series(["NFL"]), pd.Series(["spread_home"]))
    assert abs(result.iloc[0] - .6) < 1e-12
