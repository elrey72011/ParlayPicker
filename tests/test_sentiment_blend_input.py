"""Sentiment reaches the blend the same way at all three blend sites.

``compute_blended_probability`` takes sentiment as a probability, ``0.5 + sentiment_diff * 0.5``, where
0.5 is no signal. The production blend (``run_analysis_pipeline``) and the Analysis-tab blend build it
that way. The re-blend after Kalshi (``streamlit_app._recompute_consensus_from_kalshi``), which sets the
final win estimate, passed the ``sentiment_diff`` column as it came, so 0.0 ("no sentiment" everywhere
else) would have been read as a 0% vote. Candidate frames carry no ``sentiment_diff`` today, so no
estimate shown was affected; these tests hold the three sites together for the day one does.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app_core.weights_config import FALLBACK_MARKET_WEIGHT, FALLBACK_SENTIMENT_WEIGHT

NO_COLUMN = object()        # the frame has no sentiment_diff column at all, as candidate frames have none today
KINDS = (("spread_home", -3.5, None), ("spread_away", 3.5, None), ("total_over", None, 145.5), ("total_under", None, 145.5))


def _game(*sentiment):
    """One college game, a row for each value given, every row priced at 50% with nothing but its price to blend."""
    rows = []
    for (kind, spread, total), value in zip(KINDS, sentiment):
        row = dict(matchup_id="g", league="NCAAB", home_team="Rice", away_team="Tulsa", market_type=kind,
                   market_probability=0.5, odds_american=-110, opposing_odds_american=-110, total_line=total,
                   spread_line=spread, theover_probability=None, ml_probability=None, kalshi_probability=None,
                   win_prob_source="not_provided")
        if value is not NO_COLUMN:
            row["sentiment_diff"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _as_the_production_blend_converts(frame):
    """What run_analysis_pipeline passes: a missing sentiment_diff is 0.0, then 0.5 + diff * 0.5, kept within 0 to 1."""
    raw = (pd.to_numeric(frame["sentiment_diff"], errors="coerce").fillna(0.0) if "sentiment_diff" in frame.columns
           else pd.Series([0.0] * len(frame), index=frame.index))
    return (0.5 + raw * 0.5).clip(0.0, 1.0)


def _re_blend(frame):
    import streamlit_app as app
    return app._recompute_consensus_from_kalshi(frame.copy())


def _re_blended(frame):
    return list(_re_blend(frame)["calibrated_probability"])


def _analysis_blended(frame):
    from core.streamlit_pipeline import _apply_analysis_calculations
    return list(_apply_analysis_calculations(frame.copy())["calibrated_probability"])


def _blended_directly(frame):
    """The blend itself, given the price and the sentiment as the production blend passes it."""
    from core.streamlit_pipeline import compute_blended_probability
    empty = pd.Series(float("nan"), index=frame.index)
    return list(compute_blended_probability(p_market=frame["market_probability"], p_kalshi=empty, p_ml=empty, p_theover=empty,
                                            p_sentiment=_as_the_production_blend_converts(frame), league=frame["league"],
                                            market_type=frame["market_type"]))


def _weighed(sentiment_probability):
    """A 50% price and a real sentiment, at the weights the blend uses when there is no Kalshi price."""
    return ((FALLBACK_MARKET_WEIGHT * 0.5 + FALLBACK_SENTIMENT_WEIGHT * sentiment_probability)
            / (FALLBACK_MARKET_WEIGHT + FALLBACK_SENTIMENT_WEIGHT))


@pytest.mark.parametrize("sentiment", [pytest.param(NO_COLUMN, id="no column"), pytest.param(None, id="empty"),
                                       pytest.param("n/a", id="not a number"), 0.0, 0.2, -0.2, 0.03, -0.03, 0.5, 1.0, -1.0])
def test_the_three_blends_agree_for_any_sentiment_on_its_scale(sentiment):
    frame = _game(sentiment)
    assert _re_blended(frame) == pytest.approx(_blended_directly(frame))
    assert _analysis_blended(frame) == pytest.approx(_blended_directly(frame))


@pytest.mark.parametrize("sentiment", [pytest.param(NO_COLUMN, id="no column"), pytest.param(None, id="empty"), 0.0])
def test_no_sentiment_leaves_a_price_only_pick_at_its_price(sentiment):
    frame = _game(sentiment)
    assert _re_blended(frame) == pytest.approx([0.5])           # the re-blend read 0.0 as a 0% vote and gave 0.3889
    assert _analysis_blended(frame) == pytest.approx([0.5])
    assert _blended_directly(frame) == pytest.approx([0.5])


@pytest.mark.parametrize("sentiment", [0.2, -0.2])
def test_a_real_sentiment_is_weighed_as_a_probability(sentiment):
    frame = _game(sentiment)
    estimate = _weighed(0.5 + sentiment * 0.5)                  # 0.2 is a 60% vote; the re-blend read it as a 20% one
    assert _re_blended(frame) == pytest.approx([estimate])
    assert _analysis_blended(frame) == pytest.approx([estimate])
    assert _blended_directly(frame) == pytest.approx([estimate])


def test_each_row_takes_its_own_sentiment():
    frame = _game(0.2, None, -0.2, 0.0)
    frame.index = [9, 4, 7, 1]
    estimates = [_weighed(0.6), 0.5, _weighed(0.4), 0.5]
    assert _re_blended(frame) == pytest.approx(estimates)
    assert _analysis_blended(frame) == pytest.approx(estimates)
    assert _blended_directly(frame) == pytest.approx(estimates)


@pytest.mark.parametrize("sentiment,counted_as", [(1.5, 1.0), (-1.5, 0.0), (40.0, 1.0)])
def test_beyond_its_scale_the_re_blend_keeps_sentiment_within_0_to_1_as_the_production_blend_does(sentiment, counted_as):
    frame = _game(sentiment)
    assert _re_blended(frame) == pytest.approx([_weighed(counted_as)])
    assert _re_blended(frame) == pytest.approx(_blended_directly(frame))


def test_the_refresh_record_holds_the_sentiment_the_blend_was_given(monkeypatch):
    """The evidence hooks are handed the converted value, so the recorded inputs reproduce the estimate."""
    from core.streamlit_pipeline import compute_blended_probability
    records = []
    monkeypatch.setattr("app_core.nfl_inference_evidence.retain_ui_reblend",
                        lambda before, after, inputs: records.append(inputs) or after)
    frame = _game(0.2, None, -0.2, 0.0)
    frame.index = [9, 4, 7, 1]
    out = _re_blend(frame)
    recorded = records[0]
    assert list(recorded["p_sentiment"]) == pytest.approx([0.6, 0.5, 0.4, 0.5])
    assert list(recorded["p_sentiment"]) == pytest.approx(list(_as_the_production_blend_converts(frame)))
    assert list(recorded["p_sentiment"].index) == [9, 4, 7, 1]
    replayed = compute_blended_probability(**recorded, league=frame["league"], market_type=frame["market_type"])
    assert list(replayed) == pytest.approx(list(out["calibrated_probability"]))
    _re_blend(_game(NO_COLUMN, NO_COLUMN))
    assert list(records[1]["p_sentiment"]) == pytest.approx([0.5, 0.5])      # no column: recorded as no signal, as before
