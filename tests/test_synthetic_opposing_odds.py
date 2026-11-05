"""Synthetic opposite-side price used when no paired quote exists (20-cent line)."""
from core.streamlit_pipeline import american_to_prob, get_opposing_odds_from_exchange


def test_favourite_and_underdog_keep_twenty_cent_line():
    assert get_opposing_odds_from_exchange(-150) == 130
    assert get_opposing_odds_from_exchange(-120) == 100
    assert get_opposing_odds_from_exchange(110) == -130


def test_near_even_negative_price_faces_a_negative_price():
    assert get_opposing_odds_from_exchange(-108) == -112
    assert get_opposing_odds_from_exchange(-110) == -110
    assert get_opposing_odds_from_exchange(-102) == -118
    p = american_to_prob(-108)
    no_vig = p / (p + american_to_prob(get_opposing_odds_from_exchange(-108)))
    assert 0.495 < no_vig < 0.5
