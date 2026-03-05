from parlaypicker.core.ev_engine import expected_value, edge_percentage, bet_signal


def test_expected_value_positive_edge():
    assert expected_value(0.55, -110) > 0


def test_edge_percentage():
    assert round(edge_percentage(0.55, 0.5), 6) == 0.05


def test_bet_signal():
    assert bet_signal(0.04)
