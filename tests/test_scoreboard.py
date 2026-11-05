"""SYNTHETIC scoreboard metrics (roadmap E1.3). Read-only; no network."""
import math

import pytest

from app_core.evaluation import scoreboard as sb
from scripts.backtest_low_line_over import Pick


def pick(result, p=0.55, edge=0.02, ev=-0.0185, market='spread', date='2026-08-01', league='NFL'):
    return Pick(date=date, league=league, home='H', away='A', best_pick='H -2.5', market_type=market, line=-2.5,
                eff_win_prob=p, win_prob=p, edge=edge, consensus='', kelly=0.0, win_amount=None, result=result,
                ev=ev)


def test_rows_use_consistent_price_and_market_probability():
    row = sb.from_picks([pick('WIN')])[0]
    assert row.decimal_odds == pytest.approx((1 - 0.0185) / 0.55)
    assert row.market_probability == pytest.approx(0.53)
    assert 0 <= 1 / row.decimal_odds - row.market_probability <= sb.MAX_MARGIN


@pytest.mark.parametrize('ev,edge', [(49.5, 0.02), (-0.0185, -0.30)])
def test_inconsistent_units_get_no_price_or_market_benchmark(ev, edge):
    row = sb.from_picks([pick('WIN', ev=ev, edge=edge)])[0]
    if ev > 1:
        assert row.decimal_odds is None and row.market_probability is None
    else:
        assert row.decimal_odds is not None and row.market_probability is None
    assert row.probability == 0.55 and row.result == 'WIN'  # outcome and model probability are kept


def test_ungraded_rows_are_skipped_and_pushes_are_not_scored():
    rows = sb.from_picks([pick('WIN'), pick(''), pick('PUSH'), pick('LOSS')])
    assert [r.result for r in rows] == ['WIN', 'PUSH', 'LOSS']
    m = sb.metrics(rows)
    assert (m['n'], m['pushes'], m['win_rate']) == (2, 1, 0.5)
    assert m['roi_per_unit'] == pytest.approx(((1 - 0.0185) / 0.55 - 1 - 1 + 0) / 3, abs=1e-4)


def test_log_loss_brier_and_wilson_match_hand_calculation():
    assert sb.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert sb.brier([0.8, 0.2], [1, 0]) == pytest.approx(0.04)
    low, high = sb.wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    assert sb.wilson(0, 0) == (None, None)


def test_skill_is_positive_only_when_the_model_beats_the_market():
    decimal = 1 / 0.57  # fair market 0.55 plus a 0.02 bookmaker margin
    better = [pick('WIN', p=0.70, edge=0.15, ev=0.70 * decimal - 1) for _ in range(30)] + \
             [pick('LOSS', p=0.30, edge=-0.25, ev=0.30 * decimal - 1) for _ in range(30)]
    # the market is fixed at 0.55; the model is sharper and right, so its log loss is lower
    rows = sb.from_picks(better)
    assert all(r.market_probability == pytest.approx(0.55) for r in rows)
    m = sb.metrics(rows)
    assert m['skill_vs_market'] > 0 and m['skill_vs_market_ci'][0] > 0


def test_verdict_requires_a_minimum_sample_and_an_interval():
    assert sb.verdict(dict(n=10)).startswith('TOO FEW GRADED PICKS')
    assert sb.verdict(dict(n=500, skill_vs_market_ci=(0.001, 0.02))).startswith('BEATS MARKET')
    assert sb.verdict(dict(n=500, skill_vs_market_ci=(-0.02, -0.001))).startswith('WORSE THAN MARKET')
    assert sb.verdict(dict(n=500, skill_vs_market_ci=(-0.01, 0.01))).startswith('NO DEMONSTRATED EDGE')
    assert sb.verdict(dict(n=500)) == 'NO MARKET BENCHMARK'


def test_scoreboard_groups_and_markdown():
    rows = sb.from_picks([pick('WIN', league='NFL', market='spread', date='2026-08-01'),
                          pick('LOSS', league='MLB', market='total_over', date='2026-09-02')])
    board = sb.scoreboard(rows)
    assert set(board['by_league']) == {'MLB', 'NFL'} and set(board['by_league_market']) == {'MLB TOTAL', 'NFL SPREAD'}
    assert set(board['by_month']) == {'2026-08', '2026-09'}
    assert board['coverage'] == dict(rows=2, first_date='2026-08-01', last_date='2026-09-02')
    text = sb.markdown(board)
    assert '### By league and market' in text and 'CLV: UNAVAILABLE' in text


def test_metrics_are_deterministic():
    rows = sb.from_picks([pick('WIN' if i % 3 else 'LOSS') for i in range(40)])
    assert sb.metrics(rows) == sb.metrics(rows)


def test_repository_exports_load_and_report(tmp_path):
    from scripts import scoreboard as cli
    out = tmp_path / 'board.md'
    assert cli.main(['--markdown', str(out)]) == 0
    assert 'Graded slates' in out.read_text()
