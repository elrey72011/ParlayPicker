"""Owner decision D1: qualification thresholds (roadmap §1)."""
from app_core.selection import qualification as q

GOOD = dict(n=200, skill_vs_market_ci=(0.002, 0.02))
GOOD_CLV = dict(n=180, mean_line_clv_ci=(0.05, 0.4))


def test_all_three_conditions_are_required():
    board = {'by_league_market': {'NFL SPREAD': GOOD, 'NBA TOTAL': GOOD, 'MLB TOTAL': dict(n=700, skill_vs_market_ci=(-0.01, 0.01))}}
    clv = {'by_league_market': {'NFL SPREAD': GOOD_CLV, 'MLB TOTAL': GOOD_CLV}}
    a = q.assess(board, clv)
    assert a['NFL SPREAD'] == dict(qualified=True, reasons=[])
    assert a['NBA TOTAL']['reasons'] == ['NO_CLOSING_LINE_EVIDENCE']
    assert a['MLB TOTAL']['reasons'] == ['SKILL_VS_MARKET_NOT_ABOVE_ZERO']
    assert q.qualified_set(a) == {('NFL', 'SPREAD')}


def test_sample_size_and_clv_failures_are_specific():
    board = {'by_league_market': {'NFL TOTAL': dict(n=37, skill_vs_market_ci=(0.001, 0.02))}}
    clv = {'by_league_market': {'NFL TOTAL': dict(n=20, mean_line_clv_ci=(-0.1, 0.3))}}
    assert q.assess(board, clv)['NFL TOTAL']['reasons'] == [
        'GRADED_PICKS_BELOW_150:37', 'CLOSES_BELOW_150:20', 'CLV_NOT_ABOVE_ZERO']


def test_repository_baseline_qualifies_nothing():
    import contextlib
    import io
    from app_core.evaluation import scoreboard
    with contextlib.redirect_stdout(io.StringIO()):
        board = scoreboard.scoreboard(scoreboard.load('data/backtest_exports'))
    assessment = q.assess(board)
    assert assessment and q.qualified_set(assessment) == frozenset()
    assert all('NO_CLOSING_LINE_EVIDENCE' in a['reasons'] for a in assessment.values())
