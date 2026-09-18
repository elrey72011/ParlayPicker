from copy import deepcopy
import pytest
from app_core.public_history import grade_leg, grading_team_name


@pytest.mark.parametrize('away,home,away_full,home_full', [
    ('Las Vegas', 'Seattle', 'Las Vegas Aces', 'Seattle Storm'),
    ('Phoenix', 'Portland', 'Phoenix Mercury', 'Portland Fire'),
    ('Washington', 'Chicago', 'Washington Mystics', 'Chicago Sky'),
    ('Connecticut', 'Atlanta', 'Connecticut Sun', 'Atlanta Dream'),
    ('Los Angeles', 'Dallas', 'Los Angeles Sparks', 'Dallas Wings'),
])
def test_saved_wnba_city_names_grade_without_mutation(away, home, away_full, home_full):
    leg = dict(sport='WNBA', game=f'{away} at {home}', start='2026-09-18T00:00:00Z',
               market='spread_home', pick=f'{home} +1.5')
    score = dict(sport='WNBA', away=away_full, home=home_full, start=leg['start'],
                 event_id='1', result_source='ESPN', completed=True, away_score=80, home_score=85)
    original = deepcopy(leg)
    assert grade_leg(leg, [score])[0] == 'WIN'
    assert leg == original
    assert grade_leg(dict(leg, market='spread_away', pick=f'{away} +1.5'), [score])[0] == 'LOSS'
    assert grade_leg(dict(leg, market='total_over', pick='Over 165'), [score])[0] == 'PUSH'
    assert grade_leg(leg, [score, dict(score, event_id='2')])[0] == 'PENDING'
    assert grade_leg(leg, [dict(score, completed=False)])[0] == 'PENDING'
    assert grade_leg(dict(leg, espn_event_id='other'), [score])[0] == 'PENDING'
    assert grade_leg(leg, [dict(score, sport='NBA')])[0] == 'PENDING'
    assert grade_leg(leg, [dict(score, home=away_full, away=home_full)])[0] == 'PENDING'


@pytest.mark.parametrize('wrong,right', [('UConn','Connecticut Sun'), ('Dallas Stars','Dallas Wings'),
                                        ('Washington Capitals','Washington Mystics'), ('Seattle U','Seattle Storm')])
def test_wnba_does_not_alias_other_sports(wrong, right):
    assert grading_team_name(wrong, 'WNBA') != grading_team_name(right, 'WNBA')
