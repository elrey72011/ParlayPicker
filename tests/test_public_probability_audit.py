from copy import deepcopy
from app_core.public_probability_audit import build


def row(i,p,y, category='overall'):
    return dict(id=str(i),date='2026-09-20',published_at='2026-09-20T12:00:00Z',group='Research',category=category,outcome=y,
                legs=[dict(league='NFL',game=str(i),start='2026-09-20T17:00:00Z',selection='Under 40.5',odds=-110,market_type='total_under',original_win_estimate=p,outcome=y)])


def test_reverse_ranking_and_duplicate_exports():
    rows=[row(1,.7,'LOSS'),row(2,.55,'WIN')]
    a=build([{'records':rows},{'records':rows}])
    c=a['cohorts'][0]
    assert c['n']==2 and c['ranking_concordance']==0
    assert c['actual_win_rate']==.5 and c['mean_estimate']==.625
    assert len(c['probability_bands'])==2


def test_missing_probability_postgame_and_conflict_fail_closed():
    a=row(1,None,'WIN'); b=row(2,.6,'WIN');b['published_at']='2026-09-20T18:00:00Z'
    c=row(3,.6,'WIN');d=deepcopy(c);d['legs'][0]['original_win_estimate']=.7
    report=build([{'records':[a,b,c,d]}])
    assert not report['cohorts']
    assert sum(report['exclusions'].values())==3


def test_categories_and_locked_are_not_pooled():
    a=row(1,.6,'WIN');b=row(2,.6,'WIN','totals');c=row(3,.6,'WIN');c['group']='Locked'
    assert len(build([{'records':[a,b,c]}])['cohorts'])==3


def test_parlay_pairs_dedup_and_repeated_leg_exposure():
    a=row(1,.6,'WIN','parlays');a['legs']+=row(2,.7,'LOSS')['legs'];a['outcome']='LOSS'
    b=deepcopy(a);b['id']='new-ticket'
    r=build([{'records':[a,b]}])
    assert r['parlay_dependence'][0]['unique_pairs']==1
    assert r['parlay_dependence'][0]['both_win_rate']==0
    assert len(r['repeated_leg_exposure'])==2
    assert r['parlay_dependence'][0]['pooled_outcome_phi'] is None


def test_reconciliation_preserves_original_estimate_and_start():
    from app_core.public_reconciliation import selection_facts
    from test_public_history import leg
    l=leg()
    result=selection_facts({'legs':[l]})['legs'][0]
    assert result['original_win_estimate']==l['win_estimate']
    assert result['start']==l['start']


def test_published_lock_changes_preserve_original_values():
    a=row(1,.6,'WIN');b=deepcopy(a);b.update(id='lock',group='Locked',outcome='LOSS')
    b['legs'][0].update(selection='Over 40.5',market_type='total_over',outcome='LOSS',original_win_estimate=.7)
    changes=build([{'records':[a,b]}])['selection_changes']
    assert len(changes)==1
    assert changes[0]['published']['original_win_estimate']==.6
    assert changes[0]['locked']['outcome']=='LOSS'
