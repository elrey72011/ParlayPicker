from datetime import datetime, timezone, timedelta
import pytest
from test_public_history import Memory, pub, scores
from app_core.public_history import History
from app_core import public_grading_scheduler as scheduler

AT=datetime(2026,9,10,0,tzinfo=timezone.utc)

@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(scheduler,'is_open',lambda:True)
    client=Memory();store=History('site-1234','folder',client)
    p=pub();key=store.archive(p['package']);store.confirm('deploy-123',key,p['confirmed_at'])
    return client,store


def test_grades_pending_once_and_restores_status(setup):
    client,store=setup;calls=[]
    def fetch(day,sports):
        calls.append((day,sports));return {'recorded_at':AT.isoformat(),'scores':scores()}
    r=scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT,fetch=fetch)
    assert r['newly_settled']==3 and r['pending']==0 and len(calls)==1
    r2=scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT+timedelta(hours=2),fetch=fetch)
    assert r2['checked_batches']==0 and len(calls)==1
    assert store.all('grading_runs') and len(store.all('scores'))==1


def test_unfinished_retry_throttled_and_secrets_sanitized(setup):
    client,store=setup;calls=[]
    def fail(*a):calls.append(1);raise RuntimeError('private-token')
    r=scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT,fetch=fail)
    assert r['status']=='error' and r['pending']==3 and 'private-token' not in str(r)
    scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT+timedelta(minutes=30),fetch=fail)
    assert len(calls)==1
    scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT+timedelta(hours=1),fetch=fail)
    assert len(calls)==2


def test_early_and_off_hours_do_not_fetch(setup,monkeypatch):
    client,_=setup
    def fail(*a):raise AssertionError('must not fetch')
    assert scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT-timedelta(hours=3),fetch=fail)['checked_batches']==0
    monkeypatch.setattr(scheduler,'is_open',lambda:False)
    assert scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT,fetch=fail)['status']=='outside_operating_window'


def test_restore_failure_stops_provider_requests(monkeypatch):
    monkeypatch.setattr(scheduler,'is_open',lambda:True)
    class Broken(Memory):
        def paginate(self,**kw):raise RuntimeError('restore failed')
    with pytest.raises(RuntimeError):
        scheduler.run('site-1234','folder',Broken(),{'MLB'},clock=lambda:AT,fetch=lambda *a:pytest.fail('fetched after restore failure'))


def test_batch_limit_and_old_records(setup):
    from copy import deepcopy
    client,store=setup
    for i in range(1,5):
        p=deepcopy(pub());day=9-i
        for rows in p['package']['games'].values():
            for leg in rows:
                leg['start']=leg['start'].replace('09-09',f'09-{day:02d}')
                leg['as_of']=leg['as_of'].replace('09-09',f'09-{day:02d}')
        key=store.archive(p['package']);store.confirm('older-deploy-'+str(i),key,p['confirmed_at'].replace('09-09',f'09-{day:02d}'))
    calls=[]
    r=scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT,fetch=lambda *a:calls.append(a) or {'recorded_at':AT.isoformat(),'scores':[]})
    assert len(calls)==2 and r['checked_batches']==2
    calls.clear()
    scheduler.run('site-1234','folder',client,{'MLB'},clock=lambda:AT+timedelta(days=40),fetch=lambda *a:calls.append(a))
    assert not calls


@pytest.mark.parametrize('configured',[False,True])
def test_cli_optional_public_grading(monkeypatch,tmp_path,configured):
    from scripts import run_research_scheduler as cli
    monkeypatch.setattr(cli,'is_open',lambda:True)
    monkeypatch.setattr(cli,'settings',lambda:('folder',None))
    monkeypatch.setattr(cli,'DriveStore',lambda folder:object())
    monkeypatch.setattr(cli,'run',lambda *a:{'errors':[]})
    monkeypatch.setenv('GITHUB_STEP_SUMMARY',str(tmp_path/'summary.md'))
    monkeypatch.setenv('RESEARCH_SPORTS','MLB')
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID','folder')
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT','test-only-placeholder')
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','site-1234' if configured else '')
    calls=[]
    monkeypatch.setattr(scheduler,'run',lambda *a:calls.append(a) or {'status':'ok','errors':[]})
    assert cli.main()==0
    assert len(calls)==int(configured)
    if configured:
        from app_core.espn_results import ESPN_ENDPOINTS
        assert calls[0][3] == set(ESPN_ENDPOINTS)
        assert {'NFL', 'WNBA'} <= calls[0][3]
    assert ('not_configured' in (tmp_path/'summary.md').read_text()) != configured


def test_cli_public_failure_is_sanitized(monkeypatch,tmp_path):
    from scripts import run_research_scheduler as cli
    monkeypatch.setattr(cli,'is_open',lambda:True)
    monkeypatch.setattr(cli,'settings',lambda:('folder',None))
    monkeypatch.setattr(cli,'DriveStore',lambda folder:object())
    monkeypatch.setattr(cli,'run',lambda *a:{'errors':['research:failure']})
    monkeypatch.setenv('GITHUB_STEP_SUMMARY',str(tmp_path/'summary.md'))
    monkeypatch.setenv('RESEARCH_SPORTS','MLB')
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID','folder')
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT','test-only-placeholder')
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','site-1234')
    def fail(*a):raise RuntimeError('secret-value')
    monkeypatch.setattr(scheduler,'run',fail)
    assert cli.main()==1
    summary=(tmp_path/'summary.md').read_text()
    assert 'public_grading:RuntimeError' in summary and 'secret-value' not in summary


@pytest.mark.parametrize('sport,away,home,provider_away,provider_home', [
    ('NFL', 'Cleveland', 'Tampa Bay', 'Cleveland Browns', 'Tampa Bay Buccaneers'),
    ('WNBA', 'Minnesota', 'Connecticut', 'Minnesota Lynx', 'Connecticut Sun'),
])
def test_public_saved_sports_settle_with_fresh_finals(monkeypatch, sport, away, home, provider_away, provider_home):
    from app_core.espn_results import ESPN_ENDPOINTS
    monkeypatch.setattr(scheduler, 'is_open', lambda: True)
    client = Memory()
    store = History('site-1234', 'folder', client)
    publication = pub()
    for family, legs in publication['package']['games'].items():
        for leg in legs:
            leg.update(sport=sport, game=f'{away} at {home}')
            if family != 'totals':
                leg['pick'] = home + ' +1.5'
    key = store.archive(publication['package'])
    store.confirm('deploy-sport', key, publication['confirmed_at'])
    final = dict(sport=sport, away=provider_away, home=provider_home,
                 event_id='verified-event', result_source='ESPN', completed=True,
                 start='2026-09-09T20:00:00+00:00', away_score=4, home_score=3)
    calls = []
    def fetch(day, sports):
        calls.append(sports)
        return {'recorded_at': AT.isoformat(), 'scores': [final]}
    result = scheduler.run('site-1234', 'folder', client, set(ESPN_ENDPOINTS),
                           clock=lambda: AT, fetch=fetch)
    assert calls == [{sport}]
    assert result['newly_settled'] == 3
    assert result['pending'] == 0
