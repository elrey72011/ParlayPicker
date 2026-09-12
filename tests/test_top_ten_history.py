from copy import deepcopy
from datetime import datetime

import pytest
from app_core.public_history import History, digest, report, selections
from app_core.public_board import validate_package
from app_core.top_ten_history import ranked_picks, top_ten_selections
from test_public_history import Memory, pub, scores


def tracked(count=1):
    publication = pub()
    package = publication['package']
    package.update(selection_policy='qualified-v1', top_ten_policy='first-publication-v1')
    original = package['games']['overall'][0]
    original.update(quote_source='Novig', quote_time=original['as_of'])
    package['games']['overall'] = [dict(original, game=f'Away {i} at Boston', win_estimate=.5+i/100) for i in range(count)]
    if count == 1:
        package['games']['overall'][0]['game'] = 'Seattle at Boston'
    publication['package_hash'] = digest(package)
    return publication


def test_daily_cohort_is_first_confirmed_not_latest_or_first_winners():
    first = tracked(12)
    later = deepcopy(first)
    later['confirmed_at'] = '2026-09-09T19:57:00+00:00'
    for row in later['package']['games']['overall']:
        row.update(pick='Boston -1.5', odds=120, win_estimate=.99)
    later['package_hash'] = digest(later['package'])
    before = deepcopy([later, first])
    cohort = top_ten_selections([later, first, first])
    assert len(cohort) == 10
    assert [r['legs'][0]['game'] for r in cohort] == [f'Away {i} at Boston' for i in range(11, 1, -1)]
    assert all(r['legs'][0]['pick'] == 'Boston +1.5' and r['legs'][0]['odds'] == -110 for r in cohort)
    assert [later, first] == before
    assert len({r['id'] for r in cohort}) == 10


def test_short_cohort_never_refilled_and_next_day_is_separate():
    first = tracked()
    later = tracked(12); later['confirmed_at'] = '2026-09-09T19:57:00+00:00'
    next_day = deepcopy(first)
    next_day['confirmed_at'] = next_day['confirmed_at'].replace('09-09', '09-10')
    for family in next_day['package']['games'].values():
        for row in family:
            for key in ('start', 'as_of', 'quote_time'):
                if key in row: row[key] = row[key].replace('09-09', '09-10')
    result = top_ten_selections([next_day, later, first])
    assert [r['date'] for r in result] == ['2026-09-09', '2026-09-10']
    assert len({r['id'] for r in result}) == 2


def test_only_eligible_confirmed_current_day_quotes_start_tracking():
    first = tracked()
    package = first['package']
    at = datetime.fromisoformat(first['confirmed_at'])
    original = package['games']['overall'][0]
    variants = [dict(original, status='PASS'), dict(original, ev=0), dict(original, win_estimate=None),
                dict(original, quote_source='Unavailable'), dict(original, quote_time='2026-09-09T19:00:00Z'),
                dict(original, start=first['confirmed_at']), dict(original, start='2026-09-10T20:00:00Z')]
    package['games']['overall'] = variants
    assert ranked_picks(package, at) == []
    fresh = tracked();fresh['confirmed_at']='2026-09-09T19:57:00+00:00'
    assert len(top_ten_selections([fresh, first])) == 1
    fresh['confirmed_at']='2026-09-09T20:00:00Z'
    assert top_ten_selections([fresh]) == []
    legacy = tracked();del legacy['package']['top_ten_policy']
    assert top_ten_selections([legacy]) == []


def test_archive_restore_grade_and_score_correction_preserve_top_ten_lines():
    p = tracked();store=History('top-ten-site','folder',Memory())
    key = store.archive(p['package'])
    assert top_ten_selections(store.publications()) == []
    store.confirm('deploy-first',key,p['confirmed_at'])
    restored = History('top-ten-site','folder',store.client).publications()
    assert len([r for r in selections(restored) if r['category']=='top10']) == 1
    revision={'recorded_at':'2026-09-10T01:00:00Z','scores':scores()}
    def top(revisions): return next(r for r in report(restored,revisions) if r['category']=='top10')
    assert top([])['outcome']=='PENDING'
    win=top([revision]);assert win['outcome']=='WIN'
    correction={'recorded_at':'2026-09-10T02:00:00Z','scores':[dict(scores()[0],away_score=8)]}
    loss=top([revision,correction]);assert loss['outcome']=='LOSS'
    assert win['id']==loss['id'] and win['picks']==loss['picks'] and loss['odds']=='-110'
    package=deepcopy(p['package']);package.update(schema_version=5,parlays=[],results=report(restored,[revision]))
    validate_package(package)
    package['results'][-1]['group']='Research'
    with pytest.raises(ValueError,match='Top 10'): validate_package(package)


def test_top_ten_policy_is_explicit_and_validated():
    package=tracked()['package'];validate_package(package)
    package['top_ten_policy']='latest-winners'
    with pytest.raises(ValueError,match='Top 10'):validate_package(package)


def test_eastern_day_and_browser_ranking_match(tmp_path):
    import json, os, shutil, subprocess
    from pathlib import Path
    p=tracked(12)
    p['confirmed_at']='2026-09-10T00:05:00Z'  # Still September 9 in Eastern time.
    rows=p['package']['games']['overall']
    for row in rows:
        row.update(as_of='2026-09-10T00:00:00Z',quote_time='2026-09-10T00:00:00Z',start='2026-09-10T01:00:00Z')
    rows[0].update(sport='NCAAF',quote_source='DraftKings',win_estimate=.9)
    rows[1].update(win_estimate=.9,ev=.2)
    rows[-1]['start']='2026-09-10T18:00:00Z'  # Tomorrow in Eastern time: not this daily list.
    at=datetime.fromisoformat(p['confirmed_at'])
    expected=[r['game'] for r in ranked_picks(p['package'],at)]
    assert expected[:2]==['Away 1 at Boston','Away 0 at Boston']
    assert all(r['date']=='2026-09-09' for r in top_ten_selections([p]))
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    html=Path('publishing/board.html').read_text(encoding='utf-8')
    funcs='\n'.join(line for line in html.splitlines() if line.startswith((
        'function easternDay(', 'function topPicks(', 'function qualifiedPick(',
        'function state(', 'function supportedQuote(')))
    script='const data='+json.dumps(p['package'])+';Date.now=()=>Date.parse('+json.dumps(p['confirmed_at'])+');\n'+funcs
    script+='\nconsole.log(JSON.stringify(topPicks(data.games.overall).map(r=>r.game)));'
    target=tmp_path/'top-ten-parity.cjs';target.write_text(script,encoding='utf-8')
    result=subprocess.run([node,str(target)],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)==expected
