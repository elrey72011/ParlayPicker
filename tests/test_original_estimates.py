from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import pytest
from app_core.public_history import report, original_estimate
from app_core.public_board import validate_package
from test_public_history import pub, scores
from test_public_prop_history import publication
from app_core import public_prop_history


def test_first_probability_survives_later_publication_and_score_correction():
    first=pub(); later=deepcopy(first)
    later['confirmed_at']='2026-09-09T19:58:00+00:00'
    for rows in later['package']['games'].values(): rows[0]['win_estimate']=.9
    revisions=[{'recorded_at':'2026-09-10T00:00:00Z','scores':scores()},
               {'recorded_at':'2026-09-10T01:00:00Z','scores':[{**scores()[0],'away_score':8}]}]
    rows=report([later,first],revisions)
    assert all(r['original_win_estimate']==.6 for r in rows)
    assert rows[0]['outcome']=='LOSS'
    assert {(r['sport'],r['market']) for r in rows}=={('MLB','spread_home'),('MLB','total_under')}
    first['package']['games']['overall'][0].pop('win_estimate')
    assert 'original_win_estimate' not in report([first,later],[])[0]


def test_prop_original_estimate_excludes_import_and_preserves_projection():
    first=publication(); first['package']['props'][0]['expected_stat']=2.1
    later=deepcopy(first); later['confirmed_at']='2026-09-09T19:58:00+00:00'
    later['package']['props'][0]['win_estimate']=.9
    leg=first['package']['props'][0]
    rows=public_prop_history.report([later,first],imports=[{'id':'import','as_of':leg['as_of'],'props':[leg]}])
    saved=next(r for r in rows if r['group']=='Approved')
    imported=next(r for r in rows if r['group']=='Imported research')
    assert saved['original_win_estimate']==.6
    assert saved['expected_stat']==2.1
    assert 'original_win_estimate' not in imported


@pytest.mark.parametrize('invalid',[None,True,'0.6',-0.1,1.1,float('nan'),float('inf')])
def test_invalid_original_probabilities_are_not_published(invalid):
    assert original_estimate({'win_estimate':invalid})=={}
    p=pub()['package']; p.update(schema_version=3,parlays=[],results=report([pub()],[]))
    p['results'][0]['original_win_estimate']=invalid
    with pytest.raises(ValueError,match='original win estimate'): validate_package(p)


def test_legacy_results_and_probability_endpoints_remain_valid():
    p=pub()['package'];p.update(schema_version=3,parlays=[],results=report([pub()],[]))
    for value in (0,1):
        p['results'][0]['original_win_estimate']=value
        validate_package(p)
    for row in p['results']:
        for key in ('original_win_estimate','sport','market'):row.pop(key,None)
    validate_package(p)


def test_browser_estimate_bands_and_coverage(tmp_path):
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    html=Path('publishing/board.html').read_text(encoding='utf-8')
    functions='\n'.join(line for line in html.splitlines() if line.startswith(('function originalEstimateLabel(', 'function estimateComparison(', 'function renderEstimateComparison(')))
    script="""
const assert=require('node:assert/strict');
const fmt=p=>(p*100).toFixed(1)+'%';
class E {constructor(tag,text=''){this.textContent=text;this.children=[];} append(...v){this.children.push(...v)} replaceChildren(...v){this.children=v} }
const root=new E('div');const document={getElementById:()=>root};const el=(t,s)=>new E(t,s);
"""+functions+"""
const base={category:'overall',sport:'NFL',market:'spread_home',group:'Locked',outcome:'WIN',original_win_estimate:.60};
const rows=[base,{...base,market:'spread_away',outcome:'LOSS',original_win_estimate:.64},
 {...base,original_win_estimate:undefined},{...base,outcome:'PUSH'},{...base,outcome:'PENDING'},
 {...base,category:'sides'},{...base,sport:'NCAAF'},{...base,market:'total_over'},
 {...base,category:'parlays'},{...base,group:'Imported research'}];
const before=JSON.stringify(rows);const groups=estimateComparison(rows);
assert.equal(groups.length,4);
const g=groups.find(g=>g.category==='overall'&&g.sport==='NFL'&&g.market==='spread');
assert.equal(g.settled,3);assert.equal(g.missing,1);assert.equal(g.excluded,2);
const b=g.bands.get(12);assert.equal(b.n,2);assert.equal(b.wins,1);assert.equal(b.sum/2,.62);
assert.equal(originalEstimateLabel({original_win_estimate:null}),'Not recorded');
assert.equal(originalEstimateLabel({original_win_estimate:0}),'0.0%');
const endpoints=estimateComparison([0,.55,1].map(p=>({...base,original_win_estimate:p})))[0];
assert.deepEqual([...endpoints.bands.keys()],[0,11,19]);
renderEstimateComparison(rows);const text=JSON.stringify(root);
assert.ok(text.includes('62.0%')&&text.includes('50.0%')&&text.includes('-12.0 percentage points'));
assert.ok(text.includes('2 of 3 decided picks'));
assert.equal(JSON.stringify(rows),before);
"""
    target=tmp_path/'estimate-comparison.cjs';target.write_text(script,encoding='utf-8')
    subprocess.run([node,str(target)],check=True,capture_output=True,text=True)
