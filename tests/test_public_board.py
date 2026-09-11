import json
import pandas as pd
import pytest
from app_core.public_board import build_package, validate_package
from scripts.publish_board import render, publish, atomic_write


def boards():
    row = {'matchup_id':'game-1','export_run_id':'20260908T204417.080226Z','league':'MLB','matchup':'A at B',
           'pick':'A +1.5','market_type':'spread_away','odds':-110,'Bettable':True,'Play_Stake':5,'status':'APPROVED',
           'win_probability':.6,'ev':.14,'start':'2026-09-08 7:41 PM ET','bankroll':1000,'api_key':'PRIVATE','gemini_explanation':'PRIVATE'}
    return [pd.DataFrame([row]) for _ in range(3)]


def test_allowlist_preserves_status_but_not_private_fields():
    package = build_package(*boards())
    assert package['games']['overall'][0]['status']=='APPROVED'
    assert 'PRIVATE' not in json.dumps(package)
    assert 'Play_Stake' not in json.dumps(package)
    assert package['games']['overall'][0]['start']=='2026-09-08T23:41:00+00:00'


def test_mixed_runs_and_duplicate_games_rejected():
    frames = boards();frames[1]['export_run_id']='20260909T200000Z'
    with pytest.raises(ValueError): build_package(*frames)
    frames=boards();frames[0]=pd.concat([frames[0],frames[0]])
    with pytest.raises(ValueError): build_package(*frames)


def test_false_approval_and_missing_metrics_stay_pass():
    frames=boards();frames[0]['Bettable']='False'
    frames[1]['win_probability']=None
    package=build_package(*frames)
    assert package['games']['overall'][0]['status']=='PASS'
    assert package['games']['sides'][0]['status']=='PASS'


def test_html_escapes_script_injection_and_rejects_extra_fields():
    frames=boards();frames[0]['pick']='</script><script>alert(1)</script>'
    package=build_package(*frames)
    html=render(package)
    assert '</script><script>alert(1)' not in html
    assert '\\u003c/script' in html
    package['secret']='PRIVATE'
    with pytest.raises(ValueError):render(package)


def test_publish_rejects_invalid_draft_and_preserves_previous(tmp_path):
    draft=tmp_path/'draft';draft.mkdir();dest=tmp_path/'site'
    package=build_package(*boards())
    (draft/'public-board.json').write_text(json.dumps(package))
    target=publish(draft,dest);before=target.read_text()
    package['games']['overall'][0]['pick']='new selection'
    (draft/'public-board.json').write_text(json.dumps(package))
    publish(draft,dest)
    assert (dest/'previous.html').read_text()==before
    current=target.read_text();package['private']='secret'
    (draft/'public-board.json').write_text(json.dumps(package))
    with pytest.raises(ValueError):publish(draft,dest)
    assert target.read_text()==current


def test_optional_props_require_time_and_keep_private_fields_out():
    props=pd.DataFrame([{'player':'Player','matchup':'A at B','best_pick':'Over 1.5','Bettable':False,'bankroll':200}])
    with pytest.raises(ValueError):build_package(*boards(),props=props)
    package=build_package(*boards(),props=props,props_as_of='2026-09-08T20:00:00Z')
    assert package['props'][0]['player']=='Player'
    assert package['props'][0]['status']=='PASS'
    assert 'bankroll' not in json.dumps(package)


def test_dfs_requires_complete_unique_roster_and_slate():
    from app_core.draftkings_classic import DK_NFL_CLASSIC_ROSTER_SLOTS as slots
    df=pd.DataFrame([{**{slot:f'Player {i} ({i})' for i,slot in enumerate(slots)},'Salary':49000,'Projected Points':120}])
    with pytest.raises(ValueError): build_package(*boards(),dfs=df)
    kwargs=dict(dfs=df,dfs_sport='NFL',dfs_slate='Sunday Classic',dfs_start='2026-09-13T17:00:00Z')
    package=build_package(*boards(),**kwargs)
    assert validate_package(package)['dfs'][0]['salary_remaining']==1000
    df['RB1']=df['QB']
    with pytest.raises(ValueError):build_package(*boards(),**kwargs)


def test_navigation_survives_cached_legacy_renderer():
    from scripts.publish_board import ROOT
    from app_core.public_site_shell import header, STYLES
    template = (ROOT / 'publishing/board.html').read_text(encoding='utf-8')
    # A running Streamlit worker can retain the renderer from before navigation.
    legacy_html = template.replace('__PUBLIC_DATA__', json.dumps(build_package(*boards())))
    assert '__SITE_' not in legacy_html
    assert header(board=True) in legacy_html
    assert STYLES in legacy_html
    assert 'href="/how-picks-work/"' in legacy_html


def test_locked_selection_render_keeps_original_record(tmp_path):
    import os
    import shutil
    import subprocess
    from scripts.publish_board import ROOT
    node = os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser rendering regression')
    template = (ROOT / 'publishing/board.html').read_text(encoding='utf-8')
    functions = template.split('function lockedRows(', 1)[1].split('function renderParlays(', 1)[0]
    script = r"""
const assert=require('node:assert/strict');
class Element {constructor(tag,text=''){this.tag=tag;this.textContent=text;this.children=[];} append(...v){this.children.push(...v)} replaceChildren(){this.children=[]} addEventListener(){} }
const el=(tag,text)=>new Element(tag,text);
const root=new Element('div');const document={getElementById:()=>root};
const today=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York'}).format(new Date());
const original={group:'Locked',category:'overall',date:today,picks:'A at B: Over 65.5',odds:'-115',published_at:new Date().toISOString(),outcome:'PENDING'};
const availableResults=[original,{...original,group:'Research',picks:'A at B: Under 63.5'},{...original,date:'2000-01-01'}];
""" + 'function lockedRows(' + functions + r"""
renderLockedPicks();
const content=JSON.stringify(root);
assert.ok(content.includes('Over 65.5'));
assert.ok(content.includes('-115'));
assert.ok(content.includes('LOCKED'));
assert.ok(!content.includes('Under 63.5'));
assert.equal(lockedRows(availableResults,today).length,1);
assert.equal(original.outcome,'PENDING');
availableResults.length=0;renderLockedPicks();assert.ok(JSON.stringify(root).includes('No locked picks for today'));
"""
    target=tmp_path/'locked-render.cjs'
    target.write_text(script,encoding='utf-8')
    subprocess.run([node,str(target)],check=True,capture_output=True,text=True)


def test_public_prop_projection_is_model_count_and_validated():
    from app_core.public_board import pick_record
    row = dict(boards()[0].iloc[0], player='Pitcher', best_pick='Pitcher Over 5.5 Ks',
               expected_count=6.2, odds_american=-110)
    prop = pick_record(row, prop=True)
    assert prop['expected_stat'] == 6.2
    package = build_package(*boards()); package['props'] = [prop]
    validate_package(package)
    for invalid in (-1, float('nan'), True, '6.2'):
        prop['expected_stat'] = invalid
        with pytest.raises(ValueError): validate_package(package)
    del prop['expected_stat']
    validate_package(package)  # Existing saved packages remain valid.
    row.update(league='NFL', FormSampleSize=0)
    assert 'expected_stat' not in pick_record(row, prop=True)
    row['FormSampleSize'] = 5
    assert pick_record(row, prop=True)['expected_stat'] == 6.2


def test_locked_rate_excludes_other_groups_pending_and_pushes():
    import os, shutil, subprocess
    from pathlib import Path
    node = os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    html = Path('publishing/board.html').read_text(encoding='utf-8')
    function = html.split('function renderLockedWinRate(')[1].split('function renderResults(')[0]
    script = """
const assert=require('node:assert/strict');
let output=[];
const document={getElementById:()=>({replaceChildren:(...items)=>output=items})};
const el=(tag,text)=>text;
const fmt=value=>(value*100).toFixed(1)+'%';
const availableResults=['WIN','WIN','LOSS','PUSH','PENDING'].map(outcome=>({group:'Locked',category:'overall',date:'2026-09-11',outcome}));
availableResults.push({group:'Research',category:'overall',date:'2026-09-11',outcome:'WIN'});
""" + 'function renderLockedWinRate(' + function + """
renderLockedWinRate(null); assert.equal(output[1],'66.7%');
assert.match(output[2],/2 wins · 1 losses · 1 pushes · 1 pending/);
renderLockedWinRate(['2026-09-12','2026-09-12']);assert.equal(output[1],'Awaiting settled picks');
"""
    subprocess.run([node, '-e', script], check=True)
