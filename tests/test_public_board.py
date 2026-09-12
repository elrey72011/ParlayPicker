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
const before=JSON.stringify(original);
original.sport='MLB';
availableResults.push({...original,sport:'NCAAF',picks:'College at Team: Over 45.5'});
root.value='MLB';renderLockedPicks();
assert.ok(JSON.stringify(root).includes('Over 65.5'));
assert.ok(!JSON.stringify(root).includes('Over 45.5'));
assert.ok(JSON.stringify(root).includes("Today's locked picks · 1"));
root.value='NCAAF';renderLockedPicks();
assert.ok(JSON.stringify(root).includes('Over 45.5'));
assert.ok(!JSON.stringify(root).includes('Over 65.5'));
root.value='NFL';renderLockedPicks();
assert.ok(JSON.stringify(root).includes('No locked picks for NFL today'));
root.value='';renderLockedPicks();
assert.ok(JSON.stringify(root).includes("Today's locked picks · 2"));
delete original.sport;
assert.equal(JSON.stringify(original),before);
assert.equal(lockedRows(availableResults,today,'MLB').length,0); // No guessing for legacy rows.

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


def test_preview_navigation_prevents_streamlit_reload():
    import os, shutil, subprocess
    from pathlib import Path
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    html=Path('publishing/board.html').read_text(encoding='utf-8')
    handler=html.split("document.addEventListener('click',event=>")[1].split(";\nwindow.addEventListener('hashchange'")[0]
    script="""
const assert=require('node:assert/strict');
const pages={games:1,props:1,parlays:1,results:1,dfs:1};
let selected,opened,prevented;
const selectPage=value=>selected=value;
const window={self:{},top:{},open:(...args)=>opened=args};
const click=event=>"""+handler[:-1]+""";
for(const name of Object.keys(pages)){
 prevented=false;const link={getAttribute:()=> '#'+name,classList:{contains:()=>false}};
 click({target:{closest:()=>link},preventDefault:()=>prevented=true});
 assert.equal(selected,name);assert.equal(prevented,true);
}
click({target:{closest:()=>({getAttribute:()=>'/how-picks-work/',classList:{contains:()=>false}})},preventDefault:()=>{}});
assert.equal(opened[0],'https://picks.cmsvconsulting.com/how-picks-work/');
assert.equal(opened[1],'_blank');
"""
    subprocess.run([node,'-e',script],check=True)


def test_browser_college_fallback_label_and_freshness():
    import os, shutil, subprocess
    from pathlib import Path
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    html=Path('publishing/board.html').read_text(encoding='utf-8')
    funcs='\n'.join(line for line in html.splitlines() if line.startswith(('function supportedQuote(', 'function state(')))
    script="""
const assert=require('node:assert/strict');
const data={stale_after_minutes:15};
const now=Date.now();
"""+funcs+"""
const row={sport:'NCAAF',quote_source:'FanDuel',quote_time:new Date(now-60000).toISOString(),as_of:new Date(now-30000).toISOString(),start:new Date(now+3600000).toISOString(),status:'PASS'};
assert.equal(state(row),'PASS');
assert.equal(state({...row,sport:'MLB'}),'UNAVAILABLE');
assert.equal(state({...row,quote_source:'Unknown'}),'UNAVAILABLE');
assert.equal(state({...row,quote_time:new Date(now-16*60000).toISOString()}),'STALE');
assert.equal(state({...row,start:new Date(now-1000).toISOString()}),'STARTED');
Date.now=()=>now;
data.stale_after_minutes=30;
assert.equal(state({...row,quote_time:new Date(now-20*60000).toISOString()}),'PASS');
assert.equal(state({...row,quote_time:new Date(now-30*60000).toISOString()}),'PASS');
assert.equal(state({...row,quote_time:new Date(now-30*60000-1).toISOString()}),'STALE');
assert.equal(state({...row,as_of:new Date(now-30*60000-1).toISOString()}),'STALE');
"""
    subprocess.run([node,'-e',script],check=True)
    assert "supportedQuote(r)?r.quote_source+' · '" in html
    assert "r.quote_source||'Not recorded'" in html


def test_published_category_overview_keeps_locked_record_separate(tmp_path):
    import os, shutil, subprocess
    from pathlib import Path
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    html=Path('publishing/board.html').read_text(encoding='utf-8')
    funcs='\n'.join(line for line in html.splitlines() if line.startswith(('function matchesResultGroup(', 'function renderLockedWinRate(', 'function flatStakeMetrics(', 'function renderResults(')))
    script=r"""
const assert=require('node:assert/strict');
class Element {constructor(tag,text=''){this.tag=tag;this.textContent=text;this.children=[];this.value='';} append(...v){this.children.push(...v)} replaceChildren(...v){this.children=[...v]} }
const elements={};const document={getElementById:id=>elements[id]||(elements[id]=new Element('div'))};
const el=(tag,text)=>new Element(tag,text);const fmt=(x,p=false)=>p?(x*100).toFixed(1)+'%':String(x);
const availableResults=[
 {group:'Research',category:'overall',outcome:'WIN',picks:'published-overall'},
 {group:'Approved',category:'sides',outcome:'LOSS',picks:'published-side'},
 {group:'Research',category:'totals',outcome:'WIN',picks:'published-total'},
 {group:'Research',category:'parlays',outcome:'LOSS',picks:'published-parlay'},
 {group:'Locked',category:'overall',outcome:'LOSS',picks:'locked-overall'},
 {group:'Imported research',category:'totals',outcome:'WIN',picks:'imported-total'}
].map(r=>({...r,date:'2026-09-11',odds:'-110',final_score:'1-0'}));
const resultWindow=()=>({bounds:['2026-09-11','2026-09-11'],rows:availableResults});
document.getElementById('resultKind').value='games';
document.getElementById('resultGroup').value='Published picks';
"""+funcs+r"""
renderResults();
const summary=document.getElementById('resultSummary').children.find(n=>n.className==='panel scroll').children[0];
assert.deepEqual(summary.children.slice(1).map(r=>r.children.slice(0,7).map(c=>c.textContent)),[
 ['Overall Best Picks',1,0,0,0,1,'100.0%'],['Sides',0,1,0,0,1,'0.0%'],
 ['Totals',1,0,0,0,1,'100.0%'],['Parlays',0,1,0,0,1,'0.0%'],['Top 10',0,0,0,0,0,'No tracked picks']]);
assert.equal(document.getElementById('lockedWinSummary').children[1].textContent,'0.0%');
const details=JSON.stringify(document.getElementById('resultDetails'));
assert.ok(details.includes('published-parlay'));
assert.ok(!details.includes('locked-overall')&&!details.includes('imported-total'));
availableResults.push(...['WIN','LOSS','PUSH','PENDING'].map(outcome=>({group:'Approved',category:'top10',outcome,picks:'top-ten-'+outcome,date:'2026-09-11',odds:'-110',final_score:'saved score'})));
document.getElementById('resultGroup').value='Top 10';document.getElementById('resultPeriod').value='1';renderResults();
const topSummary=document.getElementById('resultSummary').children.find(n=>n.className==='panel scroll').children[0];
assert.deepEqual(topSummary.children[1].children.slice(0,7).map(c=>c.textContent),['Yesterday’s Top 10',1,1,1,1,3,'50.0%']);
assert.equal(topSummary.children.length,2);
assert.ok(JSON.stringify(document.getElementById('resultDetails')).includes('top-ten-WIN'));
assert.ok(!JSON.stringify(document.getElementById('resultDetails')).includes('published-overall'));
document.getElementById('resultGroup').value='Locked';renderResults();
assert.ok(JSON.stringify(document.getElementById('resultDetails')).includes('locked-overall'));
assert.ok(!JSON.stringify(document.getElementById('resultDetails')).includes('published-overall'));
"""
    target=tmp_path/'category-results.cjs';target.write_text(script,encoding='utf-8')
    subprocess.run([node,str(target)],check=True,capture_output=True,text=True)
    assert "const preferred=['Published picks','Locked'" in html


def test_flat_stake_returns_and_qualified_browser_view(tmp_path):
    import os, shutil, subprocess
    from pathlib import Path
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node: pytest.skip('Node unavailable')
    html=Path('publishing/board.html').read_text(encoding='utf-8')
    funcs='\n'.join(line for line in html.splitlines() if line.startswith(('function flatStakeMetrics(', 'function qualifiedPick(')))
    script=r"""
const assert=require('node:assert/strict');
const data={selection_policy:'qualified-v1'};
const state=r=>r.status;const supportedQuote=r=>r.quote_source==='Novig';
"""+funcs+r"""
const q={status:'APPROVED',quote_source:'Novig',ev:.1};
assert.equal(qualifiedPick(q),true);assert.equal(qualifiedPick({...q,status:'PASS'}),false);
assert.equal(qualifiedPick({...q,ev:-.1}),false);delete data.selection_policy;assert.equal(qualifiedPick(q),false);
const leg=(outcome,odds,category='overall')=>({outcome,odds,category});
const result=flatStakeMetrics([leg('WIN','-150'),leg('LOSS','+120'),leg('PUSH','-110'),
 leg('PENDING','-110'),leg('WIN','-110 / -120','parlays'),leg('WIN','Unavailable')]);
assert.equal(result.priced,3);assert.ok(Math.abs(result.net+1/3)<1e-10);
assert.ok(Math.abs(result.roi+1/9)<1e-10);
assert.equal(flatStakeMetrics([leg('WIN','+200')]).net,2);
assert.equal(flatStakeMetrics([leg('LOSS','-110','parlays')]).roi,null);
"""
    target=tmp_path/'qualification-returns.cjs';target.write_text(script,encoding='utf-8')
    subprocess.run([node,str(target)],check=True)
    assert 'No qualifying picks today' in html and 'Full research board' in html


def test_top_ten_cross_league_ranking_and_rendering(tmp_path):
    import os, shutil, subprocess
    from pathlib import Path
    node = os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    html = Path('publishing/board.html').read_text(encoding='utf-8')
    funcs = '\n'.join(line for line in html.splitlines() if line.startswith((
        'function supportedQuote(', 'function state(', 'function qualifiedPick(',
        'function easternDay(', 'function topPicks(', 'function renderTopPicks(', 'function table(')))
    script = r"""
const assert=require('node:assert/strict');
let now=Date.parse('2026-09-12T16:00:00Z');Date.now=()=>now;
const data={selection_policy:'qualified-v1',stale_after_minutes:30,games:{overall:[]}};
const el=(tag,text)=>({tag,text,children:[],append(...nodes){this.children.push(...nodes)},replaceChildren(){this.children=[]}});
const fmt=(x,percent=false)=>percent?(x*100).toFixed(1)+'%':String(x);
const host=el('div');let selectedLeague='MLB';
const document={getElementById(id){if(id==='topPicks')return host;if(id==='gameLeague')return {value:selectedLeague};throw Error(id)}};
""" + funcs + r"""
const row=(game,p,ev=.1,sport='MLB')=>({game,pick:'Saved selection',win_estimate:p,ev,sport,
 status:'APPROVED',quote_source:sport==='NCAAF'?'DraftKings':'Novig',odds:-110,
 as_of:'2026-09-12T15:45:00Z',quote_time:'2026-09-12T15:45:00Z',start:'2026-09-12T18:00:00Z'});
const a=row('A',.7,.1),b=row('B',.7,.2,'NCAAF'),c=row('C',.75);
const invalid=[{...row('Pass',.99),status:'PASS'}, {...row('Started',.99),start:'2026-09-12T16:00:00Z'},
 {...row('Stale quote',.99),quote_time:'2026-09-12T15:00:00Z'},
 {...row('Stale analysis',.99),as_of:'2026-09-12T15:00:00Z'},
 {...row('Missing quote',.99),quote_source:''},{...row('Tomorrow',.99),start:'2026-09-13T18:00:00Z'},row('No edge',.99,0),row('Null probability',null),row('Invalid probability',1.1)];
const rows=[a,...invalid,c,b];const before=JSON.stringify(rows);
assert.deepEqual(topPicks(rows).map(r=>r.game),['C','B','A']);assert.equal(JSON.stringify(rows),before);
assert.equal(topPicks(Array.from({length:15},(_,i)=>row('Game '+i,.5+i/100))).length,10);
assert.deepEqual(topPicks([row('Z',.7),row('D',.7)]).map(r=>r.game),['D','Z']);
data.games.overall=rows;renderTopPicks();
const text=n=>[n.text||'',...n.children.map(text)].join(' ');
assert.match(text(host),/Showing 3 qualifying picks/);assert.match(text(host),/NCAAF/);
const rendered=JSON.stringify(host);selectedLeague='NFL';renderTopPicks();assert.equal(JSON.stringify(host),rendered);
const tableNode=host.children.at(-1).children[0];
assert.deepEqual(tableNode.children[0].children[0].children.slice(0,2).map(n=>n.text),['Rank','League']);
assert.deepEqual(tableNode.children[1].children.map(n=>n.children[0].text),['1','2','3']);
now=Date.parse('2026-09-12T16:16:00Z');renderTopPicks();assert.match(text(host),/No qualifying picks available/);
assert.equal(JSON.stringify(rows),before);
now=Date.parse('2026-09-12T16:00:00Z');delete data.selection_policy;assert.deepEqual(topPicks(rows),[]);
"""
    target = tmp_path/'top-ten.cjs'
    target.write_text(script, encoding='utf-8')
    subprocess.run([node, str(target)], check=True)
    assert html.index('id="topPicks"') < html.index('id="gameLeague"')
    assert 'function render(){renderTopPicks();renderLockedPicks();' in html
