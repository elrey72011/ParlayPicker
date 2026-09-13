"""Display order uses saved probabilities, never composite scores or newer locks."""
import os
import shutil
import subprocess
from pathlib import Path
import pytest


def test_probability_display_order_and_rendered_lists(tmp_path):
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    html=Path('publishing/board.html').read_text(encoding='utf-8')
    funcs='\n'.join(line for line in html.splitlines() if line.startswith(('function probabilityOrder(', 'function lockedRows(', 'function render(){', 'function renderProps(')))
    script="""
const assert=require('node:assert/strict');
const rows=[{game:'Low',sport:'NFL',win_estimate:.51,ev:.8,selection_score:.99},
{game:'High',sport:'NFL',win_estimate:.8,ev:-.1,selection_score:.2},
{game:'Missing',sport:'NFL',win_estimate:null,selection_score:1},
{game:'Zero',sport:'MLB',win_estimate:0},
{game:'Bad',sport:'MLB',win_estimate:true}];
const elements={};const document={getElementById:id=>elements[id]||(elements[id]={value:'',textContent:'',append(){},replaceChildren(){}}),querySelectorAll:()=>[]};
const el=(tag,text)=>text;
const data={games:{overall:rows,sides:rows,totals:rows},props:rows,built_at:'2026-09-13T12:00:00Z',stale_after_minutes:30};
const renderTopPicks=()=>{},renderLockedPicks=()=>{},renderParlays=()=>{},renderResults=()=>{};
const qualifiedPick=()=>true;let tables=[];const table=(rows,props,ranked)=>{tables.push({names:rows.map(r=>r.game),ranked});return rows};
"""+funcs+"""
const before=JSON.stringify(rows);
assert.deepEqual(probabilityOrder(rows).map(r=>r.game),['High','Low','Zero','Bad','Missing']);
assert.deepEqual(probabilityOrder([{game:'B',win_estimate:.6,ev:1},{game:'A',win_estimate:.6,ev:-1}]).map(r=>r.game),['A','B']);
assert.equal(probabilityOrder([{win_estimate:1.1},{win_estimate:NaN},{win_estimate:'0.9'},{win_estimate:1}])[0].win_estimate,1);
const locks=rows.slice(0,2).map((r,i)=>({...r,group:'Locked',category:'overall',date:'2026-09-13',original_win_estimate:i?.55:.7}));
assert.equal(lockedRows(locks,'2026-09-13')[0].game,'Low');
assert.equal(lockedRows(locks,'2026-09-13','MLB').length,0);
render();assert.equal(tables.length,4);assert.ok(tables.every(t=>t.ranked&&t.names[0]==='High'));
tables=[];document.getElementById('gameLeague').value='NFL';document.getElementById('sport').value='NFL';render();
assert.ok(tables.every(t=>t.names.join(',')==='High,Low,Missing'));
assert.equal(JSON.stringify(rows),before);
"""
    target=tmp_path/'probability-order.cjs';target.write_text(script,encoding='utf-8')
    subprocess.run([node,str(target)],check=True,capture_output=True,text=True)
