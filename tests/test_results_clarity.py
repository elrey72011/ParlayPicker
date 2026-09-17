import os
from pathlib import Path
import shutil
import subprocess
import pytest


def test_browser_defaults_saved_filters_yesterday_and_overlap(tmp_path):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    html = Path("publishing/board.html").read_text(encoding="utf-8")
    initialization = html.split("let data=",1)[1].split("const el=",1)[0]
    helpers = "\n".join(line for line in html.splitlines() if line.startswith(("function easternDay(","function yesterdayDate(","function yesterdayLockedRows(","function renderYesterdayLocked(","function activeResultLabel(")))
    script = r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
const init=INIT,helpers=HELPERS;
function run(saved,locked=true){
 class E{constructor(value='',options=[]){this.value=value;this.options=options.map(([value,textContent])=>({value,textContent}));this.children=[];this.textContent='';}append(o){this.options.push(o)}replaceChildren(...v){this.children=v}}
 const nodes={resultPeriod:new E('1',[['1','Yesterday'],['all','All time']]),resultGroup:new E('Published picks',['Published picks','Locked','Research'].map(x=>[x,x])),resultKind:new E('games',[['games','Game picks & parlays'],['props','Player Props']]),resultSport:new E('',[['','All leagues']]),resultMarket:new E('',[['','All markets']])};
 const rows=[{id:'p',date:'2026-09-15',group:'Research',category:'overall',outcome:'WIN'},{id:'side',date:'2026-09-15',group:'Research',category:'sides',outcome:'WIN'}];
 if(locked)rows.push(...['WIN','LOSS','PUSH','PENDING'].map((outcome,i)=>({id:'l'+i,date:'2026-09-15',group:'Locked',category:'overall',outcome})),{date:'2026-09-14',group:'Locked',category:'overall',outcome:'WIN'});
 const clock=class extends Date{constructor(...a){super(...(a.length?a:['2026-09-16T15:00:00Z']))}static now(){return Date.parse('2026-09-16T15:00:00Z')}};
 const context={window:{},Date:clock,Intl,document:{getElementById(id){if(id==='board-data')return {textContent:JSON.stringify({results:rows})};return nodes[id]||(nodes[id]=new E());},createElement:()=>({})},localStorage:{getItem:()=>saved?JSON.stringify(saved):null},el:(tag,text)=>({tag,text}),fmt:(n,p)=>p?(100*n).toFixed(1)+'%':n};
 vm.createContext(context);vm.runInContext(helpers+'\nconst data='+init+'\nrenderYesterdayLocked();globalThis.slice=activeResultLabel();globalThis.yesterdayCount=yesterdayLockedRows().length;',context);
 return {nodes,context};
}
let r=run(null);assert.equal(r.nodes.resultPeriod.value,'1');assert.equal(r.nodes.resultGroup.value,'Locked');assert.equal(r.context.yesterdayCount,4);assert.match(JSON.stringify(r.nodes.yesterdayLockedSummary.children),/50.0%/);assert.match(r.context.slice,/Yesterday · Locked · Game picks/);
r=run(null,false);assert.equal(r.nodes.resultGroup.value,'Published picks');assert.match(r.nodes.resultFallback.textContent,/Initial table view: Yesterday/);assert.match(JSON.stringify(r.nodes.yesterdayLockedSummary.children),/No settled Locked Overall/);
r=run({period:'all',group:'Research',kind:'games'});assert.equal(r.nodes.resultPeriod.value,'all');assert.equal(r.nodes.resultGroup.value,'Research');assert.match(r.context.slice,/All time · Research/);assert.equal(r.nodes.resultFallback.textContent,'');
r=run({period:'bad',group:'bad'});assert.equal(r.nodes.resultPeriod.value,'1');assert.equal(r.nodes.resultGroup.value,'Locked');
"""
    import json
    script = script.replace("INIT",json.dumps(initialization)).replace("HELPERS",json.dumps(helpers))
    target=tmp_path/"results-clarity.cjs";target.write_text(script,encoding="utf-8")
    subprocess.run([node,str(target)],check=True,capture_output=True,text=True)
