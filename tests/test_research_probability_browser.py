"""Actual static renderer in an installed headless browser; no provider transport."""
from __future__ import annotations
from datetime import datetime, timedelta
import html as html_parser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd
import pytest
from app_core.public_board import build_package,validate_package
from scripts.publish_board import render
from test_research_probability_display import NOW,source,package_for

def browser_binary():
    candidates=[shutil.which("google-chrome"),shutil.which("chromium"),
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
    return next((str(p) for p in candidates if p and Path(p).is_file()),None)

def inspect_browser(package, directory, clock, *, screenshot=False, rendered_html=None):
    binary=browser_binary()
    if binary is None:
        pytest.skip("Installed Chrome/Edge required for actual browser regression")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    html=rendered_html if rendered_html is not None else render(package)
    at=int(clock.timestamp()*1000)
    freeze="""<script>window.__probabilityClock=AT;const SavedDate=Date;
window.Date=class extends SavedDate{constructor(...args){super(...(args.length?args:[window.__probabilityClock]));}static now(){return window.__probabilityClock;}};
window.__browserErrors=[];addEventListener('error',e=>__browserErrors.push(e.message));
window.fetch=()=>Promise.reject(Error('Fake preview transport: no refresh permitted'));</script>""".replace("AT",str(at),1)
    html=html.replace("<head>","<head>"+freeze,1)
    inspector="""<script>
const initialClock=window.__probabilityClock;
const unchanged=JSON.stringify(data);
const read=()=>({selected:data.games.overall.length,current:data.games.overall.filter(currentWager).length,
top:topPicks(data.games.overall).length,heading:document.querySelector('#board-title').textContent,
cards:[...document.querySelectorAll('#gameBoards .pp-pick-card')].map(c=>c.textContent),
shown:data.games.overall.map(r=>typeof cardEstimate==='function'?cardEstimate(r):{probability:r.win_estimate,ev:r.ev,edge:r.estimated_price_edge??null,breakEven:r.break_even_probability??null,research:false}),saved:data.games.overall.map(r=>({probability:r.win_estimate,ev:r.ev,status:r.status,
stake:r.wager_contract?.production_bet_amount??r.controlled_trial_contract?.recommended_bet_amount??0}))});
const report={initial:read()};
for(const view of ['sides','totals','overall']){document.querySelector('[data-board-view='+view+']').click();report[view]=document.querySelectorAll('#gameBoards .pp-pick-card').length;}
window.__probabilityClock=initialClock+31*60000;render();report.expired=read();
window.__probabilityClock=Math.max(...data.games.overall.map(r=>Date.parse(r.start)));render();report.started=read();
report.immutable=JSON.stringify(data)===unchanged;report.errors=window.__browserErrors;
window.__probabilityClock=initialClock;render();
const evidence=document.createElement('pre');evidence.id='probability-test-result';evidence.style.display='none';
evidence.textContent=JSON.stringify(report);document.body.append(evidence);
</script>"""
    html=html.replace("</body>",inspector+"</body>")
    target=directory/"index.html";target.write_text(html,encoding="utf-8")
    command=[binary,"--headless=new","--disable-gpu","--no-first-run",
        "--no-default-browser-check","--disable-background-networking","--disable-component-update",
        "--disable-sync","--metrics-recording-only","--host-resolver-rules=MAP * 0.0.0.0",
        "--user-data-dir="+str(directory/"profile"),"--dump-dom","--virtual-time-budget=1000",
        "--window-size=1280,1900"]
    if screenshot:
        command+=["--screenshot="+str(directory/"preview.png"),"--hide-scrollbars"]
    command.append(target.resolve().as_uri())
    run=subprocess.run(command,capture_output=True,
        timeout=40,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    try:
        output=run.stdout.decode("utf-8")
    except UnicodeDecodeError:
        if os.name!="nt":raise
        output=run.stdout.decode("cp1252")
    match=re.search(r'<pre id="probability-test-result"[^>]*>(.*?)</pre>',output,re.S)
    assert run.returncode==0 and match,"Actual browser did not complete the local renderer regression"
    result=json.loads(html_parser.unescape(match.group(1)))
    (directory/"browser-result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    assert result["errors"]==[] and result["immutable"]
    assert result["expired"]["current"]==result["started"]["current"]==0
    return result

def test_actual_browser_preserves_research_without_current_wagers(monkeypatch,tmp_path):
    _,package=package_for(monkeypatch)
    result=inspect_browser(package,tmp_path/"research",NOW)
    initial=result["initial"]
    assert initial["current"]==initial["top"]==0 and initial["selected"]==1
    assert initial["shown"][0]["probability"]==.6 and initial["shown"][0]["ev"]>0
    assert initial["saved"]==[dict(probability=None,ev=None,status="PASS",stake=0)]
    assert "Research estimate60.0%" in initial["cards"][0]
    assert "PASS" in initial["cards"][0] and "not approved" in initial["cards"][0]
    assert result["overall"]==result["sides"]==result["totals"]==1
    assert result["expired"]["shown"][0]["probability"]==.6

@pytest.mark.parametrize("updates,expected",[
    ({"best_available_probability":None},None),
    ({"best_available_probability":True},None),
    ({"ml_target":"home_win"},None),
    ({"inference_status":"FAILED"},None),
    ({"best_available_probability":0.0},0.0),
    ({"best_available_probability":.2},.2)])
def test_actual_browser_missing_target_inference_and_zero(monkeypatch,tmp_path,updates,expected):
    _,package=package_for(monkeypatch,source(**updates))
    result=inspect_browser(package,tmp_path/"case",NOW)
    assert result["initial"]["shown"][0]["probability"]==expected
    assert result["initial"]["current"]==0
    if expected is not None:
        assert result["initial"]["shown"][0]["ev"]<0

def test_actual_browser_approved_and_trial_labels_preserve_contracts(monkeypatch,tmp_path):
    from test_current_wagers_trace_and_release import _approved_source
    from test_wager_integrity_audit import NOW as approved_clock
    raw=_approved_source();frame=pd.DataFrame([raw])
    approved=build_package(frame.copy(),frame.copy(),frame.copy());validate_package(approved)
    result=inspect_browser(approved,tmp_path/"approved",approved_clock)
    assert result["initial"]["current"]==1
    assert "Validated estimate" in result["initial"]["cards"][0]
    assert result["initial"]["saved"][0]["stake"]==raw["wager_contract"]["production_bet_amount"]
    from test_controlled_trial_integration import final as trial_final,candidate as trial_candidate,RUN
    from app_core.per_game_boards import per_game_board
    raw=trial_final()
    frame=pd.DataFrame([raw]);audit=pd.DataFrame([trial_candidate()])
    trial=build_package(*[per_game_board(frame,audit,family=k,novig_only=True) for k in ("overall","sides","totals")])
    validate_package(trial)
    result=inspect_browser(trial,tmp_path/"trial",datetime.fromisoformat("2026-09-11T20:05:00+00:00"))
    assert result["initial"]["current"]==1
    assert "Controlled-trial estimate" in result["initial"]["cards"][0]
    assert result["initial"]["saved"][0]["status"]=="TRIAL"
    assert result["initial"]["saved"][0]["stake"]==raw["controlled_trial_contract"]["recommended_bet_amount"]

@pytest.mark.parametrize("key,value",[("matchup_id","other"),("pick","Away +1.5"),("line",-2.5),
    ("market_type","total_over"),("market_period","first_half"),("settlement_rules","regulation_only"),
    ("odds",-120),("quote_id","other"),("quote_time",(NOW-timedelta(minutes=17)).isoformat())])
def test_actual_browser_conflicting_snapshot_never_borrows(monkeypatch,tmp_path,key,value):
    frames,_=package_for(monkeypatch)
    for frame in frames[:2]:frame.loc[0,key]=value
    if key=="matchup_id":frames[2].loc[0,key]=value
    package=build_package(*frames);validate_package(package)
    result=inspect_browser(package,tmp_path/"conflict",NOW)
    assert result["initial"]["shown"][0]["probability"] is None
    assert result["initial"]["current"]==result["initial"]["top"]==0
    assert "Estimate does not match" in result["initial"]["cards"][0]

def test_actual_browser_push_and_legacy(monkeypatch,tmp_path):
    from test_research_probability_display import QUOTE
    raw=source(best_pick="Home -2",spread_line=-2.0,odds_american=100,
        best_available_probability=.575,probability_semantics="win_conditional_on_decision",
        push_probability=.1,provider_quotes=json.dumps([dict(book="novig",market_type="spread_home",
                                            point=-2,price=100,recorded_at=QUOTE)]))
    raw["wager_contract"].update(selection="Home -2",line=-2.0,odds=100)
    _,package=package_for(monkeypatch,raw)
    result=inspect_browser(package,tmp_path/"push",NOW)
    shown=result["initial"]["shown"][0]
    assert shown["probability"]==pytest.approx(.5175) and shown["ev"]==pytest.approx(.135)
    assert shown["edge"]==pytest.approx(.0675) and shown["breakEven"]==.45
    for rows in package["games"].values():
        for row in rows:row.pop("research_display")
    package.pop("board_diagnostics")
    result=inspect_browser(package,tmp_path/"legacy",NOW)
    assert result["initial"]["shown"][0]["probability"] is None
    assert result["initial"]["current"]==0

def test_actual_browser_thirteen_games_and_unknown_candidate_count(monkeypatch,tmp_path):
    from app_core.per_game_boards import per_game_board
    raw=[]
    for index in range(13):
        row=source(candidate_id=f"c-{index}",matchup_id=f"g-{index}",quote_id=f"q-{index}",
                   Home=f"Home{index}",Away=f"Away{index}",best_pick=f"Home{index} -1.5")
        row["wager_contract"].update(game_id=f"g-{index}",matchup_id=f"g-{index}",selection=row["best_pick"])
        raw.append(row)
    frame=pd.DataFrame(raw)
    frames=[per_game_board(frame,family=k,novig_only=True) for k in ("overall","sides","totals")]
    package=build_package(*frames);validate_package(package)
    result=inspect_browser(package,tmp_path/"thirteen",NOW)
    assert result["initial"]["selected"]==result["overall"]==result["sides"]==result["totals"]==13
    assert result["initial"]["current"]==result["initial"]["top"]==0
    assert package["board_diagnostics"]["all_market_candidate_count"] is None
    assert len(result["initial"]["cards"])==13

@pytest.mark.parametrize("updates",[
    {"market_period":""},{"settlement_rules":""},{"ml_target":""},
    {"best_available_probability":float("inf")},{"best_available_probability":float("nan")},
    {"best_available_probability":1.2},{"best_available_probability":-.1},
    {"inference_status":"UNAVAILABLE"}])
def test_actual_browser_invalid_or_missing_provenance(monkeypatch,tmp_path,updates):
    _,package=package_for(monkeypatch,source(**updates))
    result=inspect_browser(package,tmp_path/"missing",NOW)
    assert result["initial"]["shown"][0]["probability"] is None
    assert result["initial"]["current"]==0

@pytest.mark.parametrize("push,semantics,expected",[
    (0.0,"win_unconditional_with_push",.6),
    (None,"",.6),
    (None,"win_unconditional_with_push",None),
    (.1,"win_unconditional_with_push",None),
    (-.1,"win_unconditional_with_push",None),
    (True,"win_unconditional_with_push",None),
    (False,"win_unconditional_with_push",None)])
def test_actual_browser_explicit_missing_and_inconsistent_push(monkeypatch,tmp_path,push,semantics,expected):
    from scripts.publish_board import assets_from_html
    _,package=package_for(monkeypatch,source(push_probability=push,probability_semantics=semantics))
    html=render(package)
    serialized=json.loads(assets_from_html(html)["board-data.json"]);validate_package(serialized)
    result=inspect_browser(serialized,tmp_path/"push-case",NOW,rendered_html=html)
    assert result["initial"]["shown"][0]["probability"]==expected
    assert result["initial"]["current"]==result["initial"]["top"]==0
    assert result["initial"]["saved"]==[dict(probability=None,ev=None,status="PASS",stake=0)]
