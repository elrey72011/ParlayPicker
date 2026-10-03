"""The selected-game trace preserves producer facts when the browser clock moves."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from app_core.board_diagnostics import build_selected_diagnostics, validate_selected_diagnostics


AT = datetime(2026, 9, 23, 20, 0, tzinfo=timezone.utc)


def row(**changes):
    result = dict(
        sport="MLB", game="Away at Home", pick="Away +1.5",
        market="spread_away", odds=-110.0, win_estimate=.55, ev=.02,
        status="PASS", start="2026-09-23T23:00:00Z",
        as_of="2026-09-23T19:55:00Z",
        quote_source="Novig", quote_time="2026-09-23T19:55:00Z",
    )
    result.update(changes)
    return result


def test_selected_rows_are_versioned_and_candidate_population_is_unknown():
    public = [row()]
    diagnostics = build_selected_diagnostics([{"matchup_id": "g1"}], public, AT, 30)
    assert diagnostics["schema_version"] == "selected-overall-v1"
    assert diagnostics["selected_game_count"] == 1
    assert diagnostics["all_market_candidate_count"] is None
    trace = diagnostics["traces"][0]
    assert trace["game_id"] == "g1"
    assert trace["source_candidate_id"] is None
    assert trace["identity_status"] == "UNKNOWN"
    assert trace["strict_gate_status"] == "UNKNOWN"
    assert trace["producer_primary_reason"] == "STRICT_TRACE_UNAVAILABLE"
    assert trace["evaluated_at"] == AT.isoformat()
    validate_selected_diagnostics(diagnostics, public, AT.isoformat(), 30)
    with pytest.raises(ValueError, match="saved game rows"):
        validate_selected_diagnostics(diagnostics, [dict(public[0], odds=-115)], AT.isoformat(), 30)
    changed = deepcopy(diagnostics)
    changed["traces"][0]["private_balance"] = 1000
    with pytest.raises(ValueError, match="trace fields"):
        validate_selected_diagnostics(changed, public, AT.isoformat(), 30)


def test_explicit_upstream_failures_and_missing_facts_survive_snapshot():
    contract = {
        "identity_verified": False, "quote_verified": False, "quote_fresh": False,
        "data_quality_status": "UNVERIFIED", "model_version": None,
        "calibration_version": None, "evidence_version": None,
        "sport_policy_version": None, "conservative_probability": None,
        "conservative_ev": None, "production_eligible": False,
        "production_bet_amount": 0,
    }
    public = [row(wager_contract=contract, quote_time="2026-09-23T19:20:00Z")]
    trace = build_selected_diagnostics([{"matchup_id": "g1"}], public, AT, 30)["traces"][0]
    assert trace["identity_status"] == "FAILED"
    assert trace["quote_status"] == "EXPIRED"
    assert trace["strict_gate_status"] == "BLOCKED"
    assert trace["producer_primary_reason"] == "QUOTE_EXPIRED"
    assert {"IDENTITY_UNVERIFIED", "MODEL_VERSION_MISSING", "CALIBRATION_VERSION_MISSING",
            "EVIDENCE_VERSION_MISSING", "POLICY_VERSION_MISSING",
            "CONSERVATIVE_PROBABILITY_MISSING", "NO_VALIDATED_ALLOCATION"} <= set(trace["producer_blockers"])
    assert trace["probability_semantics"] is None
    assert trace["p_win"] is None


def test_saved_trial_keeps_distinct_authority_without_inventing_owner_consent():
    trial = {
        "trial_eligible": True, "identity_verified": True, "quote_verified": True,
        "quote_fresh": True, "gemini_review_status": "APPROVE",
        "recommended_bet_amount": 3.0,
        "estimated_probability": .55, "push_probability": 0.0,
        "loss_probability": .45, "probability_semantics": "win_unconditional_with_push",
    }
    public = [row(status="TRIAL", controlled_trial_contract=trial)]
    trace = build_selected_diagnostics([{"matchup_id": "g1"}], public, AT, 30)["traces"][0]
    assert trace["trial_gate_status"] == "PASS"
    assert trace["strict_gate_status"] == "UNKNOWN"
    assert trace["authorization_status"] == "CONTRACT_CLAIMED"
    assert trace["allocation_status"] == "POSITIVE_TRIAL"
    assert (trace["p_win"], trace["p_push"], trace["p_loss"]) == (.55, 0.0, .45)
    assert trace["producer_primary_reason"] == "STRICT_TRACE_UNAVAILABLE"


def test_producer_clock_distinguishes_boundary_future_and_missing():
    at_boundary = row(as_of="2026-09-23T19:30:00Z", quote_time="2026-09-23T19:30:00Z")
    trace = build_selected_diagnostics([{}], [at_boundary], AT, 30)["traces"][0]
    assert trace["quote_status"] == "CURRENT"
    assert "ANALYSIS_EXPIRED" not in trace["producer_blockers"]
    expired = row(as_of="2026-09-23T19:29:59Z", quote_time="2026-09-23T19:29:59Z")
    trace = build_selected_diagnostics([{}], [expired], AT, 30)["traces"][0]
    assert trace["quote_status"] == "EXPIRED"
    assert "ANALYSIS_EXPIRED" in trace["producer_blockers"]
    future = row(as_of="2026-09-23T20:00:01Z", quote_time="2026-09-23T20:00:01Z")
    trace = build_selected_diagnostics([{}], [future], AT, 30)["traces"][0]
    assert trace["quote_status"] == "FUTURE"
    assert "ANALYSIS_TIME_FUTURE" in trace["producer_blockers"]
    missing_quote = row(quote_time=None)
    trace = build_selected_diagnostics([{}], [missing_quote], AT, 30)["traces"][0]
    assert trace["quote_status"] == "TIME_UNAVAILABLE"


def test_browser_keeps_saved_and_current_blockers_with_reconciling_counts(tmp_path):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    html = Path("publishing/board.html").read_text(encoding="utf-8")
    helpers = html.split("// The saved producer trace is never recomputed in the browser. Current blockers\n", 1)[1].split("let boardPublicationVersion=null;", 1)[0]
    script = r"""
const assert=require('node:assert/strict');
Date.now=()=>Date.parse('2026-09-23T20:00:00Z');
const supportedQuote=r=>r.quote_source==='Novig';
const formatShortAge=iso=>String(Math.floor((Date.now()-Date.parse(iso))/60000))+' min ago';
const base={sport:'MLB',market:'spread_away',pick:'Away +1.5',quote_source:'Novig',
  start:'2026-09-23T23:00:00Z',status:'PASS',ev:.02};
const rows=[
 {...base,as_of:'2026-09-23T19:55:00Z',quote_time:'2026-09-23T19:55:00Z'},
 {...base,status:'APPROVED',as_of:'2026-09-23T19:55:00Z',quote_time:'2026-09-23T19:20:00Z'},
 {...base,status:'APPROVED',as_of:'2026-09-23T19:20:00Z',quote_time:'2026-09-23T19:55:00Z'}
];
const data={stale_after_minutes:30,games:{overall:rows},props:[{as_of:'2026-09-23T17:00:00Z'}],
  board_diagnostics:{all_market_candidate_count:null,traces:[
    {saved_status:'PASS',producer_primary_reason:'MODEL_VERSION_MISSING',producer_blockers:['MODEL_VERSION_MISSING']},
    {saved_status:'APPROVED',producer_primary_reason:'SAVED_VALIDATED_APPROVAL',producer_blockers:[]},
    {saved_status:'APPROVED',producer_primary_reason:'SAVED_VALIDATED_APPROVAL',producer_blockers:[]}
  ]}};
""" + helpers + r"""
const report=selectedBoardReport(data);
assert.equal(report.selectedGames,3);
assert.equal(report.allMarketCandidates,null);
assert.equal(Object.values(report.primary).reduce((a,b)=>a+b,0),3);
assert.equal(report.primary.MODEL_VERSION_MISSING,1);
assert.equal(report.primary.QUOTE_EXPIRED,1);
assert.equal(report.primary.ANALYSIS_EXPIRED,1);
assert.deepEqual(selectedRowDecision(rows[0]).saved,['MODEL_VERSION_MISSING']);
assert.deepEqual(selectedRowDecision(rows[1]).current,['QUOTE_EXPIRED']);
assert.equal(cohortClock(rows,'as_of').expired,1);
assert.equal(cohortClock(rows,'quote_time').expired,1);
assert.equal(cohortClock(data.props,'as_of').expired,1);
assert.equal(rows[1].status,'APPROVED'); // Browser assessment does not rewrite saved history.
const invalid={...base,start:'invalid',as_of:'2026-09-23T20:01:00Z',quote_time:null};
assert.deepEqual(currentRowBlockers(invalid),['START_TIME_UNAVAILABLE','QUOTE_TIME_UNAVAILABLE','ANALYSIS_TIME_FUTURE']);
assert.equal(cohortClock([invalid],'as_of').expired,0);
assert.equal(cohortClock([invalid],'as_of').future,1);
assert.equal(cohortClock([invalid],'as_of').unavailable,1);
"""
    target = tmp_path / "board-diagnostics.cjs"
    target.write_text(script, encoding="utf-8")
    subprocess.run([node, str(target)], check=True, capture_output=True, text=True)


def test_quote_clock_separates_expiry_from_unavailable_times_and_keeps_blockers(tmp_path):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    html = Path("publishing/board.html").read_text(encoding="utf-8")
    helpers = html.split("// The saved producer trace is never recomputed in the browser. Current blockers\n", 1)[1].split("let boardPublicationVersion=null;", 1)[0]
    banner = html.rsplit("window.parlayPicker.freshness=function(version)", 1)[1].split("\n", 1)[0]
    script = r"""
const assert=require('node:assert/strict');
const now=Date.parse('2026-10-02T22:04:42Z');
Date.now=()=>now;
const data={stale_after_minutes:30,games:{overall:[]},props:[]};
const supportedQuote=r=>r.quote_source==='Novig';
const formatShortAge=iso=>Math.floor((now-Date.parse(iso))/60000)+' min ago';
const renderBoardSummary=()=>{};
let boardPublicationVersion=null;
let priorState='current';
const priorBoardFreshness=()=>({primary:'Analysis',publication:'Publication',state:priorState});
const window={parlayPicker:{}};
""" + helpers + "\nwindow.parlayPicker.freshness=function(version)" + banner + r"""
const base={sport:'MLB',quote_source:'Novig',status:'APPROVED',ev:.02,
 start:'2026-10-03T02:30:00Z',as_of:'2026-10-02T21:50:10.731933Z'};
const withTime=value=>({...base,quote_time:value});
const fresh='2026-10-02T21:49:07Z';
assert.deepEqual(currentRowBlockers(withTime(fresh)),[]);
for(const [value,category,blocker] of [
 [null,'missing','QUOTE_TIME_UNAVAILABLE'],[undefined,'missing','QUOTE_TIME_UNAVAILABLE'],
 ['','missing','QUOTE_TIME_UNAVAILABLE'],['malformed','invalid','QUOTE_TIME_UNAVAILABLE'],
 ['2026-10-02T22:00:00','invalid','QUOTE_TIME_UNAVAILABLE'],[42,'invalid','QUOTE_TIME_UNAVAILABLE'],
 ['2026-10-02T22:04:43Z','future','QUOTE_TIME_FUTURE']
]){
 const clock=cohortClock([withTime(value)],'quote_time');
 assert.equal(clock.expired,0);assert.equal(clock[category],1);assert.equal(clock.unavailable,1);
 assert.ok(clock.label.includes(category+' timestamps: 1'));
 assert.ok(!clock.label.includes('expired'));assert.ok(!clock.label.includes('within wager window'));
 assert.deepEqual(currentRowBlockers(withTime(value)),[blocker]);
}
for(const minutes of [15,30]){
 data.stale_after_minutes=minutes;
 const boundary=new Date(now-minutes*60000).toISOString();
 assert.equal(cohortClock([withTime(boundary)],'quote_time').expired,0);
 assert.deepEqual(currentRowBlockers(withTime(boundary)),[]);
 const expired=new Date(now-minutes*60000-1).toISOString();
 assert.equal(cohortClock([withTime(expired)],'quote_time').expired,1);
 assert.deepEqual(currentRowBlockers(withTime(expired)),['QUOTE_EXPIRED']);
}
data.stale_after_minutes=30;
// The audit's frozen ten-card reproduction: five NHL, three NCAAF, one WNBA, one missing.
const rows=[...Array(5).fill(fresh),...Array(3).fill('2026-10-02T21:48:32Z'),
 '2026-10-02T21:49:10Z',null].map(withTime);
const saved=JSON.stringify(rows),clock=cohortClock(rows,'quote_time');
assert.equal(clock.expired,0);assert.equal(clock.missing,1);assert.equal(clock.unavailable,1);
assert.equal(clock.label,'16 min ago · missing timestamps: 1');
const later=cohortClock(rows,'quote_time',Date.parse('2026-10-02T23:59:30Z'));
assert.equal(later.expired,9);assert.equal(later.missing,1);
assert.equal(later.invalid,0);assert.equal(later.future,0);
const mixed=cohortClock([withTime(fresh),withTime('2026-10-02T21:00:00Z'),
 withTime(null),withTime('bad'),withTime('2026-10-02T23:00:00Z')],'quote_time');
assert.deepEqual([mixed.expired,mixed.missing,mixed.invalid,mixed.future,mixed.unavailable],[1,1,1,1,3]);
assert.ok(mixed.label.includes('wager window expired: 1'));
assert.equal(cohortClock([],'quote_time').label,'Not included');
assert.equal(JSON.stringify(rows),saved); // Clock assessment cannot alter saved decisions.
data.games.overall=[base];
assert.equal(window.parlayPicker.freshness({}).state,'current');
data.games.overall=[{...base,as_of:null}];
let bannerResult=window.parlayPicker.freshness({});
assert.equal(bannerResult.state,'unavailable');assert.ok(!bannerResult.text.includes('expired'));
assert.ok(bannerResult.text.includes('missing, invalid or future analysis timestamps for 1'));
data.games.overall=[{...base,as_of:'2026-10-02T21:00:00Z'},{...base,as_of:null}];
bannerResult=window.parlayPicker.freshness({});
assert.equal(bannerResult.state,'expired');assert.ok(bannerResult.text.includes('expired for 1'));
priorState='old';assert.equal(window.parlayPicker.freshness({}).state,'old');
"""
    target = tmp_path / "quote-clock.cjs"
    target.write_text(script, encoding="utf-8")
    subprocess.run([node, str(target)], check=True, capture_output=True, text=True)
