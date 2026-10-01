"""Display fixtures are synthetic; they confer no wagering authority."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import pandas as pd
import pytest

from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package, validate_package
from app_core.release_preflight import evaluate_release
from core.live_wager_contract import snapshot
from scripts.publish_board import render, assets_from_html

NOW = datetime(2026, 10, 1, 19, 40, tzinfo=timezone.utc)
ANALYSIS = (NOW-timedelta(minutes=14)).isoformat()
QUOTE = (NOW-timedelta(minutes=16)).isoformat()
START = (NOW+timedelta(hours=3)).isoformat()

class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz or timezone.utc)

def source(**changes):
    from test_per_game_boards import final
    contract = snapshot(dict(game_id="display-game", matchup_id="display-game", sport="MLB",
        market_type="spread_home", selection="Home -1.5", line=-1.5, odds_american=-110,
        book="Novig", quote_time=QUOTE, start=START, maturity="RESEARCH",
        production_eligible=False, production_gate_reason="missing_or_invalid_conservative_probability",
        recommended_stake=0), NOW)
    row = final(gid="display-game", candidate_id="display-candidate", quote_id="display-quote",
        export_run_id=ANALYSIS, game_time_est=START, spread_line=-1.5,
        odds_source="novig", quote_time=QUOTE, market_period="full_game", settlement_rules="includes_overtime",
        best_available_selection_policy="probability-first-v1", best_available_probability=.6,
        best_available_probability_source="calibrated_probability", ml_target="spread_cover",
        probability_semantics="win_unconditional_with_push", push_probability=0.0,
        Bettable=False, Play_Stake=0, wager_contract=contract,
        Production_Gate_Reason="missing_or_invalid_conservative_probability",
        provider_quotes=json.dumps([dict(book="novig",market_type="spread_home",point=-1.5,
                                        price=-110,recorded_at=QUOTE)]))
    row.update(changes)
    return row

def package_for(monkeypatch, value=None):
    monkeypatch.setattr("app_core.public_board.datetime", FrozenDateTime)
    raw = value or source()
    frame=pd.DataFrame([raw])
    frames=[per_game_board(frame, family=family, novig_only=True)
            for family in ("overall","sides","totals")]
    package=build_package(*frames)
    validate_package(package)
    return frames,package

def test_research_survives_real_export_builder_validator_and_renderer(monkeypatch):
    raw=source(); original=deepcopy(raw)
    frames,package=package_for(monkeypatch,raw)
    row=package["games"]["overall"][0]
    assert frames[0].iloc[0]["win_probability"] == .6
    assert row["win_estimate"] is None and row["ev"] is None
    assert row["status"]=="PASS" and row["wager_contract"]["production_bet_amount"]==0
    display=row["research_display"]
    assert display["probability"]==.6
    assert display["ev"]==pytest.approx(.6*(1+100/110)-1)
    assert display["label"]=="Research estimate"
    assert display["source_field"]=="best_available_probability"
    assert display["probability_semantics"]=="win_unconditional_with_push"
    assert display["push_probability"]==0
    assert raw==original
    html=render(package)
    assets=assets_from_html(html)
    assert json.loads(assets["board-data.json"])["games"]["overall"][0]["research_display"]==display
    assert "research_display" in html
    assert evaluate_release(package,at=NOW)["actionable_row_count"]==0

@pytest.mark.parametrize("value,reason",[
    (None,"ESTIMATE_NOT_RECORDED"),(pd.NA,"ESTIMATE_NOT_RECORDED"),
    (float("nan"),"ESTIMATE_NOT_RECORDED"),(float("inf"),"NONFINITE_PROBABILITY"),
    (True,"INVALID_PROBABILITY"),(False,"INVALID_PROBABILITY"),
    (-.1,"INVALID_PROBABILITY"),(1.1,"INVALID_PROBABILITY")])
def test_missing_invalid_and_boolean_estimates_are_never_defaults(monkeypatch,value,reason):
    _,package=package_for(monkeypatch,source(best_available_probability=value))
    display=package["games"]["overall"][0]["research_display"]
    assert display["probability"] is None and display["ev"] is None
    assert display["availability_reason"]==reason
    assert package["games"]["overall"][0]["status"]=="PASS"

@pytest.mark.parametrize("p",[0.0,.2,.6])
def test_zero_negative_ev_and_json_roundtrip(monkeypatch,p):
    import numpy as np
    _,package=package_for(monkeypatch,source(best_available_probability=np.float64(p)))
    display=package["games"]["overall"][0]["research_display"]
    assert display["probability"]==p
    assert display["ev"]==pytest.approx(p*(1+100/110)-1)
    restored=json.loads(json.dumps(package,allow_nan=False))
    validate_package(restored)
    assert restored["games"]["overall"][0]["research_display"]==display

@pytest.mark.parametrize("key,value",[
    ("matchup_id","other-game"),("pick","Away +1.5"),("line",-2.5),
    ("market_type","total_over"),("market_period","first_half"),
    ("settlement_rules","regulation_only"),("odds",-120),
    ("quote_id","other-quote"),("quote_time",(NOW-timedelta(minutes=17)).isoformat())])
def test_conflicting_export_snapshot_cannot_borrow_research(monkeypatch,key,value):
    frames,_=package_for(monkeypatch)
    for frame in frames[:2]:
        frame.loc[0,key]=value
    if key=="matchup_id":
        frames[2].loc[0,key]=value
    package=build_package(*frames)
    validate_package(package)
    display=package["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]=="ESTIMATE_IDENTITY_MISMATCH"
    assert display["probability"] is None and display["ev"] is None
    assert package["games"]["overall"][0]["status"]=="PASS"

@pytest.mark.parametrize("updates,reason",[
    ({"ml_target":""},"MODEL_TARGET_NOT_RECORDED"),
    ({"ml_target":"home_win"},"TARGET_MISMATCH"),
    ({"ml_target":"total_under"},"TARGET_MISMATCH"),
    ({"inference_status":"FAILED"},"INFERENCE_FAILED"),
    ({"inference_status":"UNAVAILABLE"},"INFERENCE_UNAVAILABLE"),
    ({"probability_semantics":"unsupported"},"UNSUPPORTED_PROBABILITY_SEMANTICS")])
def test_target_and_inference_failures_keep_specific_unavailability(monkeypatch,updates,reason):
    _,package=package_for(monkeypatch,source(**updates))
    display=package["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]==reason
    assert display["probability"] is None
    assert evaluate_release(package,at=NOW)["actionable_row_count"]==0

def test_integer_line_uses_unconditional_push_and_recorded_compatible_ev(monkeypatch):
    raw=source(best_pick="Home -2",spread_line=-2.0,odds_american=100,
        best_available_probability=.575,probability_semantics="win_conditional_on_decision",
        push_probability=.1,provider_quotes=json.dumps([dict(book="novig",market_type="spread_home",
                                            point=-2,price=100,recorded_at=QUOTE)]))
    raw["wager_contract"].update(selection="Home -2",line=-2.0,odds=100)
    _,package=package_for(monkeypatch,raw)
    d=package["games"]["overall"][0]["research_display"]
    assert d["probability"]==pytest.approx(.5175)
    assert d["push_probability"]==.1 and d["ev"]==pytest.approx(.135)
    assert d["break_even_probability"]==.45 and d["edge"]==pytest.approx(.0675)

def test_saved_ev_mismatch_is_not_combined_with_research_probability(monkeypatch):
    raw=source(best_available_selection_policy="",production_win_probability=.6,
               production_expected_value=-.5)
    _,package=package_for(monkeypatch,raw)
    d=package["games"]["overall"][0]["research_display"]
    assert d["probability"]==.6
    assert d["ev"] is None and d["edge"] is None
    assert d["value_reason"]=="PRICE_VALUE_MISMATCH"

def test_exact_contract_conflict_suppresses_display(monkeypatch):
    for key,value in [("game_id","other"),("selection","Away +1.5"),("line",-2.5),
                      ("market_type","total_over"),("odds",-120),("quote_timestamp",ANALYSIS)]:
        frames,_=package_for(monkeypatch)
        for frame in frames[:2]:
            c=deepcopy(frame.iloc[0]["wager_contract"]);c[key]=value
            frame.at[0,"wager_contract"]=c
        package=build_package(*frames);validate_package(package)
        d=package["games"]["overall"][0]["research_display"]
        assert d["probability"] is None and d["availability_reason"]=="ESTIMATE_IDENTITY_MISMATCH"

def test_real_approved_and_trial_contract_values_remain_exact(monkeypatch):
    from test_current_wagers_trace_and_release import _approved_source
    monkeypatch.setattr("app_core.public_board.datetime",FrozenDateTime)
    approved=_approved_source()
    frame=pd.DataFrame([approved])
    package=build_package(frame.copy(),frame.copy(),frame.copy());validate_package(package)
    r=package["games"]["overall"][0];c=approved["wager_contract"]
    assert r["status"]=="APPROVED" and r["win_estimate"]==c["conservative_probability"]
    assert r["ev"]==c["conservative_ev"] and r["wager_contract"]==c
    from test_controlled_trial_integration import final as trial_final, candidate as trial_candidate
    raw=trial_final()
    frames=[per_game_board(pd.DataFrame([raw]),pd.DataFrame([trial_candidate()]),
              family=family,novig_only=True) for family in ("overall","sides","totals")]
    package=build_package(*frames);validate_package(package)
    r=package["games"]["overall"][0];c=raw["controlled_trial_contract"]
    assert r["status"]=="TRIAL" and r["win_estimate"]==c["estimated_probability"]
    assert r["ev"]==c["estimated_expected_value"] and r["controlled_trial_contract"]==c
    assert frames[0].iloc[0]["Trial_Stake"]==c["recommended_bet_amount"]

def test_frozen_freshness_then_expiry_and_start_preserve_saved_research(monkeypatch):
    _,package=package_for(monkeypatch)
    saved=deepcopy(package)
    fresh=evaluate_release(package,at=NOW)
    late=evaluate_release(package,at=NOW+timedelta(minutes=31))
    started=evaluate_release(package,at=datetime.fromisoformat(START))
    assert fresh["actionable_row_count"]==late["actionable_row_count"]==0
    assert fresh["overall_current_primary_counts"]=={"SAVED_RESEARCH_PASS":1}
    assert "QUOTE_EXPIRED" in late["overall_current_primary_counts"]
    assert all("GAME_STARTED" in r["current_blockers"] for r in started["rows"] if r["section"]=="overall")
    assert package==saved

def test_thirteen_games_views_denominators_and_unknown_candidate_count(monkeypatch):
    monkeypatch.setattr("app_core.public_board.datetime",FrozenDateTime)
    raw=[]
    for index in range(13):
        row=source(candidate_id=f"c-{index}",matchup_id=f"g-{index}",quote_id=f"q-{index}",
                   Home=f"Home{index}",Away=f"Away{index}",best_pick=f"Home{index} -1.5")
        row["wager_contract"].update(game_id=f"g-{index}",matchup_id=f"g-{index}",selection=row["best_pick"])
        raw.append(row)
    frame=pd.DataFrame(raw)
    frames=[per_game_board(frame,family=k,novig_only=True) for k in ("overall","sides","totals")]
    package=build_package(*frames);validate_package(package)
    assert [len(v) for v in package["games"].values()]==[13,13,13]
    assert package["board_diagnostics"]["selected_game_count"]==13
    assert package["board_diagnostics"]["all_market_candidate_count"] is None
    assert evaluate_release(package,at=NOW)["actionable_row_count"]==0
    assert sum(r["research_display"]["probability"] is not None for r in package["games"]["overall"])==13

def test_display_never_promotes_actual_downstream_consumers(monkeypatch):
    from app_core.top_ten_history import ranked_picks
    from app_core.public_parlays import build_research_parlays
    from app_core.production_parlays import build_production_parlays
    from app_core.public_history import selections, original_estimate
    from app_core.relock_changes import compare
    _,package=package_for(monkeypatch)
    rows=package["games"]["overall"]
    assert rows[0]["research_display"]["ev"]>0
    assert ranked_picks(package,NOW)==[]
    assert build_production_parlays(rows,NOW)==build_research_parlays(rows,NOW)==[]
    assert original_estimate(rows[0])=={}
    pub={"package":package,"confirmed_at":NOW.isoformat(),"package_hash":"synthetic-test-only"}
    records=selections([pub])
    assert records and all(r["group"]=="Research" for r in records)
    saved={"id":"saved","legs":[deepcopy(rows[0])]}
    fresh=deepcopy(saved);fresh["legs"][0].pop("research_display")
    assert compare(saved,fresh).change_code=="NO_MATERIAL_CHANGE"
    assert evaluate_release(package,at=NOW)["actionable_row_count"]==0

def test_legacy_packages_and_tampered_display_validate_safely(monkeypatch):
    from app_core.board_diagnostics import build_selected_diagnostics
    frames,package=package_for(monkeypatch)
    legacy=deepcopy(package)
    for rows in legacy["games"].values():
        for row in rows:
            row.pop("research_display",None);row.pop("price_push_probability",None)
            from app_core.price_value_display import display
            row.update(display(row["win_estimate"],row["odds"],row["ev"]))
    legacy.pop("board_diagnostics")
    validate_package(legacy);assert "board-data" in render(legacy)
    broken=deepcopy(package);broken["games"]["overall"][0]["research_display"]["probability"]=True
    with pytest.raises(ValueError):validate_package(broken)

def test_research_display_cannot_supply_subscriber_authority(monkeypatch):
    from paid_launch.case_policy_and_contracts import submission
    from services.subscriber.canonical import object_hash,sign
    from services.subscriber.contracts import ReleaseSubmission
    from integrations.subscriber_release.authority import verify_reviewed_submission,AuthorityError
    _,package=package_for(monkeypatch)
    public=package["games"]["overall"][0]
    assert public["research_display"]["ev"]>0
    raw=submission(NOW)
    raw["authority"]["upstream_gate_result"]=public["status"]
    raw["authority"]["market_status"]="RESEARCH"
    parsed=ReleaseSubmission.model_validate(raw)
    raw["reviewed_payload_hash"]=object_hash(parsed.review_payload())
    with pytest.raises(AuthorityError,match="UPSTREAM_GATE_NOT_APPROVED"):
        verify_reviewed_submission(raw,signature=sign(raw,"synthetic-secret"),
            secret="synthetic-secret",environment="test",now=NOW)
    raw["recommendations"][0]["research_display"]=public["research_display"]
    with pytest.raises(ValueError,match="Extra inputs"):
        ReleaseSubmission.model_validate(raw)

def test_csv_roundtrip_preserves_target_rejection_and_exact_display(monkeypatch):
    from io import StringIO
    for raw,available in [(source(),True),(source(ml_target="home_win"),False)]:
        frames,_=package_for(monkeypatch,raw)
        restored=[pd.read_csv(StringIO(frame.to_csv(index=False))) for frame in frames]
        # CSV authorization contracts remain strings under the legacy loader;
        # research display is independently JSON and never converts them to authority.
        package=build_package(*restored);validate_package(package)
        display=package["games"]["overall"][0]["research_display"]
        assert (display["probability"] is not None)==available
        assert package["games"]["overall"][0]["status"]=="PASS"
        if available:assert display["probability"]==.6
        else:assert display["availability_reason"]=="TARGET_MISMATCH"

@pytest.mark.parametrize("push,semantics,available",[
    (0.0,"win_unconditional_with_push",True),
    (None,"",True),
    (None,"win_unconditional_with_push",False),
    (.1,"win_unconditional_with_push",False),
    (-.1,"win_unconditional_with_push",False),
    (True,"win_unconditional_with_push",False),
    (False,"win_unconditional_with_push",False)])
def test_push_zero_missing_and_inconsistent_actual_serialized_path(monkeypatch,push,semantics,available):
    frames,package=package_for(monkeypatch,source(push_probability=push,probability_semantics=semantics))
    restored=json.loads(assets_from_html(render(package))["board-data.json"]);validate_package(restored)
    d=restored["games"]["overall"][0]["research_display"]
    assert (d["probability"] is not None)==available
    if available:assert d["push_probability"]==0
    else:assert d["ev"] is None
    assert restored["games"]["overall"][0]["win_estimate"] is None
    assert restored["games"]["overall"][0]["ev"] is None
    assert restored["games"]["overall"][0]["wager_contract"]["production_bet_amount"]==0

def test_validator_rejects_inconsistent_saved_push_without_relaxing_price_math(monkeypatch):
    _,package=package_for(monkeypatch)
    package.pop("board_diagnostics")
    package["games"]["overall"][0]["price_push_probability"]=.1
    with pytest.raises(ValueError,match="push probability disagree"):validate_package(package)
    for push in (True,float("nan"),float("inf"),-.1,1.0):
        broken=deepcopy(package);broken["games"]["overall"][0]["price_push_probability"]=push
        with pytest.raises(ValueError):validate_package(broken)
