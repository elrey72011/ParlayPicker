"""Synthetic prospective inputs only; real UI pipeline and export, no network."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from app_core import source_contract as adapter, prediction_evidence as evidence
from app_core.candidate_evidence_schema import project
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package, validate_package
from app_core.research_replay import retain_export, read_export, frame_from_payload
from core import streamlit_pipeline as sp
from core.live_wager_contract import finalize_live_wagers
from scripts.benchmark_drive_history_loading import blocked_network
from scripts.publish_board import render, assets_from_html

NOW = datetime(2026, 10, 6, 19, 40, tzinfo=timezone.utc)
QUOTE = (NOW-timedelta(minutes=16)).isoformat()
INFERENCE = (NOW-timedelta(minutes=14, seconds=10)).isoformat()
CAPTURE = (NOW-timedelta(minutes=14)).isoformat()
START = (NOW+timedelta(hours=3)).isoformat()

class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz or timezone.utc)


@pytest.fixture(autouse=True)
def no_network():
    with blocked_network():
        yield


def fixture(monkeypatch,provider_quote=None):
    # These listing/effective/bridge receipts are SYNTHETIC supplied facts,
    # not a reconstruction of the October 4 run or accepted production listings.
    game = dict(id="synthetic-nfl-source-event", sport_key="americanfootball_nfl",
        matchup_id="AWAY|HOME|2026-10-06", home_team="Home", away_team="Away",
        commence_time=START, bookmakers=[dict(key="novig", last_update="2026-10-06T18:00:00Z",
            markets=[dict(key="spreads", last_update=QUOTE, outcomes=[
                dict(name="Home", point=3.5, price=104, source_contract_ref="synthetic-home"),
                dict(name="Away", point=-3.5, price=-104, source_contract_ref="synthetic-away")])])])
    book = game["bookmakers"][0]; market = book["markets"][0]
    if provider_quote:
        for outcome in market["outcomes"]:outcome["quote_id"]=provider_quote+outcome["name"]
    catalog = {}
    for outcome in market["outcomes"]:
        identity = adapter.identity(game, book, market, outcome)
        receipt = dict(version=adapter.VERSION, identity=identity, documents=adapter.DOCUMENTS.copy(),
            listing_id="synthetic-listing-"+outcome["name"], rule_version=adapter.RULES,
            effective_from="2026-10-05T00:00:00Z", effective_until="2026-10-07T00:00:00Z",
            verified_at="2026-10-05T01:00:00Z", verification_expires="2026-10-07T00:00:00Z",
            period="full_game", overtime=True, count=-outcome["point"], reference_team=identity["selection"],
            comparison="above", bridge_source="synthetic:provider-listing-bridge",
            listing_source="synthetic:original-listing", effective_source="synthetic:effective-terms",
            review_receipt="synthetic:independent-source-review")
        catalog[outcome["source_contract_ref"]] = dict(sha256=adapter.digest(receipt), receipt=receipt)
    monkeypatch.setattr(adapter, "ACCEPTED_LISTINGS", catalog)
    return game, catalog


def pipeline(monkeypatch, game, *, expected_predictions=2):
    from app_core import odds_api, football_identity_capture
    calls=[]
    class Client:
        def __init__(self, **kwargs): pass
        def get_odds(self, sport, date=None):
            calls.append(sport)
            return (deepcopy(game) if isinstance(game,list) else [deepcopy(game)]) if sport == "americanfootball_nfl" else []
    monkeypatch.setattr(odds_api, "TheOddsAPIClient", Client)
    monkeypatch.setattr(odds_api, "datetime", FrozenDateTime)
    monkeypatch.setattr(sp, "_game_date_fallback", lambda:pd.Timestamp("2026-10-06"))
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda:"offline-unused")
    monkeypatch.setattr(sp, "load_base_data", lambda:pd.DataFrame())
    monkeypatch.setattr("app_core.nfl_novig.recover_nfl_novig", lambda games,key:games)
    # Mock only scoreboard transport, keeping ordered event attachment real.
    monkeypatch.setattr(football_identity_capture.requests, "get", lambda *a,**k:SimpleNamespace(
        raise_for_status=lambda:None, json=lambda:{"events":[]}))
    def features(frame,*args):
        out=frame.copy()
        for key,value in dict(League="NFL", feature_home_ppg=24., feature_away_ppg=22.,
                feature_home_oppg=21., feature_away_oppg=23., feature_home_games_played=7,
                feature_away_games_played=7, feature_home_win_pct=.55, feature_away_win_pct=.45,
                feature_diff_last5=.1, ml_feature_eligible=True, stats_resolution_status="resolved").items():
            out[key]=value
        return out
    monkeypatch.setattr("app_core.feature_processing.enrich_with_model_features", features)
    monkeypatch.setattr("app_core.external_data_fetcher.enrich_with_external_data", lambda f:f)
    monkeypatch.setattr(sp, "ML_AVAILABLE", True)
    monkeypatch.setattr(sp, "PredictionEngine", object)
    monkeypatch.setattr(sp, "get_cached_prediction_engine", lambda:SimpleNamespace(use_fallback=False,
        predict_batch=lambda f:[.8]*len(f)))
    monkeypatch.setattr("app_core.research_estimate_trace.generated_time", lambda:INFERENCE)
    monkeypatch.setattr("app_core.candidate_chronology.now_utc", lambda:pd.Timestamp(NOW))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda:None)
    with blocked_network():
        analysis,best,diagnostics = sp.run_analysis_pipeline(sports=["NFL"], use_ml=True, max_rows=20)
    assert calls == ["americanfootball_nfl_preseason", "americanfootball_nfl"]
    assert len(analysis)==2 and diagnostics["market_specific_ml_predictions"]==expected_predictions
    return analysis


def exported(monkeypatch,tmp_path,analysis):
    monkeypatch.setattr(evidence, "now_utc", lambda:CAPTURE)
    monkeypatch.setattr("app_core.public_board.datetime", FrozenDateTime)
    with blocked_network():
        diagnostics={}
        best=sp.build_best_picks_df(analysis, diagnostics_out=diagnostics)
        authority=diagnostics["candidate_authority_df"]
        final,_=finalize_live_wagers(project(authority), best, 1000, now=NOW, policies={}, config={})
        db=tmp_path/"offline.sqlite3";root=tmp_path/"root";root.mkdir(parents=True)
        context=evidence.begin_run({},path=db,root=root)
        captured,card=evidence.capture_run(context,project(authority),final,analysis,path=db,authoritative_candidates=True)
        frames=[per_game_board(card,captured,family=f,novig_only=True) for f in ("overall","sides","totals")]
        package=build_package(*frames);validate_package(package)
        html=render(package)
        preview=json.loads(assets_from_html(html)["board-data.json"]);validate_package(preview)
    return dict(authority=authority,captured=captured,card=card,frames=frames,package=preview,html=html,db=db)


@pytest.mark.parametrize("provider_quote", [None, "synthetic-provider-offer"])
def test_actual_pipeline_complete_contract_capture_rebuild_preview_and_renderer(monkeypatch,tmp_path,provider_quote):
    game,_=fixture(monkeypatch,provider_quote)
    original=deepcopy(game)
    analysis=pipeline(monkeypatch,game)
    result=exported(monkeypatch,tmp_path,analysis)
    row=result["package"]["games"]["overall"][0];display=row["research_display"]
    assert display["availability_reason"]=="AVAILABLE"
    assert display["probability"]==pytest.approx(result["authority"].iloc[0].best_available_probability)
    assert display["ev"] is None and display["value_reason"]=="SETTLEMENT_VALUE_UNSUPPORTED"
    assert display["probability_semantics"]=="win_conditional_on_decision"
    corrupted=deepcopy(display)
    corrupted.update(probability_semantics="win_unconditional_with_push",ev=.1,value_reason="RECORDED_PRICE_VALUE")
    from app_core.research_display import validate
    with pytest.raises(ValueError,match="FVS cannot expose binary settlement value"):
        validate(corrupted)
    assert row["status"]=="PASS" and result["card"].iloc[0].wager_contract["production_bet_amount"]==0
    assert not result["captured"].production_eligible.fillna(False).any()
    assert "synthetic:original-listing" not in json.dumps(result["package"])
    for name in ("authority","captured","card"):
        frame=result[name]
        metadata=json.loads(frame.iloc[0].ml_estimate_metadata)
        assert metadata["producer_contract"]["source_contract"]["status"]=="VERIFIED"
        assert metadata["producer_contract"]["offer"]["quote_kind"]==("provider_issued" if provider_quote else "locally_derived")
        assert frame.iloc[0].prediction_generated_at==INFERENCE
        assert frame.iloc[0].ml_probability != display["probability"]
        assert frame.iloc[0].ml_probability == pytest.approx(metadata["probability"]["value"])
    assert display["identity"]["quote_time"]==pd.Timestamp(QUOTE).isoformat()
    assert display["identity"]["analysis_time"]==pd.Timestamp(INFERENCE).isoformat()
    assert display["identity"]["start"]==pd.Timestamp(START).isoformat()
    with blocked_network():
        receipt=retain_export(result["frames"],result["package"],result["card"],result["captured"],path=result["db"])
        saved,sources=read_export(receipt["export_id"],path=result["db"])
        source=next(iter(sources.values()))
        frames=[per_game_board(frame_from_payload(source["captured_card"]),frame_from_payload(source["captured_candidates"]),family=f,novig_only=True) for f in ("overall","sides","totals")]
        assert build_package(*frames)==saved["package"]
    from test_research_probability_browser import inspect_browser
    browser=inspect_browser(result["package"],tmp_path/"browser",NOW,rendered_html=result["html"])
    assert browser["initial"]["shown"][0]["probability"]==pytest.approx(display["probability"])
    assert browser["initial"]["shown"][0]["ev"] is None
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert original==game
    # Same true inference and blend with no source-adapter metadata; no math change.
    for b in game["bookmakers"]:
        for m in b["markets"]:
            m.update(period="full_game",settlement_rules="includes_overtime")
            for o in m["outcomes"]:o.pop("source_contract_ref")
    unchanged=pipeline(monkeypatch,game)
    assert analysis.ml_probability.tolist()==unchanged.ml_probability.tolist()
    for key in ("calibrated_probability","expected_value"):
        assert analysis[key].tolist()==unchanged[key].tolist()


@pytest.mark.parametrize("field,value,diagnostic",[
    ("period","first_half","SOURCE_PERIOD_CONFLICT"),
    ("rule_version","unverified-v2","SOURCE_RULE_VERSION_UNSUPPORTED"),
    ("effective_until","2026-10-05T00:00:00Z","SOURCE_CONTRACT_STALE_OR_NOT_YET_EFFECTIVE"),
    ("listing_source","","SOURCE_MISSING_LISTING_SOURCE"),
    ("effective_until","2026-10-06T19:25:00Z","SOURCE_CONTRACT_STALE_OR_NOT_YET_EFFECTIVE"),
    ("bookmaker","other","SOURCE_CONFLICT_BOOKMAKER"),
    ("provider_event_id","unrelated","SOURCE_CONFLICT_PROVIDER_EVENT_ID"),
    ("selection","away","SOURCE_CONFLICT_SELECTION"),
    ("line",-3.5,"SOURCE_CONFLICT_LINE"),
    ("price",105,"SOURCE_CONFLICT_PRICE"),
    ("provider_quote_id","other-provider-quote","SOURCE_CONFLICT_PROVIDER_QUOTE_ID"),
])
def test_actual_pipeline_rehashed_wrong_source_facts_reject(monkeypatch,tmp_path,field,value,diagnostic):
    game,catalog=fixture(monkeypatch)
    for accepted in catalog.values():
        receipt=accepted["receipt"]
        if field in receipt["identity"]:receipt["identity"][field]=value
        else:receipt[field]=value
        accepted["sha256"]=adapter.digest(receipt)
    analysis=pipeline(monkeypatch,game)
    result=exported(monkeypatch,tmp_path,analysis)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]!="AVAILABLE" and display["probability"] is None and display["ev"] is None
    trace=json.loads(result["frames"][0].iloc[0].research_estimate_trace)
    assert diagnostic in trace["origin"]["source_contract_diagnostics"]
    assert trace["first_rejection_stage"]==("per_game_export.research_display" if value=="2026-10-06T19:25:00Z" else "quote.source_contract")
    assert result["card"].iloc[0].wager_contract["production_bet_amount"]==0


def test_actual_pipeline_missing_binding_integer_and_started_guards(monkeypatch,tmp_path):
    game,_=fixture(monkeypatch)
    monkeypatch.setattr(adapter,"ACCEPTED_LISTINGS",{})
    analysis=pipeline(monkeypatch,game)
    result=exported(monkeypatch,tmp_path,analysis)
    trace=json.loads(result["frames"][0].iloc[0].research_estimate_trace)
    assert "SOURCE_LISTING_BINDING_NOT_VERIFIED" in trace["origin"]["source_contract_diagnostics"]
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["probability"] is None and display["ev"] is None
    game,catalog=fixture(monkeypatch)
    for accepted in catalog.values():accepted["sha256"]="0"*64
    result=exported(monkeypatch,tmp_path/"corrupt",pipeline(monkeypatch,game))
    assert "SOURCE_RECEIPT_INTEGRITY_FAILURE" in json.loads(result["frames"][0].iloc[0].research_estimate_trace)["origin"]["source_contract_diagnostics"]


def test_named_orientation_provider_clock_and_no_registered_real_listings(monkeypatch):
    game,catalog=fixture(monkeypatch)
    book=game["bookmakers"][0];market=book["markets"][0];outcome=market["outcomes"][0]
    assert adapter.adapt(game,book,market,outcome)["source_contract"]["status"]=="VERIFIED"
    market["last_update"]="2026-10-06T19:25:00Z"
    assert "SOURCE_CONFLICT_SOURCE_TIME" in adapter.adapt(game,book,market,outcome)["source_contract"]["diagnostics"]
    market["last_update"]=QUOTE
    game["home_team"],game["away_team"]=game["away_team"],game["home_team"]
    assert "SOURCE_CONFLICT_HOME" in adapter.adapt(game,book,market,outcome)["source_contract"]["diagnostics"]
    # Even consistently rehashed input keys cannot rewrite accepted named identity.
    game["matchup_id"]="HOME|AWAY|2026-10-06"
    assert adapter.adapt(game,book,market,outcome)["source_contract"]["status"]=="REJECTED"


@pytest.mark.parametrize("mutation,diagnostic", [
    ("integer", "SOURCE_HALF_POINT_REQUIRED"),
    ("duplicate", None),
    ("book", "SOURCE_SCOPE_UNSUPPORTED"),
    ("provider", "SOURCE_SCOPE_UNSUPPORTED"),
    ("period", "SOURCE_TRANSPORT_RULE_PERIOD_CONFLICT"),
    ("rules", "SOURCE_TRANSPORT_RULE_PERIOD_CONFLICT"),
])
def test_actual_pipeline_unsupported_transport_and_duplicate_quote(monkeypatch,tmp_path,mutation,diagnostic):
    game,_=fixture(monkeypatch)
    book=game["bookmakers"][0];market=book["markets"][0]
    if mutation=="integer":
        for outcome in market["outcomes"]:outcome["point"]=3 if outcome["name"]=="Home" else -3
    elif mutation=="duplicate":market["outcomes"].extend(deepcopy(market["outcomes"]))
    elif mutation=="book":book["key"]="fanduel"
    elif mutation=="provider":game["odds_feed_source"]="other_provider"
    elif mutation=="period":market["period"]="first_half"
    elif mutation=="rules":market["settlement_rules"]="other_rules"
    analysis=pipeline(monkeypatch,game,expected_predictions=1 if mutation=="book" else 2)
    from app_core.producer_provenance import diagnose
    from app_core.research_estimate_trace import origin_rejection
    row=analysis.loc[analysis.market_type.eq("spread_home")].iloc[0].to_dict()
    assert origin_rejection(row) is not None
    if diagnostic:
        metadata=json.loads(row["ml_estimate_metadata"])
        if mutation=="book":
            source_quotes=json.loads(row["provider_quotes"])
            assert all(diagnostic in q["source_contract"]["diagnostics"] for q in source_quotes)
        else:
            assert diagnostic in diagnose(row,metadata)["source_contract_diagnostics"]
    if mutation in {"book","provider"}:return  # No Novig public-package row is fabricated.
    if mutation=="duplicate":
        assert json.loads(row["ml_estimate_metadata"])["producer_contract"]["matched_offer_count"]==2
        with pytest.raises(ValueError,match="Cannot capture an empty candidate audit or final card"):
            exported(monkeypatch,tmp_path,analysis)
        return
    result=exported(monkeypatch,tmp_path,analysis)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]!="AVAILABLE" and display["probability"] is None and display["ev"] is None
    assert result["card"].iloc[0].wager_contract["production_bet_amount"]==0


def test_actual_pipeline_started_protection_and_unsupported_nhl_inference(monkeypatch,tmp_path):
    game,_=fixture(monkeypatch)
    game["commence_time"]=(NOW-timedelta(hours=1)).isoformat()
    class FrozenTimestamp(pd.Timestamp):
        @classmethod
        def now(cls,tz=None):return pd.Timestamp(NOW).tz_convert(tz or "UTC")
    class PandasClock:
        Timestamp=FrozenTimestamp
        def __getattr__(self,name):return getattr(pd,name)
    monkeypatch.setattr(sp,"pd",PandasClock())
    analysis=pipeline(monkeypatch,game)
    assert analysis.game_already_started_flag.all()
    diagnostics={}
    best=sp.build_best_picks_df(analysis,diagnostics_out=diagnostics)
    final,_=finalize_live_wagers(project(diagnostics["candidate_authority_df"]),best,1000,now=NOW,policies={},config={})
    assert final.empty
    with pytest.raises(ValueError,match="Cannot capture an empty candidate audit or final card"):
        exported(monkeypatch,tmp_path,analysis)
    assert pd.to_numeric(best["Kelly_Bet_Size"],errors="coerce").fillna(0).eq(0).all()
    from app_core.market_probability_model import predict_market_probabilities
    unsupported=predict_market_probabilities(pd.DataFrame([dict(league="NHL",market_type="spread_home",spread_line=1.5)]))
    assert unsupported.iloc[0].ml_inference_status=="unavailable" and pd.isna(unsupported.iloc[0].ml_probability)


def test_source_market_clock_fallback_and_malformed_capture_fail_closed(monkeypatch):
    game,catalog=fixture(monkeypatch)
    book=game["bookmakers"][0];market=book["markets"][0];outcome=market["outcomes"][0]
    book.pop("last_update")
    assert adapter.adapt(game,book,market,outcome)["source_contract"]["status"]=="VERIFIED"
    market.pop("last_update");book["last_update"]=QUOTE
    assert adapter.adapt(game,book,market,outcome)["source_contract"]["status"]=="VERIFIED"
    book["last_update"]="2026-10-06T18:00:00Z"
    assert "SOURCE_CONFLICT_SOURCE_TIME" in adapter.adapt(game,book,market,outcome)["source_contract"]["diagnostics"]
    captured=adapter.adapt(game,book,market,outcome)["source_contract"]
    captured["receipt"]={}
    assert adapter.replay(captured,INFERENCE)==["SOURCE_CAPTURE_BINDING_CONFLICT","SOURCE_CONFLICT_SOURCE_TIME"]


def test_official_source_hashes_and_no_real_listing_registration():
    from pathlib import Path
    ledger=json.loads((Path(__file__).resolve().parents[1]/"docs/paid-launch/nfl-novig-source-evidence.json").read_text(encoding="utf-8"))
    sources={r["key"]:r for r in ledger["sources"]}
    for field,key in {"provider_markets":"odds-api-markets","provider_v4":"odds-api-v4",
            "provider_books":"odds-api-books","book_nfl_001":"novig-nfl-spread","book_rulebook":"novig-rulebook-v1-3"}.items():
        assert sources[key]["content_sha256"]==adapter.DOCUMENTS[field]
        assert sources[key]["effective_date"]=="UNKNOWN"
        assert sources[key]["retrieval_url"].startswith("https://")
    assert ledger["accepted_real_listings"]==0 and adapter.ACCEPTED_LISTINGS=={}


def test_verification_after_quote_before_inference_is_not_an_inference_clock(monkeypatch):
    game,catalog=fixture(monkeypatch,"synthetic-provider-offer")
    for accepted in catalog.values():
        accepted["receipt"]["verified_at"]="2026-10-06T19:25:00Z"
        accepted["sha256"]=adapter.digest(accepted["receipt"])
    book=game["bookmakers"][0];market=book["markets"][0];outcome=market["outcomes"][0]
    bound=adapter.adapt(game,book,market,outcome)["source_contract"]
    assert bound["status"]=="VERIFIED"
    assert adapter.replay(bound,INFERENCE)==[]
    assert "SOURCE_CONTRACT_STALE_OR_NOT_YET_EFFECTIVE" in adapter.replay(bound,QUOTE)
    assert bound["identity"]["source_time"]==pd.Timestamp(QUOTE).isoformat()


@pytest.mark.parametrize("different_start", [False,True])
def test_actual_pipeline_distinct_events_with_same_unordered_key_reject(monkeypatch,tmp_path,different_start):
    game,catalog=fixture(monkeypatch)
    other=deepcopy(game);other["id"]="synthetic-other-event"
    if different_start:other["commence_time"]=(pd.Timestamp(START)+pd.Timedelta(hours=1)).isoformat()
    for outcome in other["bookmakers"][0]["markets"][0]["outcomes"]:
        receipt=deepcopy(catalog[outcome["source_contract_ref"]]["receipt"])
        outcome["source_contract_ref"]="synthetic-other-"+outcome["name"]
        receipt["identity"]=adapter.identity(other,other["bookmakers"][0],other["bookmakers"][0]["markets"][0],outcome)
        catalog[outcome["source_contract_ref"]]=dict(receipt=receipt,sha256=adapter.digest(receipt))
    analysis=pipeline(monkeypatch,[game,other])
    row=analysis.loc[analysis.market_type.eq("spread_home")].iloc[0].to_dict()
    from app_core.producer_provenance import diagnose
    from app_core.research_estimate_trace import origin_rejection
    metadata=json.loads(row["ml_estimate_metadata"])
    assert row["provider_event_id"]==game["id"]
    assert pd.Timestamp(row["game_start_utc"])==pd.Timestamp(START)
    assert metadata["producer_contract"]["event"]["provider_event_id"]==other["id"]
    assert all(q["source_contract"]["status"]=="VERIFIED" for q in json.loads(row["provider_quotes"]))
    assert origin_rejection(row)=="ESTIMATE_IDENTITY_MISMATCH"
    assert "provider_event_id" in diagnose(row,metadata)["conflicting_fields"]
    if different_start:assert "game_start_utc" in diagnose(row,metadata)["conflicting_fields"]
    result=exported(monkeypatch,tmp_path,analysis)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]=="ESTIMATE_IDENTITY_MISMATCH"
    assert display["probability"] is None and display["ev"] is None
    assert result["card"].iloc[0].wager_contract["production_bet_amount"]==0
