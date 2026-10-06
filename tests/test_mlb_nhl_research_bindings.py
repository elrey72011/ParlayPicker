"""Synthetic actual-caller regressions. Retained owner packets are never altered."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from app_core import odds_api, prediction_evidence, research_replay, source_contract
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from core import streamlit_pipeline as sp
from core.team_mapper import normalize_team_name
from scripts.benchmark_drive_history_loading import blocked_network
from test_source_contract_pipeline import exported, FrozenDateTime, INFERENCE, START, QUOTE, NOW
from test_research_probability_browser import inspect_browser


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def pipeline(monkeypatch, *, sport="MLB", book="novig", terms=False, nhl_home="New York Rangers"):
    key="baseball_mlb" if sport=="MLB" else "icehockey_nhl"
    nhl_away="New York Islanders" if nhl_home=="New York Rangers" else "New York Rangers"
    pairs=[("Minnesota Twins","Pittsburgh Pirates"),("Seattle Mariners","Texas Rangers")] if sport=="MLB" else [(nhl_home,nhl_away)]
    games=[]
    for i,(home,away) in enumerate(pairs):
        markets=[dict(key="spreads",last_update=QUOTE,outcomes=[dict(name=home,point=1.5,price=-180),dict(name=away,point=-1.5,price=170)]),
                 dict(key="totals",last_update=QUOTE,outcomes=[dict(name="Over",point=7.5+i,price=102),dict(name="Under",point=7.5+i,price=-104)])]
        if terms:
            for m in markets:m.update(period="full_game",settlement_rules="synthetic:full-game-decided-refund")
        games.append(dict(id="synthetic-event-"+str(i),sport_key=key,home_team=home,away_team=away,
                          matchup_id=home+"|"+away+"|2026-10-06",
                          commence_time=START,bookmakers=[dict(key=book,last_update=QUOTE,markets=markets)]))
    original=deepcopy(games)
    class Client:
        def __init__(self,**k):pass
        def get_odds(self,sport,date=None):return deepcopy(games) if sport==key else []
    monkeypatch.setattr(odds_api,"TheOddsAPIClient",Client)
    monkeypatch.setattr(odds_api,"datetime",FrozenDateTime)
    monkeypatch.setattr(sp,"_game_date_fallback",lambda:pd.Timestamp("2026-10-06"))
    monkeypatch.setattr(sp,"_get_odds_api_key",lambda:"synthetic-unused")
    monkeypatch.setattr(sp,"load_base_data",lambda:pd.DataFrame())
    def features(frame,*a):
        out=frame.copy()
        for k,v in dict(League=sport,feature_home_ppg=4.8,feature_away_ppg=4.1,feature_home_oppg=4.,feature_away_oppg=4.2,
                        feature_home_games_played=20,feature_away_games_played=20,feature_home_win_pct=.55,
                        feature_away_win_pct=.45,feature_diff_last5=.1,ml_feature_eligible=True,
                        stats_resolution_status="resolved").items():out[k]=v
        return out
    monkeypatch.setattr("app_core.feature_processing.enrich_with_model_features",features)
    monkeypatch.setattr("app_core.external_data_fetcher.enrich_with_external_data",lambda f:f)
    monkeypatch.setattr(sp,"ML_AVAILABLE",True)
    monkeypatch.setattr(sp,"PredictionEngine",object)
    monkeypatch.setattr(sp,"get_cached_prediction_engine",lambda:SimpleNamespace(use_fallback=False,predict_batch=lambda f:[.8]*len(f)))
    monkeypatch.setattr("app_core.research_estimate_trace.generated_time",lambda:INFERENCE)
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(NOW))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    analysis,_,diagnostics=sp.run_analysis_pipeline(sports=[sport],use_ml=True,max_rows=20)
    assert games==original
    assert len(analysis)==4*len(pairs)
    return analysis,diagnostics


@pytest.mark.parametrize("terms",[False,True])
def test_two_mlb_totals_reject_unverified_novig_values_capture_export_browser(monkeypatch,tmp_path,terms):
    analysis,diagnostics=pipeline(monkeypatch,terms=terms)
    assert diagnostics["market_specific_ml_predictions"]==8
    for row in analysis.to_dict("records"):
        md=json.loads(row["ml_estimate_metadata"])
        assert md["inference_status"]=="success" and md["probability"]["value"]==row["ml_probability"]
        assert md["producer_contract"]["source_contract"]["status"]=="UNKNOWN"
    saved=exported(monkeypatch,tmp_path,analysis)
    totals=saved["package"]["games"]["totals"]
    assert len(totals)==2
    for row in totals:
        assert row["research_display"]["availability_reason"]=="SOURCE_CONTRACT_NOT_VERIFIED"
        assert all(row.get(k) is None for k in ("win_estimate","ev","model_win_probability","estimated_price_edge","break_even_probability"))
        assert row["status"]=="PASS"
    assert not saved["captured"].production_eligible.fillna(False).any()
    receipt=research_replay.retain_export(saved["frames"],saved["package"],saved["card"],saved["captured"],path=saved["db"])
    replay,sources=research_replay.read_export(receipt["export_id"],path=saved["db"])
    source=next(iter(sources.values()))
    original=research_replay.frame_from_payload(source["original"]["producer"])
    for field in ("ml_probability","calibrated_probability","expected_value","prediction_generated_at","provider_quotes"):
        assert original[field].tolist()==analysis[field].tolist()
    boards=[per_game_board(research_replay.frame_from_payload(source["captured_card"]),research_replay.frame_from_payload(source["captured_candidates"]),family=f,novig_only=True) for f in ("overall","sides","totals")]
    assert build_package(*boards)==replay["package"]
    browser=inspect_browser(replay["package"],tmp_path/"browser",NOW)
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert all(s["probability"] is None and s["ev"] is None for s in browser["initial"]["shown"])
    assert all(c["production_bet_amount"]==0 for c in saved["card"].wager_contract)


def test_complete_synthetic_mlb_contract_displays_identity_bound_research_only(monkeypatch,tmp_path):
    analysis,_=pipeline(monkeypatch,book="draftkings",terms=True)
    saved=exported(monkeypatch,tmp_path,analysis)
    boards=[per_game_board(saved["card"],saved["captured"],family=f,novig_only=True,research_fallback=True) for f in ("overall","sides","totals")]
    package=build_package(*boards)
    rows=package["games"]["totals"]
    assert len(rows)==2
    assert all(r["research_display"]["availability_reason"]=="AVAILABLE" for r in rows), [r["research_display"] for r in rows]
    assert all(r["research_display"]["probability"] is not None and r["status"]=="PASS" for r in rows)
    assert all(c["production_bet_amount"]==0 for c in saved["card"].wager_contract)
    assert not saved["captured"].production_eligible.fillna(False).any()
    public=json.dumps(saved["package"])
    assert all(k not in public for k in ("ml_estimate_metadata","producer_contract","source_dependencies","bytes_base64"))


@pytest.mark.parametrize("name,expected",[("New York Rangers","New York Rangers"),("NY Rangers","New York Rangers"),
    ("New York Islanders","New York Islanders"),("NY Islanders","New York Islanders"),
    ("New York","New York"),("New York Knicks","New York"),("New York Mets","New York Mets"),("New York Jets","New York Jets")])
def test_distinct_names_survive_repeated_normalization(name,expected):
    assert normalize_team_name(name)==expected
    assert normalize_team_name(normalize_team_name(name))==expected


@pytest.mark.parametrize("home,away",[("New York Rangers","New York Islanders"),("New York Islanders","New York Rangers")])
def test_actual_nhl_named_orientation_provider_ids_preserved_without_cover_artifact(monkeypatch,tmp_path,home,away):
    analysis,_=pipeline(monkeypatch,sport="NHL",nhl_home=home)
    assert set(analysis.home_team)=={home}
    assert set(analysis.away_team)=={away}
    assert set(analysis.provider_event_id)=={"synthetic-event-0"}
    assert analysis.ml_probability.isna().all()
    assert not analysis.ml_probability_source.fillna("").str.contains("puck-line").any()
    saved=exported(monkeypatch,tmp_path,analysis)
    row=saved["package"]["games"]["sides"][0]
    assert row["game"]==away+" at "+home
    assert row["research_display"]["probability"] is None and row["status"]=="PASS"
    assert all(c["production_bet_amount"]==0 for c in saved["card"].wager_contract)
