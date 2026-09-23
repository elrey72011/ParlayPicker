"""Post-capture card and candidate transport through the real public lock board."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import pytest
from app_core import prediction_evidence as evidence
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from app_core.locked_picks import lock_audit, lock_candidates, locked_selections
from streamlit_app import _publication_candidates

AT = "2026-09-15T18:01:00+00:00"

@pytest.fixture(params=["spread_home", "total_under"])
def captured(request, tmp_path, monkeypatch):
    from datetime import datetime
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(AT).astimezone(tz)
    monkeypatch.setattr("app_core.public_board.datetime", FrozenDateTime)
    kind=request.param
    line=-1.5 if kind.startswith("spread") else 8.5
    pick="Chicago Cubs -1.5" if kind.startswith("spread") else "Under 8.5"
    row=dict(candidate_id="candidate-one", matchup_id="game-one", league="MLB",
        home_team="Chicago Cubs", away_team="Pittsburgh Pirates", game_date="2026-09-15",
        game_start_utc="2026-09-15T23:00:00+00:00", game_time_est="2026-09-15 7:00 PM ET",
        export_run_id="20260915T175800.000000Z", market_type=kind, best_pick=pick,
        odds_american=-110, odds_source="Novig", opposing_odds_source="novig",
        calibrated_probability=.55, best_available_selected=True, best_available_rank=1,
        best_available_family_rank=1, best_available_candidate_count=1,
        provider_event_id="provider-one", provider_namespace="odds_api",
        Bettable=False, Play_Stake=0, production_eligible=False, wager_approved=False)
    row["spread_line" if kind.startswith("spread") else "total_line"]=line
    row["provider_quotes"]=json.dumps([dict(book="novig",market_type=kind,point=line,price=-110,
        recorded_at="2026-09-15T17:59:00Z",provider_event_id="provider-one",provider_namespace="odds_api")])
    audit=pd.DataFrame([row])
    # The display card intentionally does not duplicate private raw quote payloads.
    card=audit.drop(columns=["provider_quotes"]).copy()
    root=tmp_path/"root";root.mkdir()
    monkeypatch.setattr(evidence,"now_utc",lambda:"2026-09-15T18:00:00+00:00")
    db=tmp_path/"evidence.db"
    context=evidence.begin_run({},path=db,root=root)
    authority,saved=evidence.capture_run(context,audit,card,audit,path=db,authoritative_candidates=True)
    # main's export keeps provenance quotes but omits family-specific line fields;
    # the authoritative candidates are needed for exact quote lookup.
    saved = saved.drop(columns=["spread_line", "total_line"], errors="ignore")
    return {"candidate_audit_df":audit,"candidate_authority_df":authority},saved


def package(card,candidates,*,overall_overrides=None):
    boards=[per_game_board(card,candidates,f,novig_only=True) for f in ("overall","sides","totals")]
    # Apply simulated pipeline changes before publication so the package's
    # frozen diagnostics and product records describe the same saved rows.
    for field,value in (overall_overrides or {}).items():
        boards[0][field]=value
    return build_package(*boards)


def test_capture_mismatch_and_authoritative_quote_lock(captured):
    diagnostics,card=captured
    old=package(card,diagnostics["candidate_audit_df"])
    assert "No ranked candidate evidence" in old["games"]["overall"][0]["quote_reason"]
    assert lock_audit(old,AT)[0]["Lock status"]=="Quote unavailable"
    selected=_publication_candidates(diagnostics)
    assert selected is diagnostics["candidate_authority_df"]
    for field in ("snapshot_id","export_run_id","candidate_id","matchup_id"):
        assert selected.iloc[0][field]==card.iloc[0][field]
    current=package(card,selected)
    leg=current["games"]["overall"][0]
    assert leg["quote_source"]=="Novig" and leg["quote_time"]
    assert lock_audit(current,AT)[0]["Lock status"]=="Eligible now"
    assert not selected.production_eligible.any()


def test_existing_locks_cannot_be_replaced(captured,tmp_path,monkeypatch):
    from test_public_history import History,Memory
    import app_core.public_history as history
    diagnostics,card=captured
    current=package(card,_publication_candidates(diagnostics))
    monkeypatch.setattr(history,"now",lambda:AT)
    store=History("site-1234","folder",Memory())
    ids=[x["id"] for x in lock_candidates(current,AT)]
    original=store.lock_picks(current,ids)
    assert lock_audit(current,AT,original)[0]["Lock status"]=="Already locked"
    tampered=deepcopy(current)
    tampered["games"]["overall"][0]["odds"]=-115
    with pytest.raises(ValueError,match="Selected-board diagnostics do not match saved game rows"):
        store.lock_picks(tampered,ids)
    assert locked_selections(store.all("locks"))==original
    # A later run quotes a new price for the same game. Build its frozen public
    # package from updated source evidence rather than editing a saved package.
    revised_card=card.copy(deep=True)
    revised_candidates=_publication_candidates(diagnostics).copy(deep=True)
    revised_card.loc[revised_card.index[0],"odds_american"]=-115
    revised_candidates.loc[revised_candidates.index[0],"odds_american"]=-115
    quotes=json.loads(revised_candidates.iloc[0].provider_quotes)
    quotes[0]["price"]=-115
    revised_candidates.loc[revised_candidates.index[0],"provider_quotes"]=json.dumps(quotes)
    changed=package(revised_card,revised_candidates)
    assert changed["games"]["overall"][0]["odds"]==-115
    assert store.lock_picks(changed,ids)==original
    assert locked_selections(store.all("locks"))==original


@pytest.mark.parametrize("change,status",[("started","Started"),("quote","Stale quote"),
    ("analysis","Stale analysis"),("date","Other date")])
def test_existing_time_rules(captured,change,status):
    diagnostics,card=captured
    checked_at = "2026-09-15T23:01:00+00:00" if change == "started" else AT
    overrides={
        "date":{"start":"2026-09-16T23:00:00+00:00"},
        "quote":{"quote_time":"2026-09-15T17:00:00+00:00"},
        "analysis":{"prediction_generated_at":"2026-09-15T17:00:00+00:00"},
    }.get(change)
    current=package(card,_publication_candidates(diagnostics),overall_overrides=overrides)
    assert lock_audit(current,checked_at)[0]["Lock status"]==status
    assert lock_candidates(current,checked_at)==[]


@pytest.mark.parametrize("field,value",[("point",99.5),("price",-120),("book","draftkings"),
    ("provider_event_id","other-event"),("provider_namespace","espn"),("recorded_at","invalid"),("recorded_at","2026-09-15T18:02:00Z")])
def test_bad_quote_still_blocked(captured,field,value):
    diagnostics,card=captured
    candidates=_publication_candidates(diagnostics).copy(deep=True)
    quotes=json.loads(candidates.iloc[0].provider_quotes)
    quotes[0][field]=value
    candidates.loc[candidates.index[0],"provider_quotes"]=json.dumps(quotes)
    current=package(card,candidates)
    assert lock_audit(current,AT)[0]["Lock status"]=="Quote unavailable"
    assert lock_candidates(current,AT)==[]


@pytest.mark.parametrize("authority",[None,pd.DataFrame(),"not a frame",[]])
def test_legacy_fallback(authority):
    legacy=pd.DataFrame([{"candidate_id":"legacy"}])
    assert _publication_candidates({"candidate_authority_df":authority,"candidate_audit_df":legacy}) is legacy
    assert _publication_candidates({"candidate_audit_df":legacy}) is legacy
    assert _publication_candidates({}) is None


def test_every_publish_call_selects_authoritative_frame():
    tree=ast.parse(Path("streamlit_app.py").read_text(encoding="utf-8"))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=="render_publish_panel"]
    assert len(calls)==2
    assert all(ast.unparse(n.args[1])=="_publication_candidates(diagnostics)" for n in calls)


@pytest.mark.parametrize("missing_from", ["candidate", "quote"])
@pytest.mark.parametrize("field", ["provider_event_id", "provider_namespace"])
def test_legacy_quote_identity_absence_is_preserved(captured, missing_from, field):
    diagnostics, card = captured
    candidates = _publication_candidates(diagnostics).copy(deep=True)
    if missing_from == "candidate":
        candidates = candidates.drop(columns=[field])
    else:
        quotes = json.loads(candidates.iloc[0].provider_quotes)
        quotes[0].pop(field)
        candidates.loc[candidates.index[0], "provider_quotes"] = json.dumps(quotes)
    assert lock_audit(package(card, candidates), AT)[0]["Lock status"] == "Eligible now"
