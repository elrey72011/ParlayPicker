"""Go-live rehearsal, end to end (validation roadmap 2026-11-09, M1). SYNTHETIC store, temp files only.

One NCAAB SPREAD market-skill plan on the Pinnacle model is taken through the whole authority
chain with the production functions: calibration, frozen bound plan, validation and holdout
windows, ``VALIDATION_PASSED``, deployment audit and review, exposure ledger, owner market
activation, decision capture, the app bridge, best-pick construction and the wager contract.

It pins the two bridge defects the audit found (B1: no ``game_start_utc``; B2: the label changed
under the Gemini review) and what the go-live checklist says about the open one (B3: the sport
exposure cap comes from the wager policy, so the stake is $0 without it; owner decision D23).
"""
from datetime import datetime, timedelta, timezone
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from app_core import decision_capture as dc
from app_core import deployment_audit, go_live, qualified_bridge
from app_core import pinnacle_model as pm
from app_core import prospective_calibration as pc
from app_core import prospective_closes as closes
from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from app_core.prospective_validation_plans import freeze_current_validation_plans
from core import exposure_ledger as el
from core import live_wager_contract as lwc
from core.sport_market_activation import activate_market
from core.sport_policy import SportPolicy

SPORT, FAM, BOOK = "NCAAB", "SPREAD", "draftkings"
BASE = datetime(2026, 11, 20, tzinfo=timezone.utc)
PIN_HOME, PIN_AWAY = -130, 108      # Pinnacle makes the home side about 54%; the stake book is -110 both ways
RATE = 25                           # games a day in the plan windows


def _dec(american):
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


class Store:
    """Writes SYNTHETIC games through the production insert functions, on a controlled clock."""

    def __init__(self, path):
        self.path, self.now, self.n = path, BASE, 0

    def game(self, day, *, close=True):
        """Captured 18:00, starts 19:00; certified close snapshot 18:55; final score 22:00."""
        n = self.n
        self.n += 1
        home, away = f"Home U {n}", f"Away St {n}"
        obs = BASE + timedelta(days=day, hours=18)
        start = obs + timedelta(hours=1)

        def market(home_price, away_price):
            return {"key": "spreads", "last_update": obs.isoformat(), "outcomes": [
                {"name": home, "point": -2.5, "price": home_price}, {"name": away, "point": 2.5, "price": away_price}]}
        us, pin = market(-110, -110), market(PIN_HOME, PIN_AWAY)
        pid, eid = f"prov-{n}", f"event-NCAAB-{n}"
        raw = {"id": pid, "sport_key": pe.ODDS_API_SPORT_KEYS[SPORT], "home_team": home, "away_team": away,
               "commence_time": start.isoformat(),
               "bookmakers": [{"key": BOOK, "markets": [us]}, {"key": "pinnacle", "markets": [pin]}]}
        self.now = obs
        pe.insert_event(self.path, dict(
            event_id=eid, sport=SPORT, game_id=pid, provider_namespace="THE_ODDS_API", provider_event_id=pid,
            home_team=home, away_team=away, home_team_id=f"provider:h{n}", away_team_id=f"provider:a{n}",
            scheduled_start=start.isoformat(), observed_at=obs.isoformat(), source_id=f"capture-{pid}", raw_source=raw))
        for book, selection, line, price, source in [(BOOK, home, -2.5, -110, us), (BOOK, away, 2.5, -110, us),
                                                     ("pinnacle", home, -2.5, PIN_HOME, pin),
                                                     ("pinnacle", away, 2.5, PIN_AWAY, pin)]:
            pe.insert_quote(self.path, dict(
                quote_id=f"q-{eid}-{book}-{selection}", event_id=eid, sport=SPORT, market_family=FAM,
                selection=selection, line=line, american_odds=price, decimal_odds=_dec(price), sportsbook=book,
                quote_timestamp=obs.isoformat(), quote_source="THE_ODDS_API", quote_verified=True,
                source_id=f"{pid}:{book}:spreads:{selection}:{obs.isoformat()}", raw_source=source))
        return dict(n=n, eid=eid, pid=pid, home=home, away=away, start=start, obs=obs, close=close)

    def finish(self, g):
        start, n = g["start"], g["n"]
        if g["close"]:      # the stake book closes shorter on the home side: the entry beat the close
            self.now = start + timedelta(minutes=5)
            snap = start - timedelta(minutes=5)
            market = {"key": "spreads", "last_update": (snap - timedelta(minutes=1)).isoformat(), "outcomes": [
                {"name": g["home"], "point": -2.5, "price": (-115, -118, -120, -112)[n % 4]},
                {"name": g["away"], "point": 2.5, "price": -102}]}
            payload = {"timestamp": snap.isoformat(), "previous_timestamp": (snap - timedelta(minutes=5)).isoformat(),
                       "next_timestamp": start.isoformat(),
                       "data": {"id": g["pid"], "sport_key": pe.ODDS_API_SPORT_KEYS[SPORT], "home_team": g["home"],
                                "away_team": g["away"], "commence_time": start.isoformat(),
                                "bookmakers": [{"key": BOOK, "markets": [market]}]}}
            event = pe.load_record(self.path, "prospective_event", g["eid"])
            quotes = [dict(sportsbook=BOOK, market_family=FAM, selection=g[side]) for side in ("home", "away")]
            for row in closes.closes_from_snapshot(event, quotes, payload, self.now):
                pe.insert_certified_close(self.path, row)
        self.now = start + timedelta(hours=3)
        home_score, away_score = (78, 70) if (n * 7) % 12 < 7 else (70, 71)   # home covers -2.5 in 7 of 12 games
        raw = {"id": g["pid"], "sport_key": pe.ODDS_API_SPORT_KEYS[SPORT], "home_team": g["home"],
               "away_team": g["away"], "commence_time": start.isoformat(), "completed": True,
               "last_update": (start + timedelta(hours=2)).isoformat(),
               "scores": [{"name": g["home"], "score": str(home_score)}, {"name": g["away"], "score": str(away_score)}]}
        pe.insert_result(self.path, dict(
            result_id=f"score-{n}", event_id=g["eid"], sport=SPORT, market_family=None, selection=None,
            result_source="THE_ODDS_API", result_source_id=g["pid"], observed_at=self.now.isoformat(),
            available_at=self.now.isoformat(), home_score=home_score, away_score=away_score, outcome=None,
            grading_version=1, raw_source=raw))


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    """The whole chain up to the bridge, built once: every step is the production function."""
    root = tmp_path_factory.mktemp("rehearsal")
    path, ledger, active = root / "SYNTHETIC-evidence.sqlite3", root / "ledger.sqlite3", root / "active_markets"
    active.mkdir()
    store = Store(path)
    patch = pytest.MonkeyPatch()
    patch.setattr(pe, "_clock", lambda: store.now)
    patch.setattr(models, "_utcnow", lambda: store.now)
    patch.setattr(models, "source_commit", lambda: "a" * 40)
    for name, value in {"PARLAYPICKER_MARKET_ACTIVATIONS_DIR": active, "PARLAYPICKER_EXPOSURE_LEDGER": ledger,
                        "PARLAYPICKER_QUALIFIED_MARKET_FILE": "true", "PARLAYPICKER_QUALIFIED_MARKET_STORE": path,
                        "PARLAYPICKER_QUALIFIED_BRIDGE": "true",
                        "PARLAYPICKER_WAGER_POLICY_PATH": root / "no-wager-policy.json"}.items():
        patch.setenv(name, str(value))
    try:
        # Pinnacle reference model, then its calibration from 60 settled games before the plan.
        model = pm.register(path, SPORT, FAM, now=store.now)
        for day in range(3):
            for g in [store.game(day, close=False) for _ in range(20)]:
                store.finish(g)
        store.now = BASE + timedelta(days=3, hours=6)
        pm.settle_reference_games(path, SPORT, FAM, now=store.now)
        pm.fit_calibration(path, SPORT, FAM, model, now=store.now)
        drafted_at = store.now
        store.now = datetime(2026, 10, 1, tzinfo=timezone.utc)
        freeze_current_validation_plans(path, source_commit="b" * 40)
        store.now = drafted_at
        # The reviewed bound plan: market-skill gates, staking terms, default entry and push rules.
        validation_start = BASE + timedelta(days=4)
        spec = pc.draft_bound_plan(path, SPORT, FAM, validation_start=validation_start.isoformat(), rate=RATE,
                                   methodology="market_skill", model_id=model["model_id"],
                                   entry_books=sorted(pm.PLAN_BOOKS), staking_terms="pinnacle-provisional-v1",
                                   entry_rule=None, push_rule=None)
        pc.freeze_reviewed_plan(path, expected_hash=pc.plan_hash(spec), now=drafted_at, **pc.freeze_inputs(spec))
        holdout_end = pe._time(spec["holdout_end"], "holdout_end")
        last_day = 4 + (holdout_end - validation_start).days
        for day in range(4, last_day):
            games = [store.game(day) for _ in range(RATE)]
            store.now = games[0]["obs"] + timedelta(minutes=1)
            pc.predict_bound(path, SPORT, FAM, now=store.now)
            for g in games:
                store.finish(g)
            store.now = games[0]["start"] + timedelta(hours=3, minutes=5)
            pc.settle_scope(path, SPORT, FAM)
        # Verdict, audit, review, bankroll, owner activation.
        store.now = holdout_end + timedelta(hours=1)
        report = pe.evaluate_validation_plan(path, spec["validation_plan_id"])
        audit = deployment_audit.audit(path, spec["validation_plan_id"])
        import deployment_review
        deployment_review.record(path, spec["validation_plan_id"], audit["audit_hash"], "rehearsal-owner")
        state = pe.deployment_state(path, SPORT, FAM)
        el.append(ledger, dict(status="CONFIGURED", bankroll=1000.0, unit_value=10.0, currency="USD", total_cap=0.10,
                               daily_cap=0.05, weekly_cap=0.10, game_cap=0.01, team_cap=0.02),
                  confirmed=True, now=store.now)
        policy = dict(state["deployment_criteria"]["validated_policy"], validation_id=state["validation_id"])
        record = activate_market(state, SportPolicy(**policy), el.snapshot(ledger, now=store.now),
                                 owner_id="rehearsal-owner", expires_at=(store.now + timedelta(days=30)).isoformat(),
                                 confirm=True, now=store.now)
        (active / f"{SPORT}-{FAM}.json").write_text(json.dumps(record), encoding="utf-8")
        # An upcoming game: the bound prediction, then a fresh decision quote and prediction.
        g = store.game(last_day + 1)
        store.now = g["obs"] + timedelta(minutes=1)
        pc.predict_bound(path, SPORT, FAM, now=store.now)
        store.now = fresh = g["obs"] + timedelta(minutes=10)

        def market(home_price, away_price):
            return {"key": "spreads", "last_update": fresh.isoformat(), "outcomes": [
                {"name": g["home"], "point": -2.5, "price": home_price},
                {"name": g["away"], "point": 2.5, "price": away_price}]}
        raw = {"id": g["pid"], "sport_key": pe.ODDS_API_SPORT_KEYS[SPORT], "home_team": g["home"],
               "away_team": g["away"], "commence_time": g["start"].isoformat(), "bookmakers": [
                   {"key": BOOK, "title": "DraftKings", "last_update": fresh.isoformat(), "markets": [market(-110, -110)]},
                   {"key": "pinnacle", "title": "Pinnacle", "last_update": fresh.isoformat(),
                    "markets": [market(PIN_HOME, PIN_AWAY)]}]}
        event = pe.load_record(path, "prospective_event", g["eid"])
        dc.store_quotes(path, SPORT, FAM, {g["pid"]: event}, [raw], clock=lambda: store.now, reference=True)
        store.now = fresh + timedelta(minutes=1)
        dc.predict(path, SPORT, FAM, now=store.now)
        store.now = g["obs"] + timedelta(minutes=12)
        yield dict(path=path, ledger=ledger, active=active, root=root, report=report, audit=audit, state=state,
                   record=record, game=g, now=store.now)
    finally:
        patch.undo()


def test_synthetic_plan_passes_validation_review_and_activation(rehearsal):
    report, state = rehearsal["report"], rehearsal["state"]
    assert report["status"] == "VALIDATION_PASSED" and not report["blockers"]
    assert state["deployment_state"] == "PROVISIONAL_VALIDATED"
    assert rehearsal["record"]["sport"] == SPORT and rehearsal["record"]["market_family"] == FAM


def _settings(rehearsal):
    values = {"PARLAYPICKER_QUALIFIED_BRIDGE": "true", "PARLAYPICKER_DRIVE_FOLDER_ID": "folder",
              "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT": "{...}", "GEMINI_API_KEY": "k",
              "PARLAYPICKER_WAGER_POLICY_PATH": str(rehearsal["root"] / "no-wager-policy.json")}
    return lambda name, default=None: values.get(name, default)


def test_checklist_is_not_ready_while_the_stake_is_zero(rehearsal):
    """R1: the eight chain items all pass, and the full checklist still says a stake is not reachable."""
    kwargs = dict(path=rehearsal["path"], now=rehearsal["now"], setting=_settings(rehearsal),
                  ledger_path=rehearsal["ledger"], activations_dir=rehearsal["active"])
    chain = go_live.checklist(SPORT, FAM, **kwargs)
    assert [i["status"] for i in chain] == ["PASS"] * 8
    items = go_live.full_checklist(SPORT, FAM, **kwargs)
    by_key = {i["key"]: i for i in items}
    assert [i["key"] for i in items][8:] == ["staking_clv", "stake_reachable"]
    assert by_key["staking_clv"]["status"] == "PASS"
    assert by_key["stake_reachable"]["status"] == "ACTION" and "every stake is $0" in by_key["stake_reachable"]["detail"]
    head = go_live.summary(items)
    assert head["ready"] is False and head["next"]["key"] == "stake_reachable"
    assert "Ready to stake" not in go_live.markdown(SPORT, FAM, items)


def _bridged(rehearsal):
    analysis, best, diagnostic = qualified_bridge.inject(pd.DataFrame(), pd.DataFrame(), now=rehearsal["now"])
    assert diagnostic["status"] == "INJECTED" and diagnostic["rows"] == 1
    return analysis, best


def test_bridge_row_survives_best_pick_construction(rehearsal):
    """B1: the app rebuilds best picks from the analysis frame after the bridge; the row must stay."""
    from core.streamlit_pipeline import build_best_picks_df
    analysis, best = _bridged(rehearsal)
    g = rehearsal["game"]
    assert analysis.iloc[0]["game_start_utc"] == g["start"].isoformat()
    rebuilt = build_best_picks_df(analysis, diagnostics_out={})
    assert len(rebuilt) == 1
    assert rebuilt.iloc[0]["best_pick"] == best.iloc[0]["best_pick"] == f"{g['home']} -2.5"


def test_label_matches_the_wager_contract_rewrite_for_whole_number_lines():
    """B2: the contract rewrites the label from ``market_line_used``; the bridge writes the same text."""
    from core.streamlit_pipeline import _format_best_pick
    for market_type, selection, line, family in [("spread_home", "Home U", -3.0, "SPREAD"),
                                                 ("spread_away", "Away St", 2.5, "SPREAD"),
                                                 ("total_over", "Over", 145.0, "TOTAL"),
                                                 ("total_under", "Under", 144.5, "TOTAL")]:
        row = pd.Series({"market_type": market_type, "home_team": "Home U", "away_team": "Away St",
                         "market_line_used": line})
        assert qualified_bridge._label(selection, line, family) == _format_best_pick(row)


def _finalize(rehearsal, **kwargs):
    analysis, best = _bridged(rehearsal)
    reviews = best.copy()                     # the review Gemini returns for the row it was shown
    reviews["gemini_review_status"], reviews["gemini_stake_multiplier"] = "APPROVE", 1
    out, _ = lwc.finalize_live_wagers(analysis.copy(), best.copy(), 1000.0, now=rehearsal["now"], reviews=reviews,
                                      **kwargs)
    assert len(out) == 1
    return out.iloc[0]


def test_matched_review_and_market_policy_reach_a_stake(rehearsal):
    """B2 end to end: the approved review still matches the row, and the validated policy stakes it."""
    _, config, _ = lwc.load_configuration(rehearsal["now"])
    policy = SportPolicy(**rehearsal["record"]["policy"])
    row = _finalize(rehearsal, policies={SPORT: policy}, config=config)
    assert row["selection"] == rehearsal["game"]["home"] and row["gemini_review_status"] == "APPROVE"
    assert row["maturity"] == "PROVISIONAL" and not row["production_gate_reason"]
    assert bool(row["production_eligible"]) is True
    assert 0 < row["production_bet_amount"] <= 1000.0 * policy.provisional_stake_cap


def test_stake_checks_agree_with_the_wager_contract(rehearsal):
    """The checklist's ``stake_reachable`` item and the contract's own stake never disagree."""
    row = _finalize(rehearsal)                # the app's configuration: no wager policy file
    item = go_live.stake_checks(SPORT, FAM, path=rehearsal["path"], now=rehearsal["now"],
                                setting=_settings(rehearsal), ledger_path=rehearsal["ledger"])[1]
    assert (row["production_bet_amount"] > 0) == (item["status"] == "PASS")
