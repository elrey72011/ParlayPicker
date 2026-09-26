"""Exact quote timing remains distinct from ranking and wager authority."""
import json

import pandas as pd
import pytest

from app_core.candidate_chronology import assert_integrity, classify, model_authority
from core.parlay_safety import row_is_untrusted
from core.smart_parlay_engine import generate_probability_ranked_parlays, generate_smart_parlays
from core.streamlit_pipeline import build_best_picks_df
from scripts.audit_candidate_chronology import reclassify


AS_OF = pd.Timestamp("2026-09-25T21:33:17Z")


def quote_row(home, away, start, recorded, league, *, odds=-125, line=3.5, source=None):
    quote = dict(book="novig", market_type="spread_home", point=line, price=odds,
                 provider_event_id=home, provider_namespace="odds_api", recorded_at=recorded)
    return dict(league=league, home_team=home, away_team=away,
                game_date="2026-09-25", game_start_utc=start,
                market_type="spread_home", spread_line=line, odds_american=odds,
                odds_source="novig", opposing_odds_source="novig",
                provider_quotes=json.dumps([quote]),
                calibrated_probability=.72, ml_probability=.72, model_probability=.72,
                expected_value=.2, edge=.15, market_probability=.5,
                line_source="live", market_line_source="live", live_spread_line=line,
                is_live_data=True, used_stale_features=False,
                ml_probability_source=source or "", export_run_id="20260925T213317Z",
                best_pick=f"{home} +{line}", best_available_selected=True,
                best_available_finalist=True, final_pick_valid=True,
                final_pick_valid_reason="validated_live_line", wager_approved=False,
                export_role="RANKING CANDIDATE - BACKTEST ONLY")


@pytest.mark.parametrize("home,away,start,recorded,league", [
    ("New York Yankees", "Baltimore", "2026-09-25T20:08:31Z", "2026-09-25T21:32:31Z", "MLB"),
    ("Temple", "Army Black Knights", "2026-09-25T20:07:00Z", "2026-09-25T21:31:46Z", "NCAAF"),
])
def test_reported_post_start_games_become_diagnostics(home, away, start, recorded, league):
    raw = pd.DataFrame([quote_row(home, away, start, recorded, league)])
    corrected, summary = reclassify(raw)
    row = corrected.iloc[0]
    assert row.quote_chronology_status == "POST_START_QUOTE"
    assert row.candidate_context == "POST_START_DIAGNOSTIC"
    assert row.minutes_to_start_at_quote < 0
    assert not row.pregame_quote_valid
    assert not row.final_pick_valid
    assert row.final_pick_valid_reason == "post_start_quote"
    assert not row.best_available_finalist and not row.best_available_selected
    assert row.best_available_rejection_reason == "post_start_quote"
    assert not row.wager_approved
    assert summary["chronology_contradictions"] == summary["semantic_contradictions"] == 0


def test_strict_boundary_missing_times_and_provider_order():
    base = quote_row("Home", "Away", "2026-09-25T22:00:00Z", "2026-09-25T21:59:59Z", "MLB")
    assert classify(base, as_of=AS_OF)["quote_chronology_status"] == "VERIFIED_PREGAME"
    at_start = dict(base, provider_quotes=json.dumps([dict(json.loads(base["provider_quotes"])[0], recorded_at="2026-09-25T22:00:00Z")]))
    assert classify(at_start, as_of=AS_OF)["quote_chronology_status"] == "POST_START_QUOTE"
    after = dict(base, provider_quotes=json.dumps([dict(json.loads(base["provider_quotes"])[0], recorded_at="2026-09-25T22:00:01Z")]))
    assert classify(after, as_of=AS_OF)["quote_chronology_status"] == "POST_START_QUOTE"
    assert classify(dict(base, provider_quotes="", odds_recorded_at=""), as_of=AS_OF)["quote_chronology_status"] == "QUOTE_TIME_MISSING"
    assert classify(dict(base, game_start_utc=""), as_of=AS_OF)["quote_chronology_status"] == "GAME_START_MISSING"
    assert classify(dict(base, selected_quote_recorded_at="2026-09-25T21:59:58Z"), as_of=AS_OF)["quote_chronology_status"] == "PROVIDER_TIME_INCONSISTENT"
    assert classify(dict(base, provider_last_update="2026-09-25T22:00:01Z"), as_of=AS_OF)["quote_chronology_status"] == "PROVIDER_TIME_INCONSISTENT"
    missing_exact_time = dict(base, provider_quotes=json.dumps([dict(json.loads(base["provider_quotes"])[0], recorded_at="")]), odds_recorded_at="2026-09-25T21:59:59Z")
    assert classify(missing_exact_time, as_of=AS_OF)["quote_chronology_status"] == "QUOTE_TIME_MISSING"


def test_pipeline_does_not_select_post_start_positive_ev(monkeypatch):
    monkeypatch.setattr("app_core.candidate_chronology.now_utc", lambda: AS_OF)
    post = quote_row("New York Yankees", "Baltimore", "2026-09-25T20:08:31Z", "2026-09-25T21:32:31Z", "MLB", odds=-186, line=1.5)
    future = quote_row("Future Home", "Future Away", "2026-09-25T23:00:00Z", "2026-09-25T21:32:31Z", "MLB")
    diagnostics = {}
    best = build_best_picks_df(pd.DataFrame([post, future]), diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]
    rejected = audit[audit.home_team.eq("New York Yankees")].iloc[0]
    assert rejected.candidate_context == "POST_START_DIAGNOSTIC"
    assert not rejected.best_available_finalist and not rejected.best_available_selected
    assert not rejected.final_pick_valid and not rejected.wager_approved
    assert len(best) == 1 and best.iloc[0].home_team == "Future Home"
    assert_integrity(audit)


def test_research_models_and_integrity_fail_closed():
    nc = model_authority({"league": "NCAAF", "selection_probability_source": "football_research_blend_no_independent_model"})
    mlb = model_authority({"league": "MLB", "ml_probability_source": "score-distribution-v1:mlb"})
    assert nc["independent_model_available"] is False
    assert nc["probability_authority"] == "RESEARCH_BLEND_ONLY"
    assert mlb["model_scope_status"] == "RESEARCH_MODEL"
    assert mlb["model_validation_status"] == "UNVALIDATED"
    assert mlb["probability_authority"] == "NOT_PRODUCTION_AUTHORITY"
    assert not nc["production_model_eligible"] and not mlb["production_model_eligible"]
    assert row_is_untrusted(dict(candidate_context="POST_START_DIAGNOSTIC", pregame_quote_valid=False))
    assert row_is_untrusted(dict(candidate_context="CURRENT_PREGAME", pregame_quote_valid=True,
                                 production_model_eligible=False))
    with pytest.raises(ValueError, match="CANDIDATE_AUDIT_INTEGRITY_FAILURE"):
        assert_integrity(pd.DataFrame([dict(candidate_context="POST_START_DIAGNOSTIC", pregame_quote_valid=False,
                                            best_available_selected=True, final_pick_valid=True,
                                            final_pick_valid_reason="validated_live_line", wager_approved=True,
                                            production_model_eligible=False, market_validation_status="UNVALIDATED")]))


def test_parlay_builders_reject_post_start_and_research_legs():
    def leg(game):
        return dict(matchup_id=game, best_pick=f"{game} -1.5", league="NFL",
                    Pick_Status="Actionable", production_eligible=True,
                    calibrated_probability=.65, effective_win_probability=.65,
                    market_probability=.52, edge=.05, odds_american=-110,
                    decimal_odds=1.91, consensus_agreement="Agrees",
                    candidate_context="CURRENT_PREGAME", pregame_quote_valid=True,
                    production_model_eligible=True, final_pick_valid=True)

    valid = pd.DataFrame([leg("game-a"), leg("game-b")])
    assert not generate_smart_parlays(valid, calibration=None).empty
    assert not generate_probability_ranked_parlays(valid).empty
    for change in (dict(candidate_context="POST_START_DIAGNOSTIC", pregame_quote_valid=False),
                   dict(production_model_eligible=False),
                   dict(market_validation_status="UNVALIDATED")):
        blocked = valid.copy()
        for field, value in change.items():
            blocked.loc[0, field] = value
        assert generate_smart_parlays(blocked, calibration=None).empty
        assert generate_probability_ranked_parlays(blocked).empty


def test_live_contract_vetoes_bad_candidate_or_final_line_before_funding(tmp_path):
    from activation_fixture import NOW, setup
    from core.live_wager_contract import finalize_live_wagers

    row, policy, config = setup(tmp_path / "ledger.db")
    row.update(candidate_context="CURRENT_PREGAME", pregame_quote_valid=True,
               production_model_eligible=True, market_validation_status="VALIDATED",
               final_pick_valid=True)

    def settle(candidate, final):
        out, _ = finalize_live_wagers(pd.DataFrame([candidate]), pd.DataFrame([final]),
                                      1000, now=NOW, policies={"NFL": policy}, config=config)
        return out.iloc[0]

    good = settle(row, row)
    assert good.production_bet_amount > 0
    for candidate, final in ((dict(row, pregame_quote_valid=False), row),
                             (row, dict(row, final_pick_valid=False)),
                             (row, dict(row, market_validation_status="UNVALIDATED"))):
        blocked = settle(candidate, final)
        assert blocked.production_bet_amount == 0
        assert blocked.wager_contract["production_bet_amount"] == 0
        assert not blocked.wager_approved
