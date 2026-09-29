"""Synthetic release plumbing tests; none of these fixtures grant real authority."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest

from app_core import hosted_board_reconciliation as hosted
from app_core.current_wagers_trace import build_private_candidate_trace
from app_core.release_preflight import (
    ReleasePreflightError, evaluate_release, require_actionable_release,
)
from app_core.public_board import build_package
from core.price_value import price_value
from core.probability_semantics import unconditional_from_conditional
from scripts.publish_board import assets_from_html, render
from scripts.trace_current_wagers import write_artifacts
from test_live_wager_contract import leg
from test_wager_integrity_audit import NOW


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz or timezone.utc)


def _approved_source(**changes):
    public = leg()
    contract = deepcopy(public["wager_contract"])
    row = {
        "candidate_id": "candidate-eligible",
        "matchup_id": contract["matchup_id"],
        "export_run_id": NOW.isoformat(),
        "league": contract["sport"],
        "matchup": "Away0 at Home0",
        "Home": "Home0", "Away": "Away0", "game_date": "2026-09-14",
        "pick": contract["selection"], "best_pick": contract["selection"],
        "market_type": contract["market_type"], "odds": contract["odds"],
        "odds_american": contract["odds"], "status": "APPROVED",
        "Bettable": True, "Play_Stake": contract["production_bet_amount"],
        "win_probability": contract["conservative_probability"],
        "ev": contract["conservative_ev"],
        "game_start_utc": contract["start"],
        "quote_source": contract["sportsbook"],
        "quote_time": contract["quote_timestamp"],
        "qualification_reason": "Passed final wager checks with a positive approved stake",
        "wager_contract": contract,
        "spread_line": contract["line"],
    }
    row.update(changes)
    return row


def _package(monkeypatch, rows=None):
    monkeypatch.setattr("app_core.public_board.datetime", FrozenDateTime)
    frame = pd.DataFrame(rows or [_approved_source()])
    package = build_package(frame.copy(), frame.copy(), frame.copy())
    package["schema_version"] = 5
    package["results"] = []
    return package


def _candidate(source=None, **changes):
    source = source or _approved_source()
    contract = source.get("wager_contract")
    row = {
        "candidate_id": source.get("candidate_id", "candidate-eligible"),
        "canonical_event_id": source["matchup_id"], "matchup_id": source["matchup_id"],
        "export_run_id": source["export_run_id"], "league": source["league"],
        "home_team": source["Home"], "away_team": source["Away"],
        "game_date": source["game_date"], "game_start_utc": source["game_start_utc"],
        "best_pick": source["pick"], "market_type": source["market_type"],
        "spread_line": source["spread_line"], "odds_american": source["odds_american"],
        "opposing_odds_source": str(source["quote_source"]).casefold(),
        "provider_event_id": "provider-event-eligible", "provider_namespace": "fixture",
        "provider_quotes": json.dumps([{
            "book": str(source["quote_source"]).casefold(),
            "market_type": source["market_type"], "point": source["spread_line"],
            "price": source["odds_american"], "recorded_at": source["quote_time"],
            "provider_event_id": "provider-event-eligible", "provider_namespace": "fixture",
        }]),
        "quote_id": "quote-eligible", "quote_observed_at": source["quote_time"],
        "source_predictor_version": "predictor-fixture-v1",
        "Calibration_Consumer_Status": "PRODUCTION_CALIBRATION_APPLIED",
        "Calibration_Version": "calibration-fixture-v1",
        "Calibration_Artifact_SHA256": "a" * 64,
        "production_model_eligible": True, "Production_Gate_Pass": True,
        "Production_Gate_Reason": "qualified", "production_eligible": True,
        "wager_approved": True, "production_bet_amount": source["Play_Stake"],
        "best_available_selected": True,
        "Final_P_Win": .58, "Final_P_Push": 0.0, "Final_P_Loss": .42,
        "Price_Break_Even": 11 / 21, "Mean_EV_Per_Unit": .1072727272727274,
        "P_Win_Conservative": .58, "Conservative_EV_Per_Unit": .1072727272727274,
        "Absolute_Edge": .58 - 11 / 21, "Upstream_Model_EV": .10,
        "Probability_Semantics": "win_unconditional_with_push",
        "wager_contract": contract,
    }
    row.update(changes)
    return row


def _research_rows(count=1):
    rows = []
    for index in range(count):
        rows.append({
            "candidate_id": f"research-{index}", "matchup_id": f"research-game-{index}",
            "export_run_id": NOW.isoformat(), "league": "MLB",
            "matchup": f"Away{index} at Home{index}", "Home": f"Home{index}",
            "Away": f"Away{index}", "game_date": "2026-09-14",
            "pick": f"Away{index} +1.5", "best_pick": f"Away{index} +1.5",
            "market_type": "spread_away", "spread_line": 1.5,
            "odds": -190, "odds_american": -190, "status": "PASS",
            "Bettable": False, "Play_Stake": 0.0, "win_probability": .625,
            "ev": price_value(.625, 0.0, 1 + 100 / 190)["expected_value"],
            "game_start_utc": (NOW + timedelta(hours=3 + index)).isoformat(),
            "quote_source": "Novig", "quote_time": (NOW - timedelta(minutes=1)).isoformat(),
            "qualification_reason": "model EV is not positive",
        })
    return rows


def test_w03_w04_eleven_negative_rows_remain_research_after_expiry(monkeypatch):
    package = _package(monkeypatch, _research_rows(11))
    original = deepcopy(package)
    report = evaluate_release(package, at=NOW + timedelta(minutes=40))

    assert package == original
    assert len(package["games"]["overall"]) == 11
    assert all(row["status"] == "PASS" and row["ev"] < 0 for row in package["games"]["overall"])
    assert report["actionable_row_count"] == 0
    assert report["actionable_release_allowed"]
    assert all(row["current_status"] == "EXPIRED_OR_INVALID_RESEARCH" for row in report["rows"])
    assert report["selected_current_primary_counts"] == {"QUOTE_EXPIRED": 11}
    assert report["selected_current_overlapping_blocker_counts"] == {
        "ANALYSIS_EXPIRED": 11, "QUOTE_EXPIRED": 11,
    }
    assert report["overall_current_primary_counts"] == {"QUOTE_EXPIRED": 11}
    assert report["overall_current_overlapping_blocker_counts"] == {
        "ANALYSIS_EXPIRED": 11, "QUOTE_EXPIRED": 11,
    }
    assert all(trace["producer_primary_reason"] == "UPSTREAM_MODEL_EV_NOT_POSITIVE"
               for trace in package["board_diagnostics"]["traces"])


def test_w04_a_new_clock_does_not_freshen_or_approve_negative_value(monkeypatch):
    package = _package(monkeypatch, _research_rows())
    original = deepcopy(package)
    fresh_observation = evaluate_release(package, at=NOW + timedelta(minutes=1))

    assert fresh_observation["overall_current_primary_counts"] == {"SAVED_RESEARCH_PASS": 1}
    assert fresh_observation["actionable_row_count"] == 0
    assert fresh_observation["rows"][0]["saved_producer_primary_reason"] == "UPSTREAM_MODEL_EV_NOT_POSITIVE"
    assert package == original


def test_w06_w18_fully_eligible_fixture_passes_then_expires(monkeypatch):
    package = _package(monkeypatch)
    fresh = evaluate_release(package, at=NOW + timedelta(minutes=1))
    assert fresh["actionable_row_count"] == 1
    assert fresh["actionable_release_allowed"]
    assert fresh["rows"][0]["remaining_validity_seconds"] > 0

    stale = evaluate_release(package, at=NOW + timedelta(minutes=31))
    assert not stale["actionable_release_allowed"]
    assert stale["blocker_counts"]["QUOTE_EXPIRED"] == 1
    assert stale["blocker_counts"]["ANALYSIS_EXPIRED"] == 1
    with pytest.raises(ReleasePreflightError):
        require_actionable_release(package, at=NOW + timedelta(minutes=31))


@pytest.mark.parametrize("at,reason", [
    (NOW - timedelta(seconds=1), "ANALYSIS_TIME_FUTURE"),
    (datetime.fromisoformat("2026-09-14T18:00:00+00:00"), "GAME_STARTED"),
])
def test_w19_future_and_exact_start_fail_closed(monkeypatch, at, reason):
    package = _package(monkeypatch)
    report = evaluate_release(package, at=at)
    assert not report["actionable_release_allowed"]
    assert reason in report["blocker_counts"]


def test_w20_price_change_never_inherits_frozen_authority(monkeypatch):
    package = _package(monkeypatch)
    changed = deepcopy(package)
    for section in changed["games"].values():
        section[0]["odds"] = -115.0
    report = evaluate_release(changed, at=NOW + timedelta(minutes=1), validate=False)
    assert not report["actionable_release_allowed"]
    assert report["blocker_counts"]["PRICE_CHANGED"] == 1
    assert package["games"]["overall"][0]["odds"] == -110.0


def test_w20_changed_line_and_w22_expired_current_authority_block(monkeypatch):
    package = _package(monkeypatch)
    changed = deepcopy(package)
    for section in changed["games"].values():
        section[0]["wager_contract"]["line"] = -3.5
    line_report = evaluate_release(changed, at=NOW + timedelta(minutes=1), validate=False)
    assert line_report["blocker_counts"]["LINE_CHANGED"] == 1

    fresh = evaluate_release(package, at=NOW + timedelta(minutes=1))
    row_id = fresh["rows"][0]["row_id"]
    authority_report = evaluate_release(
        package, at=NOW + timedelta(minutes=1),
        current_authority={row_id: {
            "status": "APPROVED", "withdrawn": False,
            "expires_at": NOW.isoformat(),
        }},
    )
    assert not authority_report["actionable_release_allowed"]
    assert authority_report["blocker_counts"]["CURRENT_AUTHORITY_EXPIRED"] == 1


def test_w05_numeric_value_without_authority_is_not_a_wager(monkeypatch):
    package = _package(monkeypatch, _research_rows())
    candidate = _candidate(_approved_source(), wager_contract=None,
                           production_model_eligible=False,
                           Calibration_Consumer_Status="CALIBRATION_REJECTED",
                           production_eligible=False, wager_approved=False,
                           production_bet_amount=0.0)
    candidate["best_pick"] = package["games"]["overall"][0]["pick"]
    candidate["market_type"] = package["games"]["overall"][0]["market"]
    candidate["odds_american"] = package["games"]["overall"][0]["odds"]
    report = build_private_candidate_trace(
        pd.DataFrame([candidate]), package, evaluated_at=NOW,
        selection_options={"nfl_fallback": True, "research_fallback": True},
    )
    assert report["funnel"]["currently_usable_wager"]["candidate_count"] == 0
    assert report["candidates"][0]["stages"]["model_calibration"] == {
        "status": "BLOCK", "reason": "MODEL_NOT_PRODUCTION_ELIGIBLE", "executed": True,
    }
    assert report["candidates"][0]["stages"]["authority_review"]["status"] == "BLOCK"


def test_w06_w10_w11_private_trace_maps_eligible_candidate_and_deduplicates(monkeypatch):
    source = _approved_source()
    package = _package(monkeypatch, [source])
    candidate = _candidate(source)
    frame = pd.DataFrame([candidate, deepcopy(candidate)], index=[7, 7])
    report = build_private_candidate_trace(
        frame, package, evaluated_at=NOW, current_at=NOW + timedelta(minutes=1),
        selection_options={"nfl_fallback": True},
    )
    assert report["all_market_candidate_count"] == 1
    assert report["funnel"]["raw_candidate_rows"]["count"] == 2
    record = report["candidates"][0]
    assert record["exact_duplicate_count"] == 2
    assert record["output"] == {"section": "overall", "position": 0, "status": "APPROVED"}
    assert all(record["stages"][stage]["status"] == "PASS" for stage in (
        "identity_market", "pregame_fresh_quote", "model_calibration", "price_gate",
        "authority_review", "finalist_selection", "packaged_output",
        "release_preflight", "currently_usable_wager",
    ))


def test_w10_distinct_quote_book_price_and_selection_do_not_collapse(monkeypatch):
    source = _approved_source()
    package = _package(monkeypatch, [source])
    rows = [
        _candidate(source),
        _candidate(source, quote_id="quote-two"),
        _candidate(source, opposing_odds_source="other-book"),
        _candidate(source, odds_american=-108),
        _candidate(source, best_pick="Home0 +2.5", spread_line=2.5),
    ]
    report = build_private_candidate_trace(
        pd.DataFrame(rows), package, evaluated_at=NOW,
        selection_options={"nfl_fallback": True},
    )
    assert report["all_market_candidate_count"] == len(rows)
    assert report["funnel"]["deduplicated_candidates"]["count"] == len(rows)
    assert len(set(report["funnel"]["deduplicated_candidates"]["candidate_ids"])) == len(rows)


def test_w13_w14_shared_push_value_examples():
    negative = price_value(.625, 0.0, 1 + 100 / 190)
    assert negative["expected_value"] == pytest.approx(-.04605263157894735)
    assert negative["edge"] < 0

    mass = unconditional_from_conditional(.575, .10)
    mean = price_value(mass["p_win"], mass["p_push"], 2.0)
    conservative_mass = unconditional_from_conditional(.55, .10)
    conservative = price_value(conservative_mass["p_win"], conservative_mass["p_push"], 2.0)
    assert mass == pytest.approx({"p_win": .5175, "p_push": .10, "p_loss": .3825})
    assert mean["break_even"] == pytest.approx(.45)
    assert mean["edge"] == pytest.approx(.0675)
    assert mean["expected_value"] == pytest.approx(.135)
    assert conservative["expected_value"] == pytest.approx(.09)


def test_w23_publication_boundaries_recheck_without_writes(monkeypatch, tmp_path):
    package = _package(monkeypatch)
    stale_at = NOW + timedelta(minutes=31)
    monkeypatch.setattr("app_core.release_preflight.datetime", type("Clock", (datetime,), {
        "now": classmethod(lambda cls, tz=None: stale_at.astimezone(tz or timezone.utc))
    }))
    from scripts.publish_board import publish_package
    from app_core import netlify_publishing, sftp_publishing
    with pytest.raises(ReleasePreflightError):
        publish_package(package, tmp_path / "site")
    with pytest.raises(ReleasePreflightError):
        sftp_publishing.prepare(package)
    called = []
    monkeypatch.setattr(netlify_publishing, "api_call", lambda *args, **kwargs: called.append(args))
    with pytest.raises(ReleasePreflightError):
        netlify_publishing.deploy(package, "site-1234", "unused-test-token")
    assert called == []
    assert not (tmp_path / "site").exists()


def test_w24_hosted_parity_still_rejects_expired_actionable_content(monkeypatch):
    package = _package(monkeypatch)
    encoded = assets_from_html(render(package, live=True))
    assets = {name: encoded[name].encode("utf-8") for name in hosted.ASSETS}
    stale_at = NOW + timedelta(minutes=31)
    monkeypatch.setattr("app_core.hosted_board_reconciliation.datetime", type(
        "HostedClock", (datetime,), {
            "now": classmethod(lambda cls, tz=None: stale_at.astimezone(tz or timezone.utc))
        },
    ))
    with pytest.raises(hosted.HostedMismatch, match="HOSTED_ACTIONABLE_CONTENT_EXPIRED"):
        hosted.verify_assets(assets)


def test_w08_trace_cli_writes_sanitized_complete_candidate_artifact(monkeypatch, tmp_path):
    package = _package(monkeypatch)
    package_path = tmp_path / "board-data.json"
    candidate_path = tmp_path / "candidate-audit.csv"
    output = tmp_path / "private-audit"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    pd.DataFrame([_candidate()]).to_csv(candidate_path, index=False)

    report = write_artifacts(
        candidate_path, package_path, output,
        evaluated_at=NOW.isoformat(), current_at=(NOW + timedelta(minutes=1)).isoformat(),
    )

    assert report["trace_status"] == "RECORDED_CANDIDATE_AUDIT"
    trace = (output / "current-wagers-candidate-trace.jsonl").read_text(encoding="utf-8")
    assert len(trace.splitlines()) == 1
    assert "candidate-eligible" in trace
    assert "unused-test-token" not in trace
    manifest = json.loads((output / "source-runtime-publication-manifest.json").read_text())
    assert manifest["historical_reconstruction_complete"] is True
    assert manifest["candidate_sha256"]


def test_w28_selected_only_replay_keeps_candidate_count_unknown(monkeypatch):
    package = _package(monkeypatch, _research_rows())
    report = build_private_candidate_trace(None, package, evaluated_at=NOW)
    assert report["trace_status"] == "HISTORICAL_CANDIDATES_UNAVAILABLE"
    assert report["all_market_candidate_count"] is None
    assert "not supplied" in report["all_market_candidate_count_unavailable_reason"]
    assert report["funnel"]["parsed_candidate"]["candidate_count"] is None
    assert report["funnel"]["packaged_output"]["output_row_count"] == 1
