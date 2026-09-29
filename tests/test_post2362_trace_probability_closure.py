"""Post-2362 trace identity and public probability consumer regressions."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import json

import pandas as pd
import pytest

from app_core.current_wagers_trace import build_private_candidate_trace
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from test_current_wagers_trace_and_release import (
    FrozenDateTime,
    _approved_source,
    _package,
    _research_rows,
)
from test_per_game_boards import final
from test_wager_integrity_audit import NOW


def _collision_sources():
    rows = _research_rows(2)
    for row in rows:
        row.update({
            "pick": "Same +1.5",
            "best_pick": "Same +1.5",
            "market_type": "spread_away",
            "spread_line": 1.5,
            "line": 1.5,
            "odds": -190,
            "odds_american": -190,
        })
    return rows


def _trace_candidate(source, **changes):
    quote_time = source["quote_time"]
    event_id = source["matchup_id"]
    row = {
        "candidate_id": source["candidate_id"],
        "canonical_event_id": event_id,
        "matchup_id": event_id,
        "export_run_id": source["export_run_id"],
        "league": source["league"],
        "market_type": source["market_type"],
        "best_pick": source["best_pick"],
        "spread_line": source["spread_line"],
        "odds_american": source["odds_american"],
        "opposing_odds_source": "novig",
        "quote_id": f"quote-{event_id}",
        "quote_observed_at": quote_time,
        "game_start_utc": source["game_start_utc"],
        "provider_event_id": event_id,
        "provider_namespace": "fixture",
        "provider_quotes": json.dumps([{
            "book": "novig",
            "market_type": source["market_type"],
            "point": source["spread_line"],
            "price": source["odds_american"],
            "recorded_at": quote_time,
            "provider_event_id": event_id,
            "provider_namespace": "fixture",
        }]),
        "best_available_selected": False,
        "production_model_eligible": False,
        "production_eligible": False,
        "production_bet_amount": 0.0,
    }
    row.update(changes)
    return row


def _trace(frame, package):
    return build_private_candidate_trace(
        pd.DataFrame(frame), package, evaluated_at=NOW,
        selection_options={"research_fallback": True},
    )


def test_b01_b02_identical_selections_bind_selected_id_to_own_game_position(monkeypatch):
    sources = _collision_sources()
    package = _package(monkeypatch, sources)

    report = _trace([_trace_candidate(sources[1])], package)
    record = report["candidates"][0]

    assert record["output"] == {"section": "overall", "position": 1, "status": "PASS"}
    assert record["output_resolution"]["status"] == "MATCHED"
    assert record["output_resolution"]["selected_identity"]["event_id"] == "research-game-1"


def test_b01_explicit_other_game_never_falls_back_to_same_text_and_price(monkeypatch):
    source = _collision_sources()[0]
    package = _package(monkeypatch, [source])
    candidate = _trace_candidate(
        source,
        candidate_id="other-game-candidate",
        canonical_event_id="other-game",
        matchup_id="other-game",
    )

    record = _trace([candidate], package)["candidates"][0]

    assert record["output"] is None
    assert record["output_resolution"]["status"] == "UNRESOLVED"
    assert record["output_resolution"]["reason"] == "EXPLICIT_CANDIDATE_ID_NOT_SELECTED"
    assert record["stages"]["packaged_output"]["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"canonical_event_id": "doubleheader-2", "matchup_id": "doubleheader-2"}, "event_id"),
        ({"export_run_id": (NOW - timedelta(minutes=1)).isoformat()}, "run_id"),
        ({"opposing_odds_source": "other-book"}, "sportsbook"),
        ({"spread_line": 2.5}, "line"),
        ({"quote_observed_at": (NOW - timedelta(minutes=2)).isoformat()}, "quote_timestamp"),
    ],
)
def test_b03_explicit_identity_conflicts_never_use_display_fallback(
    monkeypatch, changes, field
):
    sources = _collision_sources()
    package = _package(monkeypatch, sources)

    record = _trace([_trace_candidate(sources[1], **changes)], package)["candidates"][0]

    assert record["output"] is None
    assert record["output_resolution"]["status"] == "UNRESOLVED"
    assert field in record["output_resolution"]["reason"]


def test_b04_idless_legacy_requires_complete_event_and_quote_evidence(monkeypatch):
    source = _collision_sources()[0]
    package = _package(monkeypatch, [source])
    exact = _trace_candidate(source, candidate_id="")
    incomplete = dict(exact, canonical_event_id="", matchup_id="")

    matched = _trace([exact], package)["candidates"][0]
    unresolved = _trace([incomplete], package)["candidates"][0]

    assert matched["output_resolution"]["status"] == "MATCHED"
    assert matched["output_resolution"]["reason"] == "EXACT_LEGACY_EVENT_QUOTE_IDENTITY"
    assert unresolved["output"] is None
    assert unresolved["output_resolution"]["status"] == "UNRESOLVED"
    assert "event_id" in unresolved["output_resolution"]["reason"]


def test_b05_rejected_candidate_cannot_borrow_approved_output(monkeypatch):
    package = _package(monkeypatch, [_approved_source()])
    candidate = _trace_candidate(
        _collision_sources()[0],
        candidate_id="rejected-other-candidate",
        production_model_eligible=False,
        production_eligible=False,
        production_bet_amount=0.0,
    )

    record = _trace([candidate], package)["candidates"][0]

    assert record["output"] is None
    assert record["stages"]["model_calibration"]["status"] == "BLOCK"
    assert record["stages"]["currently_usable_wager"]["status"] == "BLOCK"
    assert record["package_actionable_release_allowed"] is True


def test_b10_trace_identity_is_invariant_to_mix_order_and_partition(monkeypatch):
    sources = _collision_sources()
    package = _package(monkeypatch, sources)
    candidates = [_trace_candidate(source) for source in sources]

    mixed = _trace(candidates, package)
    reordered = _trace(list(reversed(candidates)), package)
    partitioned = [_trace([candidate], package) for candidate in candidates]
    expected = {
        row["source_candidate_id"]: (row["trace_id"], row["output"], row["output_resolution"])
        for row in mixed["candidates"]
    }

    assert {
        row["source_candidate_id"]: (row["trace_id"], row["output"], row["output_resolution"])
        for row in reordered["candidates"]
    } == expected
    for report in partitioned:
        row = report["candidates"][0]
        assert (row["trace_id"], row["output"], row["output_resolution"]) == expected[
            row["source_candidate_id"]
        ]


def _probability_consumer(monkeypatch, **changes):
    monkeypatch.setattr("app_core.public_board.datetime", FrozenDateTime)
    values = {
        "export_run_id": NOW.isoformat(),
        "game_time_est": (NOW + timedelta(hours=4)).isoformat(),
        "best_available_selection_policy": "probability-first-v1",
        "best_available_probability": .60,
        "best_available_probability_source": "calibrated_probability",
        "spread_line": -1.5,
    }
    values.update(changes)
    source = final(**values)
    board = pd.DataFrame([source])
    views = [per_game_board(board, family=family)
             for family in ("overall", "sides", "totals")]
    return views[0].iloc[0], build_package(*views)


def test_b06_absent_legacy_half_point_remains_research_compatible(monkeypatch):
    row, package = _probability_consumer(monkeypatch)
    public = package["games"]["overall"][0]

    assert row["status"] == public["status"] == "PASS"
    assert row["win_probability"] == public["win_estimate"] == pytest.approx(.60)
    assert row["ev"] == public["ev"] == pytest.approx(.60 * (1 + 100 / 110) - 1)


def test_b07_unsupported_explicit_semantics_stays_unavailable_and_unapproved(monkeypatch):
    row, package = _probability_consumer(
        monkeypatch,
        probability_semantics="unsupported-explicit",
        push_probability=.10,
    )
    public = package["games"]["overall"][0]

    assert row["status"] == public["status"] == "PASS"
    assert pd.isna(row["win_probability"]) and public["win_estimate"] is None
    assert pd.isna(row["ev"]) and public["ev"] is None
    assert row["probability_basis"] == "Unavailable"


def test_b08_half_point_nonzero_push_is_rejected_by_board_and_package(monkeypatch):
    row, package = _probability_consumer(
        monkeypatch,
        probability_semantics="win_unconditional_with_push",
        push_probability=.10,
    )
    public = package["games"]["overall"][0]

    assert row["status"] == public["status"] == "PASS"
    assert pd.isna(row["win_probability"]) and public["win_estimate"] is None
    assert pd.isna(row["push_probability"])
    assert public["value_status"] == "VALUE UNAVAILABLE"


def test_b09_b11_integer_line_push_values_survive_actual_board_and_package(monkeypatch):
    row, package = _probability_consumer(
        monkeypatch,
        best_pick="Home -2",
        spread_line=-2.0,
        odds_american=100,
        best_available_probability=.575,
        probability_semantics="win_conditional_on_decision",
        push_probability=.10,
        conservative_ev=.09,
    )
    public = package["games"]["overall"][0]
    diagnostic = package["board_diagnostics"]["traces"][0]

    assert row["win_probability"] == public["win_estimate"] == pytest.approx(.5175)
    assert row["push_probability"] == pytest.approx(.10)
    assert 1 - row["win_probability"] - row["push_probability"] == pytest.approx(.3825)
    assert row["price_break_even"] == public["break_even_probability"] == pytest.approx(.45)
    assert row["edge"] == public["estimated_price_edge"] == pytest.approx(.0675)
    assert row["ev"] == public["ev"] == public["estimated_expected_value"] == pytest.approx(.135)
    assert public["conservative_ev"] == pytest.approx(.09)
    assert diagnostic["game_id"] == "g1"
    assert diagnostic["selection"] == "Home -2"
