from __future__ import annotations

import math

import pandas as pd
import pytest

from app_core.lean_card import score_best_picks_rows


def _row(candidate_id: str, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "canonical_event_id": candidate_id,
        "quote_id": f"quote-{candidate_id}",
        "league": "NFL",
        "Home": f"Home {candidate_id}",
        "Away": f"Away {candidate_id}",
        "market_type": "spread_home",
        "line": -2.0,
        "effective_win_probability": 0.60,
        "effective_expected_value": 0.20,
        "effective_edge": 0.10,
        "odds_american": 100,
        "Pick_Status": "Actionable",
        "consensus_agreement": "Agrees",
        "best_pick": f"Home {candidate_id} -2",
        "qualified_pick": False,
        "game_already_started_flag": False,
    }
    row.update(updates)
    return row


def _explicit(candidate_id: str, **updates: object) -> dict[str, object]:
    contract = {
        "probability_semantics": "win_conditional_on_decision",
        "push_probability": 0.0,
        "decimal_odds": 2.0,
    }
    contract.update(updates)
    return _row(candidate_id, **contract)


def _score(rows: list[dict[str, object]], *, index: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(rows, index=index)
    return score_best_picks_rows(frame, calibration=None, bucket_stats=None)


def _contract(result: pd.DataFrame, candidate_id: str) -> dict[str, object]:
    row = result.loc[result["Candidate_ID"].eq(candidate_id)].iloc[0]
    return {
        "status": row["Value_Contract_Status"],
        "semantics": row["Probability_Semantics"],
        "p_win": row["Final_P_Win"],
        "p_push": row["Final_P_Push"],
        "p_loss": row["Final_P_Loss"],
        "break_even": row["Price_Break_Even"],
        "edge": row["Absolute_Edge"],
        "mean_ev": row["Mean_EV_Per_Unit"],
        "conservative_ev": row["Conservative_EV_Per_Unit"],
        "quote_id": row["Quote_ID"],
        "calibration_id": row["Calibration_Version"],
    }


def _assert_same_contract(left: dict[str, object], right: dict[str, object]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], float) and math.isnan(left[key]):
            assert isinstance(right[key], float) and math.isnan(right[key])
        elif isinstance(left[key], float):
            assert right[key] == pytest.approx(left[key])
        else:
            assert right[key] == left[key]


def test_a03_legacy_contract_is_unchanged_by_unrelated_explicit_row():
    legacy = _row("legacy")
    alone = _contract(_score([legacy]), "legacy")
    mixed = _contract(_score([legacy, _explicit("explicit")]), "legacy")

    assert alone["status"] == "LEGACY_NO_PUSH_COMPATIBILITY"
    assert alone["p_win"] == pytest.approx(0.60)
    assert alone["p_push"] == pytest.approx(0.0)
    assert alone["p_loss"] == pytest.approx(0.40)
    assert alone["mean_ev"] == pytest.approx(0.20)
    _assert_same_contract(alone, mixed)


def test_a04_a05_explicit_contract_and_optional_bound_are_row_isolated():
    no_bound = _explicit("no-bound")
    with_bound = _explicit(
        "with-bound",
        conservative_probability=0.55,
        conservative_probability_semantics="win_conditional_on_decision",
    )
    alone = _contract(_score([no_bound]), "no-bound")
    mixed = _contract(_score([with_bound, no_bound]), "no-bound")

    assert alone["status"] == "PUSH_AWARE_VERIFIED"
    assert math.isnan(alone["conservative_ev"])
    _assert_same_contract(alone, mixed)


def test_a01_empty_primary_fields_coalesce_case_normalized_contract_aliases():
    aliased = _row(
        "aliased",
        probability_semantics="",
        source_probability_semantics="WIN_CONDITIONAL_ON_DECISION",
        push_probability=None,
        p_push=0.10,
        decimal_odds=None,
        odds_decimal=2.0,
        conservative_probability=None,
        p_win_conservative=0.55,
        p_win_conservative_semantics="WIN_CONDITIONAL_ON_DECISION",
    )

    contract = _contract(_score([aliased]), "aliased")

    assert contract["status"] == "PUSH_AWARE_VERIFIED"
    assert contract["semantics"] == "win_unconditional_with_push"
    assert contract["p_win"] == pytest.approx(0.54)
    assert contract["p_push"] == pytest.approx(0.10)
    assert contract["conservative_ev"] == pytest.approx(0.09)


def test_a06_reorder_partition_and_duplicate_indexes_preserve_each_contract():
    rows = [
        _row("legacy"),
        _explicit(
            "bounded",
            push_probability=0.10,
            conservative_probability=0.55,
            conservative_probability_semantics="win_conditional_on_decision",
        ),
        _explicit("plain"),
    ]
    baseline = _score(rows)
    reordered = _score(list(reversed(rows)))
    duplicate_index = _score(rows, index=[7, 7, 3])

    for candidate_id in ("legacy", "bounded", "plain"):
        expected = _contract(baseline, candidate_id)
        _assert_same_contract(expected, _contract(reordered, candidate_id))
        _assert_same_contract(expected, _contract(duplicate_index, candidate_id))
        singleton = next(
            row for row in rows if row["canonical_event_id"] == candidate_id
        )
        _assert_same_contract(
            expected, _contract(_score([singleton]), candidate_id)
        )


@pytest.mark.parametrize(
    ("updates", "expected_status"),
    [
        ({"probability_semantics": "win_conditional_on_decision", "decimal_odds": 2.0}, "INVALID"),
        ({"probability_semantics": "unknown", "push_probability": 0.0, "decimal_odds": 2.0}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
        ({"probability_semantics": "win_conditional_on_decision", "push_probability": 0.10, "decimal_odds": 2.0, "line": -2.5}, "ILLEGAL_LINE_PUSH_PAIRING"),
        ({"probability_semantics": "win_conditional_on_decision", "push_probability": 0.0, "decimal_odds": 1.90}, "QUOTE_PRICE_MISMATCH"),
    ],
)
def test_a07_invalid_contracts_remain_invalid_in_any_batch(
    updates: dict[str, object], expected_status: str
):
    invalid = _row("invalid", **updates)
    alone = _contract(_score([invalid]), "invalid")
    mixed = _contract(_score([_row("legacy"), _explicit("valid"), invalid]), "invalid")

    assert alone["status"] == expected_status
    assert mixed["status"] == expected_status
