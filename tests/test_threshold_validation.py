import json

import pandas as pd
import pytest

from core.threshold_validation import compare_thresholds, render_threshold_report, wilson_interval
from test_selector_validation import candidates, final_picks


def compare(frame=None, final=None, **kwargs):
    return compare_thresholds(candidates() if frame is None else frame, train_through="2026-09-02",
                              development_through="2026-09-04", selections=final, bootstrap_repetitions=100, **kwargs)


def aggregate(report, threshold):
    return next(r for r in report["rows"] if r["league"] == "ALL" and r["threshold"] == threshold)


def test_approved_wagers_and_threshold_denominators():
    f = candidates()
    r = compare(f, final_picks(f), thresholds=[.5, .6, .75])
    low = aggregate(r, .6)
    assert low["selector"]["games"] == 1
    assert low["selector"]["coverage"] == .5
    assert low["market_only_same_games"]["losses"] == 1
    assert aggregate(r, .75)["selector"]["games"] == 0
    assert aggregate(r, .75)["selector"]["hit_rate"] is None
    assert r["recommended_threshold"] is None and not r["production_changes"]
    json.dumps(r, allow_nan=False)


def test_unknown_approval_cannot_enter_qualified_scope():
    r = compare(thresholds=[.5])
    assert r["status"] == "no_known_approved_wagers"
    assert aggregate(r, .5)["selector"]["games"] == 0
    assert aggregate(compare(scope="all_selected", thresholds=[.5]), .5)["selector"]["games"] == 2


def test_later_outcomes_and_approvals_cannot_change_development_results():
    frame = candidates()
    final = final_picks(frame)
    expected = compare(frame, final)
    future = frame.iloc[-2:].copy()
    future["matchup_id"] = "future-game"
    for column in ("game_start_utc", "prediction_generated_at", "odds_recorded_at"):
        future[column] = future[column].str.replace("2026-09-04", "2026-09-05")
    future["export_run_id"] = "20260905T160000Z"
    future["candidate_outcome"] = "WIN"
    changed = compare(pd.concat([frame, future], ignore_index=True), final)
    assert expected == changed


def test_repeated_downloads_leave_metrics_unchanged():
    f = candidates()
    assert compare(f, scope="all_selected")["rows"] == compare(pd.concat([f, f], ignore_index=True), scope="all_selected")["rows"]


def test_pushes_are_not_losses_and_intervals_need_observations():
    f = candidates()
    f.loc[f.matchup_id.eq("game-3"), "candidate_outcome"] = "PUSH"
    r = aggregate(compare(f, scope="all_selected", thresholds=[.5]), .5)
    assert r["selector"]["decided"] == 1
    assert r["selector"]["pushes"] == 1
    assert r["selector"]["flat_roi"] == .5
    assert r["hit_rate_wilson_95"][0] < .75
    assert wilson_interval(0, 0) is None


def test_slate_bootstrap_is_deterministic_and_single_slate_is_unavailable():
    f = candidates()
    r = aggregate(compare(f, scope="all_selected", thresholds=[.5]), .5)
    assert r["slate_bootstrap"]["hit_rate_95"] == [1., 1.]
    assert r == aggregate(compare(f, scope="all_selected", thresholds=[.5]), .5)
    f = f[~f.matchup_id.eq("game-4")]
    one = aggregate(compare(f, scope="all_selected", thresholds=[.5]), .5)
    assert one["slate_bootstrap"]["hit_rate_95"] is None


@pytest.mark.parametrize("thresholds", [[], [-.1], [1.1], [float("nan")]])
def test_invalid_thresholds_fail(thresholds):
    with pytest.raises(ValueError, match="Thresholds"):
        compare(thresholds=thresholds)


def test_missing_provenance_yields_explicit_empty_report():
    r = compare(candidates().drop(columns="model_version"), scope="all_selected")
    assert r["status"] == "insufficient_verified_data"
    assert r["rows"] == []
    assert "Input verification" in render_threshold_report(r)


def test_future_only_inputs_are_not_exposed():
    f = candidates()
    for column in ("game_start_utc", "prediction_generated_at", "odds_recorded_at"):
        f[column] = f[column].str.replace("2026-09", "2026-10")
    r = compare(f, scope="all_selected")
    assert r["verification"]["inventory"]["raw_rows"] == 0
    assert r["rows"] == []
