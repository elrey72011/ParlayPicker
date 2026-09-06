"""Synthetic evidence fixtures exercise evaluation contracts, not model quality."""
import json

import pandas as pd
import pytest

from core.selector_validation import KEY, build_report, render_markdown
from scripts.validate_selector import main


def candidates():
    rows = []
    for day in range(1, 5):
        for market, p in [("total_over", .6), ("total_under", .4)]:
            rows.append({
                "export_run_id": f"202609{day:02d}T160000Z", "matchup_id": f"game-{day}",
                "league": "MLB", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                "game_start_utc": f"2026-09-{day:02d}T23:00:00Z",
                "prediction_generated_at": f"2026-09-{day:02d}T15:00:00Z",
                "odds_recorded_at": f"2026-09-{day:02d}T14:59:00Z",
                "model_version": "fixture-v1", "model_trained_through": "2026-08-30T00:00:00Z",
                "model_available_at": "2026-08-31T00:00:00Z",
                "probability_semantics": "win_conditional_on_decision",
                "market_type": market, "best_pick": market + " 8.5",
                "best_available_selected": market == "total_over", "best_available_candidate_count": 2,
                "candidate_outcome": "WIN" if market == "total_over" else "LOSS",
                "calibrated_probability": p, "market_probability": 1 - p,
                "odds_american": 100.0, "odds_source": "draftkings",
                "wager_approved": False,
            })
    return pd.DataFrame(rows)


def final_picks(frame):
    final = frame[frame.best_available_selected][KEY].copy()
    final["wager_approved"] = ["True", "False", "True", "False"]
    return final


def report(frame=None, **kwargs):
    return build_report(candidates() if frame is None else frame, train_through="2026-09-02", **kwargs)


def test_whole_slates_paired_baseline_approval_and_arithmetic():
    frame = candidates()
    r = report(frame, selections=final_picks(frame))
    assert r["inventory"]["eligible_events"] == 2
    assert r["inventory"]["evaluation_days"] == ["2026-09-03", "2026-09-04"]
    all_picks = r["comparisons"]["all_selected"]
    assert all_picks["selector"]["wins"] == 2
    assert all_picks["market_only"]["losses"] == 2
    assert all_picks["selector"]["brier"] == pytest.approx(.16)
    assert all_picks["selector"]["flat_roi"] == 1
    assert all_picks["market_only"]["flat_roi"] == -1
    assert r["comparisons"]["qualified_wagers"]["selector"]["coverage"] == .5
    assert r["comparisons"]["pass_picks"]["selector"]["games"] == 1
    assert r["comparisons"]["approval_unknown"]["selector"]["games"] == 0
    assert r["evidence"]["out_of_sample_independently_verified"] is False
    assert "Calibration" in render_markdown(r)
    json.dumps(r, allow_nan=False)


def test_audit_approval_placeholder_never_becomes_a_pass():
    r = report()
    assert r["comparisons"]["approval_unknown"]["selector"]["games"] == 2
    assert r["comparisons"]["pass_picks"]["selector"]["games"] == 0


def test_exact_duplicate_downloads_do_not_change_results():
    frame = candidates()
    a = report(frame)
    b = report(pd.concat([frame, frame], ignore_index=True))
    assert a["comparisons"] == b["comparisons"]
    assert b["inventory"]["duplicate_rows_removed"] == len(frame)


@pytest.mark.parametrize("column,value,reason", [
    ("prediction_generated_at", "", "missing_or_invalid_prediction_timestamp"),
    ("prediction_generated_at", "2026-09-03T15:00:00", "missing_or_invalid_prediction_timestamp"),
    ("model_version", "", "missing_model_version"),
    ("model_trained_through", "2026-09-03T05:00:00Z", "training_cutoff_unverified_or_leaking"),
    ("model_available_at", "2026-09-03T17:00:00Z", "model_availability_unverified"),
    ("odds_recorded_at", "2026-09-03T17:00:00Z", "odds_timestamp_unverified"),
    ("odds_source", "synthetic", "price_or_source_unverified"),
    ("odds_american", 0, "price_or_source_unverified"),
    ("odds_american", float("inf"), "price_or_source_unverified"),
    ("calibrated_probability", 1.4, "missing_or_invalid_probability"),
    ("probability_semantics", "ranking", "probability_semantics_unverified"),
    ("candidate_outcome", "N/A", "unsettled_or_invalid_outcome"),
])
def test_invalid_candidate_excludes_entire_game(column, value, reason):
    frame = candidates()
    frame.loc[4, column] = value
    r = report(frame)
    assert r["inventory"]["eligible_events"] == 1
    assert reason in {e["reason"] for e in r["exclusions"]}


def test_latest_snapshot_is_chosen_before_checking_outcomes():
    frame = candidates()
    latest = frame[frame.matchup_id.eq("game-3")].copy()
    latest["export_run_id"] = "20260903T170000Z"
    latest.loc[latest.best_available_selected, "candidate_outcome"] = "N/A"
    r = report(pd.concat([frame, latest], ignore_index=True))
    assert r["inventory"]["eligible_events"] == 1
    assert "superseded_pregame_snapshot" in {e["reason"] for e in r["exclusions"]}


def test_postgame_download_does_not_replace_pregame_snapshot():
    frame = candidates()
    late = frame[frame.matchup_id.eq("game-3")].copy()
    late["export_run_id"] = "20260904T000000Z"
    r = report(pd.concat([frame, late], ignore_index=True))
    assert r["inventory"]["eligible_events"] == 2


def test_contradictory_downloads_exclude_game_and_approval_conflicts_fail():
    frame = candidates()
    conflict = frame.iloc[[4]].copy()
    conflict["candidate_outcome"] = "LOSS"
    r = report(pd.concat([frame, conflict], ignore_index=True))
    assert r["inventory"]["eligible_events"] == 1
    final = final_picks(frame)
    other = final.iloc[[2]].copy()
    other["wager_approved"] = "False"
    with pytest.raises(ValueError, match="Conflicting final"):
        report(frame, selections=pd.concat([final, other], ignore_index=True))


def test_pushes_are_not_losses_or_binary_calibration_labels():
    frame = candidates()
    frame.loc[frame.matchup_id.eq("game-3"), "candidate_outcome"] = "PUSH"
    m = report(frame)["comparisons"]["all_selected"]["selector"]
    assert m["wins"] == 1 and m["pushes"] == 1 and m["losses"] == 0
    assert m["flat_roi"] == .5 and m["hit_rate"] == 1
    assert m["probability_n"] == 1


def test_missing_provenance_and_rank_scores_fail_closed():
    frame = candidates().drop(columns="prediction_generated_at")
    r = report(frame)
    assert r["status"] == "insufficient_verified_data"
    assert r["comparisons"]["all_selected"]["selector"]["hit_rate"] is None
    json.dumps(r, allow_nan=False)
    with pytest.raises(ValueError, match="Ranking scores"):
        report(probability_column="selection_probability_used")


def test_market_ties_ignore_outcomes_and_selection_flags():
    frame = candidates()
    frame["market_probability"] = .5
    a = report(frame)
    frame["candidate_outcome"] = frame.candidate_outcome.map({"WIN": "LOSS", "LOSS": "WIN"})
    b = report(frame)
    assert a["comparisons"]["all_selected"]["market_only"]["wins"] == 2
    assert b["comparisons"]["all_selected"]["market_only"]["losses"] == 2


def test_frozen_specification_checks_timing_configuration_and_version():
    base = report()
    spec = {"configuration": base["configuration"], "frozen_at": "2026-09-02T12:00:00Z", "model_version": "fixture-v1"}
    assert report(specification=spec)["evidence"]["preregistered"]
    spec["frozen_at"] = "2026-09-04T12:00:00Z"
    assert not report(specification=spec)["evidence"]["preregistered"]
    spec["frozen_at"] = "2026-09-02T12:00:00Z"
    spec["model_version"] = "different-version"
    assert not report(specification=spec)["evidence"]["preregistered"]


def test_aware_start_assigns_games_to_eastern_slate():
    frame = candidates()
    # Still September 2 Eastern, so this entire game belongs to development.
    frame.loc[frame.matchup_id.eq("game-3"), "game_start_utc"] = "2026-09-03T02:00:00Z"
    assert report(frame)["inventory"]["eligible_events"] == 1


def test_cli_writes_reports_with_hashes_and_rejects_unmatched_glob(tmp_path):
    path = tmp_path / "candidates.csv"
    candidates().to_csv(path, index=False)
    output = tmp_path / "report"
    args = ["report", "--audits", str(path), "--train-through", "2026-09-02", "--output", str(output)]
    assert main(args) == 0
    data = json.loads(output.with_suffix(".json").read_text())
    assert len(data["reproducibility"]["audits"][0]["sha256"]) == 64
    assert output.with_suffix(".md").exists()
    args[2] = str(tmp_path / "missing*.csv")
    with pytest.raises(SystemExit):
        main(args)


def test_freeze_refuses_backdating_and_overwrite(tmp_path):
    path = tmp_path / "spec.json"
    args = ["freeze", "--train-through", "2000-01-01", "--model-version", "fixture-v1", "--output", str(path)]
    with pytest.raises(SystemExit):
        main(args)
    args[2] = "2099-01-01"
    assert main(args) == 0
    with pytest.raises(FileExistsError):
        main(args)


def test_compact_final_export_joins_exact_run_date_and_teams():
    frame = candidates()
    final = frame[frame.best_available_selected].copy()
    final["wager_approved"] = "True"
    final["Local Date"] = final.game_start_utc.str[:10]
    final = final.rename(columns={"home_team": "Home", "away_team": "Away"}).drop(columns="matchup_id")
    r = report(frame, selections=final)
    assert r["comparisons"]["qualified_wagers"]["selector"]["games"] == 2
    final["export_run_id"] = "wrong-run"
    r = report(frame, selections=final)
    assert r["comparisons"]["approval_unknown"]["selector"]["games"] == 2


def test_missing_numeric_evidence_is_excluded_without_crashing():
    frame = candidates().drop(columns="calibrated_probability")
    frame.loc[4, "odds_american"] = pd.NA
    assert report(frame)["status"] == "insufficient_verified_data"


def test_missing_alternative_is_not_treated_as_a_complete_pool():
    r = report(candidates().drop(index=5))
    assert r["inventory"]["eligible_events"] == 1
    assert "candidate_pool_completeness_unverified" in {e["reason"] for e in r["exclusions"]}


def test_doubleheaders_remain_separate_and_alias_duplicates_do_not_count_twice():
    frame = candidates()
    second = frame[frame.matchup_id.eq("game-3")].copy()
    second["matchup_id"] = "game-3-doubleheader"
    second["game_start_utc"] = "2026-09-04T01:00:00Z"
    r = report(pd.concat([frame, second], ignore_index=True))
    assert r["inventory"]["eligible_events"] == 3
    second["game_start_utc"] = "2026-09-03T23:00:00Z"
    r = report(pd.concat([frame, second], ignore_index=True))
    assert r["inventory"]["eligible_events"] == 1


def test_american_odds_returns_and_slate_drawdown():
    frame = candidates()
    frame["odds_american"] = -200.0
    frame.loc[frame.matchup_id.eq("game-4"), "candidate_outcome"] = "LOSS"
    m = report(frame)["comparisons"]["all_selected"]["selector"]
    assert m["flat_profit_units"] == -.5
    assert m["flat_roi"] == -.25
    assert m["max_slate_drawdown_units"] == 1


def test_blank_selection_flag_invalidates_game():
    frame = candidates()
    frame["best_available_selected"] = frame.best_available_selected.astype(object)
    frame.loc[5, "best_available_selected"] = ""
    assert report(frame)["inventory"]["eligible_events"] == 1


def test_cli_protects_input_files_and_rejects_changed_frozen_code(tmp_path):
    path = tmp_path / "candidates.csv"
    candidates().to_csv(path, index=False)
    spec = tmp_path / "frozen.json"
    spec.write_text(json.dumps({"evaluator_sha256": "wrong"}))
    args = ["report", "--audits", str(path), "--train-through", "2026-09-02", "--output", str(tmp_path / "result"), "--specification", str(spec)]
    with pytest.raises(SystemExit):
        main(args)
    # A JSON source specification must never be replaced by the report JSON.
    from scripts.validate_selector import evaluator_identity
    spec.write_text(json.dumps({"evaluator_sha256": evaluator_identity()[0]}))
    args[6] = str(spec)
    before = spec.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert spec.read_bytes() == before
