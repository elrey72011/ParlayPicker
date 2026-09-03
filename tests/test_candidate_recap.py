from io import StringIO

import pandas as pd

from app_core.candidate_recap import (
    grade_candidate_audit,
    load_candidate_results_ledger,
    merge_candidate_ledgers,
    persist_candidate_results_ledger,
    summarize_candidate_performance,
    summarize_selected_trend,
)


def _audit_rows():
    base = {
        "pipeline_build": "test-build",
        "league": "MLB",
        "home_team": "Home Club",
        "away_team": "Away Club",
        "game_date": "2026-07-28",
        "matchup_id": "2026-07-28|home club|away club",
        "selection_probability_used": 0.60,
        "expected_value": 0.08,
        "market_family": "side",
        "best_available_selection_verified": True,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "market_type": "spread_home",
                "best_pick": "Home Club -1.5",
                "best_available_rank": 1,
                "best_available_family_rank": 1,
                "best_available_selected": True,
            },
            {
                **base,
                "market_type": "total_over",
                "best_pick": "Over 6.5",
                "best_available_rank": 2,
                "best_available_family_rank": 1,
                "best_available_selected": False,
                "market_family": "total",
            },
            {
                **base,
                "market_type": "spread_away",
                "best_pick": "Away Club +1.5",
                "best_available_rank": 3,
                "best_available_family_rank": 2,
                "best_available_selected": False,
            },
            {
                **base,
                "market_type": "total_under",
                "best_pick": "Under 6.5",
                "best_available_rank": 4,
                "best_available_family_rank": 2,
                "best_available_selected": False,
                "market_family": "total",
            },
        ]
    )


def test_grade_candidate_audit_scores_every_alternative():
    scores = pd.DataFrame(
        [
            {
                "League": "MLB",
                "Home": "Home Club",
                "Away": "Away Club",
                "actual_home_score": 4,
                "actual_away_score": 3,
            }
        ]
    )

    graded = grade_candidate_audit(_audit_rows(), scores)

    assert graded["candidate_outcome"].tolist() == ["LOSS", "WIN", "WIN", "LOSS"]
    assert graded["candidate_graded"].all()
    assert graded["candidate_ledger_key"].nunique() == 4
    assert graded["actual_home_score"].tolist() == [4, 4, 4, 4]


def test_grade_candidate_audit_leaves_postponed_zero_zero_ungraded():
    scores = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Home Club",
                "away_team": "Away Club",
                "actual_home_score": 0,
                "actual_away_score": 0,
            }
        ]
    )

    graded = grade_candidate_audit(_audit_rows(), scores)

    assert graded["candidate_outcome"].eq("N/A").all()
    assert not graded["candidate_graded"].any()


def test_candidate_summaries_expose_rank_and_market_family_signal():
    scores = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Home Club",
                "away_team": "Away Club",
                "actual_home_score": 4,
                "actual_away_score": 3,
            }
        ]
    )
    graded = grade_candidate_audit(_audit_rows(), scores)
    summaries = summarize_candidate_performance(graded)

    rank = summaries["rank"].set_index("Candidate Rank")
    assert rank.loc[1, "Wins"] == 0
    assert rank.loc[1, "Losses"] == 1
    assert rank.loc[2, "Hit Rate"] == 1.0

    family = summaries["market_family"].set_index("Market Family")
    assert family.loc["side", "Wins"] == 1
    assert family.loc["side", "Losses"] == 1
    assert family.loc["total", "Wins"] == 1
    assert family.loc["total", "Losses"] == 1


def test_merge_candidate_ledgers_replaces_duplicate_with_current_grade():
    prior = _audit_rows().iloc[[0]].copy()
    prior["candidate_outcome"] = "N/A"
    current = prior.copy()
    current["candidate_outcome"] = "WIN"

    ledger = merge_candidate_ledgers(current, prior)

    assert len(ledger) == 1
    assert ledger.loc[0, "candidate_outcome"] == "WIN"
    assert bool(ledger.loc[0, "candidate_graded"])


def test_merge_candidate_ledgers_keeps_only_latest_snapshot_per_event():
    prior = _audit_rows().copy()
    prior["export_run_id"] = "20260829T120000Z"
    prior["candidate_outcome"] = "LOSS"
    prior.loc[0, "best_pick"] = "Home Club -2.5"
    prior.loc[2, "best_pick"] = "Away Club +2.5"

    current = _audit_rows().copy()
    current["export_run_id"] = "20260829T130000Z"
    current["candidate_outcome"] = "WIN"
    current["best_available_selected"] = [False, True, False, False]

    ledger = merge_candidate_ledgers(current, prior)

    assert len(ledger) == 4
    assert ledger["export_run_id"].eq("20260829T130000Z").all()
    assert ledger.loc[ledger["best_available_selected"], "best_pick"].tolist() == [
        "Over 6.5"
    ]


def test_candidate_ledger_recovers_runtime_and_multiple_downloads(tmp_path):
    runtime_path = tmp_path / "candidate_results_runtime.csv"
    first = _audit_rows().copy()
    first["candidate_outcome"] = ["WIN", "LOSS", "LOSS", "WIN"]
    assert persist_candidate_results_ledger(first, runtime_path)

    second = _audit_rows().copy()
    second["game_date"] = "2026-07-29"
    second["matchup_id"] = "2026-07-29|home club|away club"
    second["candidate_outcome"] = ["LOSS", "WIN", "WIN", "LOSS"]
    third = _audit_rows().copy()
    third["game_date"] = "2026-07-30"
    third["matchup_id"] = "2026-07-30|home club|away club"
    third["candidate_outcome"] = ["WIN", "LOSS", "WIN", "LOSS"]

    restored = load_candidate_results_ledger(
        runtime_path,
        uploaded=[
            StringIO(second.to_csv(index=False)),
            StringIO(third.to_csv(index=False)),
        ],
    )

    assert restored is not None
    assert len(restored) == 12
    assert set(restored["game_date"]) == {
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
    }
    assert restored["candidate_graded"].all()


def test_candidate_ledger_upload_replaces_runtime_duplicate(tmp_path):
    runtime_path = tmp_path / "candidate_results_runtime.csv"
    prior = _audit_rows().iloc[[0]].copy()
    prior["candidate_outcome"] = "N/A"
    assert persist_candidate_results_ledger(prior, runtime_path)

    current = prior.copy()
    current["candidate_outcome"] = "WIN"
    restored = load_candidate_results_ledger(
        runtime_path,
        uploaded=StringIO(current.to_csv(index=False)),
    )

    assert restored is not None
    assert len(restored) == 1
    assert restored.loc[0, "candidate_outcome"] == "WIN"
    assert bool(restored.loc[0, "candidate_graded"])


def test_candidate_summary_parses_exported_false_string_fail_closed():
    ledger = _audit_rows().iloc[:2].copy()
    ledger["candidate_outcome"] = ["WIN", "LOSS"]
    ledger["best_available_selected"] = ["TRUE", "False"]

    summary = summarize_candidate_performance(ledger)["rank"].set_index(
        "Candidate Rank"
    )

    assert summary["Selected Rows"].sum() == 1


def test_selected_trend_marks_one_slate_as_insufficient_history():
    rows = []
    for index in range(15):
        rows.append(
            {
                "league": "MLB",
                "home_team": f"Home {index}",
                "away_team": f"Away {index}",
                "game_date": "2026-09-01",
                "matchup_id": f"2026-09-01|home {index}|away {index}",
                "export_run_id": "20260901T230644Z",
                "best_available_selected": True,
                "candidate_outcome": "WIN" if index < 7 else "LOSS",
                "selection_probability_used": 0.5628133777456431,
            }
        )

    trend = summarize_selected_trend(pd.DataFrame(rows))

    assert trend["status"] == "INSUFFICIENT_HISTORY"
    assert trend["decisions"] == 15
    assert trend["wins"] == 7
    assert trend["slates"] == 1
    assert round(float(trend["expected_wins"]), 3) == 8.442
    assert 0.24 < float(trend["confidence_interval_low"]) < 0.26
    assert 0.69 < float(trend["confidence_interval_high"]) < 0.71
    assert 0.30 < float(trend["lower_tail_probability"]) < 0.32


def test_selected_trend_flags_large_multi_slate_shortfall():
    rows = []
    for index in range(50):
        slate = index // 10 + 1
        rows.append(
            {
                "league": "MLB",
                "home_team": f"Home {index}",
                "away_team": f"Away {index}",
                "game_date": f"2026-08-{slate:02d}",
                "matchup_id": f"2026-08-{slate:02d}|home {index}|away {index}",
                "export_run_id": f"202608{slate:02d}T120000Z",
                "best_available_selected": True,
                "candidate_outcome": "WIN" if index < 25 else "LOSS",
                "selection_probability_used": 0.80,
            }
        )

    trend = summarize_selected_trend(pd.DataFrame(rows))

    assert trend["status"] == "REGRESSION_SIGNAL"
    assert trend["decisions"] == 50
    assert trend["slates"] == 5
    assert float(trend["lower_tail_probability"]) < 0.05
