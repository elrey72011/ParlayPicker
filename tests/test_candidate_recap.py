import pandas as pd

from app_core.candidate_recap import (
    grade_candidate_audit,
    merge_candidate_ledgers,
    summarize_candidate_performance,
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
