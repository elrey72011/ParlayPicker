import pandas as pd

from app.ui.results_dashboard import _precision_card_mask
from app_core.precision_card import attach_precision_card, precision_shortlist


def _row(name: str, probability: float, odds: int = -110, **overrides):
    row = {
        "best_pick": name,
        "WinProbability": probability,
        "ml_probability": probability,
        "odds_american": odds,
        "final_pick_valid": True,
        "best_available_selection_verified": True,
        "best_available_ranking_verified": True,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "market_line_source": "live",
        "best_available_score": probability,
        "expected_value": 0.03,
        "Bettable": False,
        "Play_Stake": 0.0,
    }
    row.update(overrides)
    return row


def test_precision_card_selects_one_global_top_pick_by_independent_ml_probability():
    source = pd.DataFrame(
        [
            _row("Third", 0.64),
            _row("First", 0.71),
            _row("Second", 0.68),
            _row("Below floor", 0.59),
        ]
    )

    result = attach_precision_card(source)
    shortlist = precision_shortlist(result)

    assert result["Precision_Card"].tolist() == [False, True, False, False]
    assert shortlist["best_pick"].tolist() == ["First"]
    assert shortlist["Precision_Rank"].astype(int).tolist() == [1]
    assert shortlist["Precision_Probability_Source"].tolist() == [
        "INDEPENDENT ML PROBABILITY"
    ]


def test_precision_card_uses_ml_probability_not_composite_selection_score():
    source = pd.DataFrame(
        [
            _row(
                "Composite score is inflated",
                0.68,
                ml_probability=0.59,
                selection_probability_used=0.75,
                best_available_score=0.75,
            ),
            _row(
                "Independent model clears",
                0.61,
                ml_probability=0.61,
                selection_probability_used=0.58,
                best_available_score=0.58,
            ),
        ]
    )

    result = attach_precision_card(source)

    assert result["Precision_Card"].tolist() == [False, True]
    assert result["Precision_Probability"].tolist() == [0.59, 0.61]
    assert "below 60%" in result.loc[0, "Precision_Card_Reason"]


def test_precision_card_uses_composite_score_only_as_a_tiebreaker():
    source = pd.DataFrame(
        [
            _row(
                "Lower tie-breaker",
                0.65,
                ml_probability=0.64,
                best_available_score=0.58,
            ),
            _row(
                "Higher tie-breaker",
                0.65,
                ml_probability=0.64,
                best_available_score=0.63,
            ),
        ]
    )

    result = attach_precision_card(source)

    assert result["Precision_Probability"].tolist() == [0.64, 0.64]
    assert result["Precision_Card"].tolist() == [False, True]


def test_precision_card_requires_final_evidence_corroboration():
    source = pd.DataFrame(
        [
            _row(
                "Raw ML outlier",
                0.65,
                ml_probability=0.65,
                best_available_score=0.52,
            ),
            _row(
                "Corroborated signal",
                0.62,
                ml_probability=0.62,
                best_available_score=0.56,
            ),
        ]
    )

    result = attach_precision_card(source)

    assert result["Precision_Card"].tolist() == [False, True]
    assert not bool(result.loc[0, "Precision_Signal_Corroborated"])
    assert bool(result.loc[1, "Precision_Signal_Corroborated"])
    assert result.loc[1, "Precision_Corroborating_Source"] == (
        "FINAL EVIDENCE SCORE"
    )
    assert "below 55%" in result.loc[0, "Precision_Card_Reason"]


def test_precision_card_fails_closed_without_final_evidence_score():
    source = pd.DataFrame([_row("Raw-only candidate", 0.70)]).drop(
        columns=["best_available_score"]
    )

    result = attach_precision_card(source)

    assert not bool(result.loc[0, "Precision_Card"])
    assert not bool(result.loc[0, "Precision_Signal_Corroborated"])
    assert result.loc[0, "Precision_Corroborating_Source"] == ""
    assert "evidence score is unavailable" in result.loc[0, "Precision_Card_Reason"]


def test_precision_card_fails_closed_without_independent_ml_probability():
    source = pd.DataFrame([
        _row(
            "Incomplete export",
            0.90,
            selection_probability_used=0.90,
            best_available_score=0.90,
        )
    ]).drop(columns=["ml_probability"])

    result = attach_precision_card(source)

    assert pd.isna(result.loc[0, "Precision_Probability"])
    assert result.loc[0, "Precision_Probability_Source"] == ""
    assert not bool(result.loc[0, "Precision_Card"])
    assert "unavailable" in result.loc[0, "Precision_Card_Reason"]


def test_precision_card_fails_closed_on_price_and_verification_gates():
    source = pd.DataFrame(
        [
            _row("Eligible", 0.65, -220),
            _row("Too expensive", 0.80, -225),
            _row("Unverified line", 0.79, line_consistency_flag=False),
            _row("Wrong event", 0.78, line_event_identity_match_flag=False),
            _row("Not live", 0.77, market_line_source="fallback"),
            _row("Started", 0.76, Started=True),
        ]
    )

    result = attach_precision_card(source)

    assert result.loc[result["Precision_Card"], "best_pick"].tolist() == ["Eligible"]
    assert "shorter than -220" in result.loc[1, "Precision_Card_Reason"]
    assert result.loc[2, "Precision_Card_Reason"].startswith("Excluded: final selection")


def test_precision_card_excludes_started_game_from_public_export_status():
    source = pd.DataFrame(
        [
            _row("Started decision", 0.80, Bet_Decision="STARTED"),
            _row("Started tier", 0.79, Play_Tier="started"),
            _row("Live candidate", 0.70, Bet_Decision="BEST AVAILABLE - PASS"),
        ]
    )

    result = attach_precision_card(source)
    shortlist = precision_shortlist(result)

    assert shortlist["best_pick"].tolist() == ["Live candidate"]
    assert not result.loc[0, "Precision_Card"]
    assert not result.loc[1, "Precision_Card"]
    assert result.loc[0, "Precision_Card_Reason"] == "Excluded: game has started."
    assert result.loc[1, "Precision_Card_Reason"] == "Excluded: game has started."


def test_precision_shortlist_never_promotes_an_unfunded_row_to_a_bet():
    source = pd.DataFrame([_row("Research pick", 0.72)])

    result = attach_precision_card(source)

    assert bool(result.loc[0, "Precision_Card"])
    assert not bool(result.loc[0, "Bettable"])
    assert float(result.loc[0, "Play_Stake"]) == 0.0
    assert not bool(result.loc[0, "Precision_Wager_Approved"])
    assert result.loc[0, "Precision_Card_Instruction"] == (
        "PRECISION SHORTLIST - NO APP-APPROVED STAKE"
    )
    assert pd.isna(result.loc[0, "Precision_Target_Hit_Rate"])
    assert "no fixed hit-rate target" in result.loc[0, "Precision_Card_Reason"]


def test_precision_shortlist_recognizes_independently_approved_positive_stake():
    source = pd.DataFrame(
        [_row("Approved", 0.72, Bettable=True, Play_Stake=5.0)]
    )

    result = attach_precision_card(source)

    assert bool(result.loc[0, "Precision_Wager_Approved"])
    assert result.loc[0, "Precision_Card_Instruction"] == "BET - APP APPROVED"


def test_precision_card_missing_verification_columns_is_not_eligible():
    source = pd.DataFrame(
        [{"best_pick": "Incomplete", "WinProbability": 0.90, "odds_american": -110}]
    )

    result = attach_precision_card(source)

    assert not bool(result.loc[0, "Precision_Card"])
    assert not bool(result.loc[0, "Precision_Wager_Approved"])


def test_precision_recap_mask_parses_exported_booleans_fail_closed():
    source = pd.DataFrame(
        {"Precision_Card": [True, "TRUE", "1", "False", None, "unexpected"]}
    )

    assert _precision_card_mask(source).tolist() == [True, True, True, False, False, False]
    assert not _precision_card_mask(pd.DataFrame({"other": [True]})).any()
