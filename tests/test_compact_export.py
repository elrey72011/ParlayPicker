import pandas as pd

from streamlit_app import (
    _build_compact_export_frame,
    _compact_summary_pnl_formula,
    _compact_total_amount_formula,
    _compact_win_amount_formula,
)


def _compact_source(**overrides):
    row = {
        "pipeline_build": "2026-08-01c-compact-export-stake-safety",
        "export_run_id": "20260816T145118Z",
        "Wager_Instruction": "DO NOT BET - $0 PASS / RESEARCH",
        "Export_Scope": "COVERAGE / RESEARCH",
        "Bettable": False,
        "Play_Stake": 0.0,
        "Play_Tier": "AVOID",
        "Pick_Status": "No Edge",
        "league": "MLB",
        "Home": "Chicago Cubs",
        "Away": "New York Yankees",
        "Commence (Local)": "2026-08-01 7:16 PM ET",
        "odds_american": -220.0,
        "odds_source": "odds_api",
        "market_line_source_detail": "fanduel_standard_spread_consensus",
        "best_pick": "Chicago Cubs +1.5",
        "Kelly_Bet_Size": 1000.0,
    }
    row.update(overrides)
    return row


def test_compact_export_zeroes_stale_kelly_for_non_bettable_rows():
    source = pd.DataFrame([
        _compact_source(),
        _compact_source(
            Home="Houston",
            Away="Texas",
            best_pick="Houston +1.5",
            Pick_Status="Near Miss",
            Bettable="False",
            Kelly_Bet_Size=2500.0,
        ),
        _compact_source(
            Home="Chicago",
            Away="Las Vegas",
            best_pick="Las Vegas line unresolved",
            Bettable=True,
            Play_Stake=25.0,
            Kelly_Bet_Size=25.0,
        ),
    ])

    compact = _build_compact_export_frame(source)

    assert compact["Kelly_Bet_Size"].tolist() == [0.0, 0.0, 0.0]
    assert compact["Play_Stake"].tolist() == [0.0, 0.0, 0.0]
    assert compact["pipeline_build"].eq(
        "2026-08-01c-compact-export-stake-safety"
    ).all()
    assert compact["export_run_id"].eq("20260816T145118Z").all()


def test_compact_export_retains_only_explicitly_approved_stake():
    source = pd.DataFrame([
        _compact_source(
            Wager_Instruction="APPROVED: wager the exported Play_Stake amount.",
            Export_Scope="PRODUCTION WAGERS ONLY",
            Bettable=True,
            Play_Tier="BET",
            Play_Stake=12.5,
            Kelly_Bet_Size=18.75,
        )
    ])

    compact = _build_compact_export_frame(source)

    assert compact.loc[0, "Play_Stake"] == 12.5
    assert compact.loc[0, "Kelly_Bet_Size"] == 18.75
    assert compact.loc[0, "Play_Tier"] == "BET"


def test_compact_export_preserves_controlled_value_disclosure():
    source = pd.DataFrame([_compact_source(
        Wager_Instruction="BET - CONTROLLED VALUE CARD / SMALL STAKE / NOT PREMIUM",
        Export_Scope="CONTROLLED VALUE BET",
        Bettable=True,
        Play_Tier="BET",
        Play_Stake=5.0,
        Kelly_Bet_Size=5.0,
        commercial_tier="Controlled Value Pick",
        sellable_as_premium=False,
        sellable_as_value_card=True,
        controlled_card_recovery=True,
    )])

    compact = _build_compact_export_frame(source)

    assert compact.loc[0, "commercial_tier"] == "Controlled Value Pick"
    assert not bool(compact.loc[0, "sellable_as_premium"])
    assert bool(compact.loc[0, "sellable_as_value_card"])
    assert bool(compact.loc[0, "controlled_card_recovery"])
    assert compact.loc[0, "Play_Stake"] == 5.0


def test_compact_export_reconciles_unfunded_qualified_bet_tier_to_lean():
    source = pd.DataFrame([
        _compact_source(
            Wager_Instruction="DO NOT BET - QUALIFIED LEAN HAS NO APPROVED STAKE",
            Export_Scope="QUALIFIED LEAN / RESEARCH",
            Bettable=False,
            Play_Tier="BET",
            Play_Stake=2.0,
        )
    ])

    compact = _build_compact_export_frame(source)

    assert compact.loc[0, "Play_Tier"] == "LEAN"
    assert float(compact.loc[0, "Play_Stake"]) == 0.0


def test_compact_export_keeps_truthful_price_provenance():
    source = pd.DataFrame([
        _compact_source(),
        _compact_source(
            Home="Phoenix",
            Away="New York",
            best_pick="Under 176.5",
            market_line_source_detail="",
            odds_source="novig",
            odds_american=-135.0,
        ),
    ])

    compact = _build_compact_export_frame(source)

    assert "Pick Price" in compact.columns
    assert "Price Source" in compact.columns
    assert "odds_source" not in compact.columns
    assert "market_line_source_detail" not in compact.columns
    assert compact["Pick Price"].tolist() == [-220.0, -135.0]
    assert compact["Price Source"].tolist() == [
        "fanduel_standard_spread_consensus",
        "novig",
    ]


def test_compact_export_prefers_ranked_pick_and_keeps_display_fallback():
    source = pd.DataFrame([
        _compact_source(
            best_pick="Under 11.5",
            display_pick="NO QUALIFIED PICK",
            Export_Scope="NO QUALIFIED PICK / RESEARCH",
            Wager_Instruction=(
                "DO NOT BET - NO CANDIDATE CLEARS THE QUALIFIED-PICK GATE"
            ),
        ),
        _compact_source(
            Home="Philadelphia",
            Away="Washington",
            best_pick="Washington +1.5",
            display_pick="Washington +1.5",
        ),
        _compact_source(
            Home="Milwaukee",
            Away="Pittsburgh",
            best_pick="Pittsburgh +1.5",
            display_pick="",
        ),
    ])

    compact = _build_compact_export_frame(source)

    assert compact["best_pick"].tolist() == [
        "Under 11.5",
        "Washington +1.5",
        "Pittsburgh +1.5",
    ]
    assert "display_pick" not in compact.columns


def test_compact_export_preserves_precision_shortlist_disclosure():
    source = pd.DataFrame([
        _compact_source(
            Precision_Card=True,
            Precision_Rank=1,
            Precision_Probability=0.71,
            Precision_Probability_Source="INDEPENDENT ML PROBABILITY",
            Precision_Target_Hit_Rate=pd.NA,
            Precision_Wager_Approved=False,
            Precision_Card_Instruction=(
                "PRECISION SHORTLIST - NO APP-APPROVED STAKE"
            ),
            Precision_Card_Reason=(
                "Selected by global independent ML probability for research monitoring; "
                "no fixed hit-rate target is claimed."
            ),
        )
    ])

    compact = _build_compact_export_frame(source)

    assert bool(compact.loc[0, "Precision_Card"])
    assert compact.loc[0, "Precision_Rank"] == 1
    assert compact.loc[0, "Precision_Probability"] == 0.71
    assert compact.loc[0, "Precision_Probability_Source"] == (
        "INDEPENDENT ML PROBABILITY"
    )
    assert not bool(compact.loc[0, "Precision_Wager_Approved"])
    assert compact.loc[0, "Play_Stake"] == 0.0


def test_compact_win_amount_formula_uses_approved_stake_for_positive_odds():
    formula = _compact_win_amount_formula("AC", "AJ", 2)

    assert formula == (
        '=IF(OR(AC2="",AJ2="",AJ2=0),"",'
        'AC2*IF(AJ2>0,AJ2/100,100/ABS(AJ2)))'
    )


def test_compact_win_amount_formula_handles_negative_odds():
    formula = _compact_win_amount_formula("AC", "AJ", 9)

    assert "100/ABS(AJ9)" in formula
    assert "AC9*IF(AJ9>0" in formula


def test_compact_total_amount_formula_handles_win_loss_and_push():
    formula = _compact_total_amount_formula("AO", "AN", "AC", 2)

    assert formula == (
        '=IF(AO2="W",AN2,IF(AO2="L",-AC2,IF(AO2="P",0,"")))'
    )


def test_compact_summary_pnl_uses_approved_stake_and_refunds_pushes():
    formula = _compact_summary_pnl_formula("AC2:AC18", "AN2:AN18", "AO2:AO18")
    actionable = _compact_summary_pnl_formula(
        "AC2:AC18",
        "AN2:AN18",
        "AO2:AO18",
        actionable_range="AD2:AD18",
    )

    assert formula == (
        '=SUMIF(AO2:AO18,"W",AN2:AN18)-SUM(AC2:AC18)'
        '+SUMIF(AO2:AO18,"W",AC2:AC18)+SUMIF(AO2:AO18,"P",AC2:AC18)'
    )
    assert actionable == (
        '=SUMIFS(AN2:AN18,AD2:AD18,"Actionable",AO2:AO18,"W")'
        '-SUMIF(AD2:AD18,"Actionable",AC2:AC18)'
        '+SUMIFS(AC2:AC18,AD2:AD18,"Actionable",AO2:AO18,"W")'
        '+SUMIFS(AC2:AC18,AD2:AD18,"Actionable",AO2:AO18,"P")'
    )
