import pandas as pd

from streamlit_app import _build_compact_export_frame


def _compact_source(**overrides):
    row = {
        "pipeline_build": "2026-08-01c-compact-export-stake-safety",
        "Wager_Instruction": "DO NOT BET - $0 PASS / RESEARCH",
        "Export_Scope": "COVERAGE / RESEARCH",
        "Bettable": False,
        "Play_Stake": 0.0,
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


def test_compact_export_retains_only_explicitly_approved_stake():
    source = pd.DataFrame([
        _compact_source(
            Wager_Instruction="APPROVED: wager the exported Play_Stake amount.",
            Export_Scope="PRODUCTION WAGERS ONLY",
            Bettable=True,
            Play_Stake=12.5,
            Kelly_Bet_Size=18.75,
        )
    ])

    compact = _build_compact_export_frame(source)

    assert compact.loc[0, "Play_Stake"] == 12.5
    assert compact.loc[0, "Kelly_Bet_Size"] == 18.75


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


def test_compact_export_uses_public_display_pick_and_keeps_legacy_fallback():
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
        "NO QUALIFIED PICK",
        "Washington +1.5",
        "Pittsburgh +1.5",
    ]
    assert "display_pick" not in compact.columns
