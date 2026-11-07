"""TheOver research signals (owner 2026-11-07): kept per row, never blended."""
import pandas as pd

from core.streamlit_pipeline import (BEST_PICK_COLUMNS, CANONICAL_BET_COLUMNS, THEOVER_RESEARCH_COLUMNS,
                                     _pick_code_names, build_theover_bet_rows)

SIDES = pd.DataFrame({
    "League": ["NCAAB", "NCAAB", "NCAAB", "NBA"],
    "HomeTeam": ["Michigan", "Kansas", "Oklahoma State", "Atlanta"],
    "AwayTeam": ["UConn", "Middle Tennessee", "Queens University", "Washington"],
    "HomeKalshi": ["MICHIGAN", "KANSAS", "OKLAHOMA STATE", "ATL"], "AwayKalshi": ["UCONN", "MIDDLE TENNESSEE", "QUEENS UNIVERSITY", "WAS"],
    "Pick": ["MICH", "MTU", "OKST", "ATL"], "Line": [7.5, 29.5, -19.5, -7.5], "WinProbability": [None] * 4,
    "Market": ["Spread"] * 4, "Matchup": ["UConn @ Michigan", "Middle Tennessee @ Kansas", "Queens University @ Oklahoma State",
                                          "Washington @ Atlanta"]})
TOTALS = pd.DataFrame({
    "League": ["NBA", "NBA", "NCAAB"], "GameTime": ["7:00 PM"] * 3,
    "HomeTeam": ["Charlotte", "Atlanta", "Kennesaw State"], "AwayTeam": ["Boston", "Washington", "Austin Peay"],
    "HomeKalshi": ["CHARLOTTE", "ATLANTA", "KENNESAW STATE"], "AwayKalshi": ["BOSTON", "WASHINGTON", "AUSTIN PEAY"],
    "Pick": ["OVER", "UNDER", "OVER"], "PickCode": ["OU_OVER", "OU_UNDER", "OU_OVER"], "Line": [207.5, 228.5, 157.5],
    "WinProbability": [None] * 3, "PublicBettingPct": [0.87, None, 0.66], "BettingDirection": ["OVER", None, "UNDER"],
    "ModelHitRate": [None, 0.714, None], "ModelText": [None, "Backed by Pulsar", None],
    "WinProbSource": ["not_provided"] * 3, "Market": ["Total"] * 3,
    "Matchup": ["Boston @ Charlotte", "Washington @ Atlanta", "Austin Peay @ Kennesaw State"]})


def _row(rows, home, market):
    return rows[(rows.home_team == home) & (rows.market_type == market)].iloc[0]


def test_totals_carry_pick_side_public_split_and_hit_rate_without_a_probability():
    rows = build_theover_bet_rows(None, TOTALS, None)
    over, under = _row(rows, "Charlotte", "total_over"), _row(rows, "Charlotte", "total_under")
    assert (over.theover_pick_agrees, under.theover_pick_agrees) == (1.0, 0.0)
    assert abs(over.theover_public_pct - 0.87) < 1e-9 and abs(under.theover_public_pct - 0.13) < 1e-9
    kennesaw = _row(rows, "Kennesaw State", "total_over")
    assert abs(kennesaw.theover_public_pct - 0.34) < 1e-9          # 66% of bets on the Under
    atl_under = _row(rows, "Atlanta", "total_under")
    assert atl_under.theover_pick_agrees == 1.0 and abs(atl_under.theover_model_hit_rate - 0.714) < 1e-9
    assert rows["theover_probability"].isna().all()                # nothing turned into a probability


def test_side_pick_codes_resolve_to_the_right_team_and_line():
    rows = build_theover_bet_rows(SIDES, None, None)
    mich = _row(rows, "Michigan", "spread_home")
    assert mich.theover_pick_agrees == 1.0 and mich.theover_pick_line == 7.5
    kansas_home = _row(rows, "Kansas", "spread_home")
    assert kansas_home.theover_pick_agrees == 0.0 and kansas_home.theover_pick_line == -29.5
    assert _row(rows, "Kansas", "spread_away").theover_pick_agrees == 1.0
    assert _row(rows, "Oklahoma State", "spread_home").theover_pick_agrees == 1.0
    assert _row(rows, "Atlanta", "spread_home").theover_pick_line == -7.5


def test_pick_code_matching_and_ambiguity():
    assert _pick_code_names("OKST", "OKLAHOMASTATE", "")
    assert _pick_code_names("CBU", "CALIFORNIABAPTIST", "")
    assert not _pick_code_names("MICH", "UCONN", "UCONN")
    assert _pick_code_names("GT", "GEORGIATECH", "") and _pick_code_names("GT", "GEORGIASOUTHERN", "")  # both: left unresolved
    assert not _pick_code_names("G", "GEORGIATECH", "")


def test_columns_reach_the_canonical_rows_and_the_export():
    for column in THEOVER_RESEARCH_COLUMNS:
        assert column in CANONICAL_BET_COLUMNS and column in BEST_PICK_COLUMNS


def _workbook(tmp_path):
    table = TOTALS.copy()
    page = pd.DataFrame({0: [
        "Picks", "NBA", "19:00:00", "PURPLE STAR", "Washington Wizards", "Washington", "Wizards", "0",
        "Atlanta Hawks", "Atlanta", "Hawks", "0", "Backed by Pulsar, which hit 71.4% of the last 14 NBA games.",
        "Fade the public", "Sharp money agrees · 3 plays", "UNDER", "228.5", "PENDING", "NBA", "19:00:00",
        "Boston Celtics", "Boston", "Celtics", "0", "Charlotte Hornets", "Charlotte", "Hornets", "0",
        "87% betting over", "o207.5", "Best line", "u208.5", "87% Over", "% of bets", "Under 13%", "79% Over",
        "% of money", "Under 21%", "OVER", "207.5", "PENDING", "NCAAB", "19:00:00", "Austin Peay Governors",
        "Austin Peay", "Austin Peay", "0", "Kennesaw State Owls", "Kennesaw State", "Kennesaw St", "0",
        "66% betting under", "OVER", "157.5", "PENDING"]})
    path = tmp_path / "Theover_Totals.xlsx"
    with pd.ExcelWriter(path) as writer:
        table.to_excel(writer, sheet_name="TotalsRaw", index=False)
        page.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
    return path


def test_workbook_page_signals_parse_and_attach(tmp_path):
    from app_core.theover_signals import read_workbook
    table, matched = read_workbook(_workbook(tmp_path))
    assert matched == 3
    atl = table[table.HomeTeam == "Atlanta"].iloc[0]
    assert atl.SharpMoneyAgrees is True and atl.SharpPlays == 3 and atl.PurpleStar and atl.FadePublic
    cha = table[table.HomeTeam == "Charlotte"].iloc[0]
    assert cha.SharpMoneyAgrees is False and cha.BetsPctOver == 0.87 and cha.MoneyPctOver == 0.79
    ken = table[table.HomeTeam == "Kennesaw State"].iloc[0]
    assert abs(ken.BetsPctOver - 0.34) < 1e-9 and pd.isna(ken.MoneyPctOver)   # "66% betting under"


def test_workbook_upload_reaches_rows_oriented_to_each_side(tmp_path):
    import io
    from core.theover_loader import load_theover_csv
    upload = io.BytesIO(_workbook(tmp_path).read_bytes())
    upload.name = "Theover_Totals.xlsx"
    totals, warning = load_theover_csv(upload)
    rows = build_theover_bet_rows(None, totals, None)
    under = _row(rows, "Atlanta", "total_under")
    assert (under.theover_sharp_agrees, under.theover_sharp_plays, under.theover_purple_star, under.theover_fade_public) == (1.0, 3.0, 1.0, 1.0)
    over, cha_under = _row(rows, "Charlotte", "total_over"), _row(rows, "Charlotte", "total_under")
    assert abs(over.theover_money_pct - 0.79) < 1e-9 and abs(cha_under.theover_money_pct - 0.21) < 1e-9
    assert rows["theover_probability"].isna().all()


def test_csv_upload_has_no_page_signals():
    rows = build_theover_bet_rows(None, TOTALS, None)
    assert rows["theover_sharp_agrees"].isna().all() and rows["theover_money_pct"].isna().all()
