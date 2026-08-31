from collections import Counter

import pandas as pd
import pytest

from app_core.draftkings_classic import (
    DK_MLB_CLASSIC_ROSTER_SLOTS,
    build_draftkings_mlb_classic_lineups,
    parse_draftkings_mlb_classic_salary_csv,
)


GAMES = {
    "BOS": "BOS@NYY 08/31/2026 07:05PM ET",
    "NYY": "BOS@NYY 08/31/2026 07:05PM ET",
    "LAD": "LAD@SF 08/31/2026 09:45PM ET",
    "SF": "LAD@SF 08/31/2026 09:45PM ET",
    "CHC": "CHC@STL 08/31/2026 07:45PM ET",
    "STL": "CHC@STL 08/31/2026 07:45PM ET",
    "HOU": "HOU@SEA 08/31/2026 09:40PM ET",
    "SEA": "HOU@SEA 08/31/2026 09:40PM ET",
}


def _mlb_salary_rows() -> pd.DataFrame:
    rows = []
    teams = list(GAMES)
    for number, team in enumerate(teams):
        rows.append({
            "Position": "SP",
            "Name + ID": f"{team} Pitcher ({team}P)",
            "Name": f"{team} Pitcher",
            "ID": f"{team}P",
            "Roster Position": "P",
            "Salary": 7_200 + number * 250,
            "Game Info": GAMES[team],
            "TeamAbbrev": team,
            "AvgPointsPerGame": 18 + number * 0.5,
            "Projected Points": 20 + number * 0.7,
            "Status": "",
        })

    positions = ("C", "1B", "2B", "3B", "SS", "OF")
    for position_number, position in enumerate(positions):
        for number, team in enumerate(teams):
            source_position = position
            roster_position = position
            if position == "OF" and number == 0:
                source_position = "LF"
            if position == "1B" and number == 0:
                source_position = "1B/3B"
                roster_position = "1B/3B"
            rows.append({
                "Position": source_position,
                "Name + ID": f"{team} {position} Hitter ({team}{position}{number})",
                "Name": f"{team} {position} Hitter",
                "ID": f"{team}{position}{number}",
                "Roster Position": roster_position,
                "Salary": 3_000 + position_number * 150 + number * 90,
                "Game Info": GAMES[team],
                "TeamAbbrev": team,
                "AvgPointsPerGame": 7 + position_number * 0.4 + number * 0.15,
                "Projected Points": 8 + position_number * 0.5 + number * 0.2,
                "Status": "",
            })
    return pd.DataFrame(rows)


def _opponent(team: str) -> str:
    away, home = GAMES[team].split(" ", 1)[0].split("@")
    return home if team == away else away


def test_parse_mlb_salary_csv_normalizes_pitchers_outfield_and_multi_position():
    pool = parse_draftkings_mlb_classic_salary_csv(_mlb_salary_rows())

    assert pool.loc[pool["Name"].eq("BOS Pitcher"), "Position"].iloc[0] == "P"
    assert pool.loc[pool["Name"].eq("BOS OF Hitter"), "Position"].iloc[0] == "OF"
    assert (
        pool.loc[pool["Name"].eq("BOS 1B Hitter"), "EligiblePositions"].iloc[0]
        == "1B/3B"
    )
    assert pool["ProjectionSource"].eq("uploaded_projection").all()


def test_optimizer_returns_five_complete_valid_and_unique_lineups():
    pool = parse_draftkings_mlb_classic_salary_csv(_mlb_salary_rows())
    lineups = build_draftkings_mlb_classic_lineups(pool, top_n=5)

    assert len(lineups) == 5
    assert lineups["Lineup Rank"].tolist() == [1, 2, 3, 4, 5]
    assert lineups["Lineup Key"].nunique() == 5
    assert lineups["Salary"].le(50_000).all()
    assert lineups["Unused Salary"].ge(0).all()
    assert lineups["Unique Players"].eq(10).all()
    assert lineups["Max Hitters / Team"].le(5).all()
    assert lineups["Teams"].ge(2).all()
    assert lineups["Projected Points"].is_monotonic_decreasing
    assert lineups[list(DK_MLB_CLASSIC_ROSTER_SLOTS)].notna().all().all()


def test_optimizer_honors_positions_team_cap_and_pitcher_hitter_conflicts():
    pool = parse_draftkings_mlb_classic_salary_csv(_mlb_salary_rows())
    lineups = build_draftkings_mlb_classic_lineups(pool, top_n=5)
    by_name_id = pool.set_index("Name + ID")

    for _, lineup in lineups.iterrows():
        selected = {
            slot: by_name_id.loc[lineup[slot]]
            for slot in DK_MLB_CLASSIC_ROSTER_SLOTS
        }
        assert len({lineup[slot] for slot in DK_MLB_CLASSIC_ROSTER_SLOTS}) == 10
        for slot, player in selected.items():
            required = "P" if slot.startswith("P") else "OF" if slot.startswith("OF") else slot
            assert required in player["EligiblePositions"].split("/")

        hitters = [player for slot, player in selected.items() if not slot.startswith("P")]
        hitter_counts = Counter(player["Team"] for player in hitters)
        assert max(hitter_counts.values()) <= 5
        pitcher_opponents = {
            _opponent(selected[slot]["Team"])
            for slot in ("P1", "P2")
        }
        assert pitcher_opponents.isdisjoint(hitter_counts)


def test_optimizer_uses_draftkings_average_as_labeled_projection_fallback():
    frame = _mlb_salary_rows().drop(columns=["Projected Points"])
    pool = parse_draftkings_mlb_classic_salary_csv(frame)
    lineups = build_draftkings_mlb_classic_lineups(pool, top_n=1)

    assert len(lineups) == 1
    assert lineups.iloc[0]["Projection Sources"] == "draftkings_average_fppg"


def test_parser_excludes_draftkings_injured_list_rows():
    frame = _mlb_salary_rows()
    frame.loc[frame["Name"].eq("BOS Pitcher"), "Status"] = "IL"

    pool = parse_draftkings_mlb_classic_salary_csv(frame)

    assert "BOS Pitcher" not in set(pool["Name"])


def test_optimizer_uses_confirmed_probable_starters_when_available():
    frame = _mlb_salary_rows()
    frame["Starting"] = ""
    frame.loc[frame["Name"].isin({"BOS Pitcher", "LAD Pitcher"}), "Starting"] = "SP"
    frame.loc[frame["Name"].eq("NYY Pitcher"), "Projected Points"] = 100.0

    pool = parse_draftkings_mlb_classic_salary_csv(frame)
    lineups = build_draftkings_mlb_classic_lineups(
        pool,
        top_n=1,
        avoid_pitcher_hitter_conflicts=False,
    )

    selected_pitchers = {lineups.iloc[0]["P1"], lineups.iloc[0]["P2"]}
    assert selected_pitchers == {
        "BOS Pitcher (BOSP)",
        "LAD Pitcher (LADP)",
    }


def test_pitcher_hitter_conflict_guard_fails_closed_when_no_clean_lineup_exists():
    frame = _mlb_salary_rows()
    pitchers = frame["Position"].eq("SP") & frame["TeamAbbrev"].isin({"BOS", "LAD"})
    hitters = ~frame["Position"].eq("SP") & frame["TeamAbbrev"].isin({"NYY", "SF"})
    extra_outfielder = frame["Position"].eq("OF") & frame["TeamAbbrev"].eq("HOU")
    pool = parse_draftkings_mlb_classic_salary_csv(
        frame[pitchers | hitters | extra_outfielder]
    )

    assert build_draftkings_mlb_classic_lineups(
        pool,
        top_n=1,
        avoid_pitcher_hitter_conflicts=True,
    ).empty
    assert len(build_draftkings_mlb_classic_lineups(
        pool,
        top_n=1,
        avoid_pitcher_hitter_conflicts=False,
    )) == 1


def test_optimizer_returns_empty_when_a_required_position_is_unavailable():
    frame = _mlb_salary_rows()
    frame = frame[~frame["Position"].eq("C")]
    pool = parse_draftkings_mlb_classic_salary_csv(frame)

    assert build_draftkings_mlb_classic_lineups(pool).empty


def test_mlb_parser_and_optimizer_explain_invalid_inputs():
    with pytest.raises(ValueError, match="salary"):
        parse_draftkings_mlb_classic_salary_csv(pd.DataFrame({
            "Position": ["P"],
            "Name": ["Pitcher"],
        }))

    pool = parse_draftkings_mlb_classic_salary_csv(_mlb_salary_rows())
    with pytest.raises(ValueError, match="top_n"):
        build_draftkings_mlb_classic_lineups(pool, top_n=0)
