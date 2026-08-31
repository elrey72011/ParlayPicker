import pandas as pd
import pytest

from app_core.draftkings_classic import (
    DK_NFL_CLASSIC_ROSTER_SLOTS,
    build_draftkings_classic_lineups,
    build_draftkings_classic_shortlist,
    parse_draftkings_classic_salary_csv,
)


def _salary_rows():
    rows = []
    for position, count in (("QB", 6), ("RB", 7), ("WR", 7), ("TE", 6), ("DST", 6)):
        for number in range(count):
            rows.append({
                "Position": position,
                "Name + ID": f"{position} Player {number} ({position}{number})",
                "Name": f"{position} Player {number}",
                "ID": f"{position}{number}",
                "Roster Position": position if position in {"QB", "DST"} else f"{position}/FLEX",
                "Salary": 3000 + number * 200,
                "Game Info": "BUF@NYJ 09/13/2026 01:00PM ET",
                "TeamAbbrev": "BUF",
                "AvgPointsPerGame": 10 + number,
                "Projected Points": 12 + number,
                "Status": "",
            })
    return pd.DataFrame(rows)


NFL_GAMES = {
    "BUF": "BUF@HOU 09/13/2026 01:00PM ET",
    "HOU": "BUF@HOU 09/13/2026 01:00PM ET",
    "BAL": "BAL@IND 09/13/2026 01:00PM ET",
    "IND": "BAL@IND 09/13/2026 01:00PM ET",
    "PHI": "WAS@PHI 09/13/2026 04:25PM ET",
    "WAS": "WAS@PHI 09/13/2026 04:25PM ET",
    "DET": "NO@DET 09/13/2026 01:00PM ET",
    "NO": "NO@DET 09/13/2026 01:00PM ET",
}


def _nfl_lineup_rows():
    rows = []
    for team_number, (team, game_info) in enumerate(NFL_GAMES.items()):
        for position, count, base_salary, base_projection in (
            ("QB", 1, 5_500, 19.0),
            ("RB", 3, 4_500, 12.0),
            ("WR", 5, 3_500, 10.0),
            ("TE", 2, 3_000, 8.0),
            ("DST", 1, 2_500, 7.0),
        ):
            for number in range(count):
                player_id = f"{team}{position}{number}"
                rows.append({
                    "Position": position,
                    "Name + ID": f"{team} {position} {number} ({player_id})",
                    "Name": f"{team} {position} {number}",
                    "ID": player_id,
                    "Roster Position": (
                        f"{position}/FLEX"
                        if position in {"RB", "WR", "TE"}
                        else position
                    ),
                    "Salary": base_salary + team_number * 80 + number * 120,
                    "Game Info": game_info,
                    "TeamAbbrev": team,
                    "AvgPointsPerGame": base_projection + team_number * 0.2 + number * 0.3,
                    "Projected Points": base_projection + team_number * 0.25 + number * 0.4,
                    "Status": "",
                })
    return pd.DataFrame(rows)


def _nfl_opponent(team: str) -> str:
    away, home = NFL_GAMES[team].split(" ", 1)[0].split("@")
    return home if team == away else away


def test_parse_standard_draftkings_classic_salary_csv():
    pool = parse_draftkings_classic_salary_csv(_salary_rows())

    assert set(pool["Position"]) == {"QB", "RB", "WR", "TE", "DST"}
    assert pool["Salary"].dtype.kind in "iu"
    assert pool["ProjectionSource"].eq("uploaded_projection").all()
    assert pool["ValuePer1000"].notna().all()


def test_shortlist_returns_five_per_position_and_flex_eligibility():
    pool = parse_draftkings_classic_salary_csv(_salary_rows())
    shortlist = build_draftkings_classic_shortlist(pool)

    assert shortlist.groupby("Classic Position").size().to_dict() == {
        "QB": 5,
        "RB": 5,
        "WR": 5,
        "TE": 5,
        "FLEX": 5,
        "DST": 5,
    }
    flex = shortlist[shortlist["Classic Position"].eq("FLEX")]
    assert set(flex["Position"]).issubset({"RB", "WR", "TE"})
    assert not set(flex["Position"]).intersection({"QB", "DST"})
    assert shortlist.groupby("Classic Position")["Rank"].max().eq(5).all()
    assert shortlist["ClassicScore"].between(0, 1).all()


def test_rank_is_projection_first_and_average_is_labeled_as_fallback():
    frame = _salary_rows().drop(columns=["Projected Points"])
    frame.loc[frame["Name"].eq("QB Player 0"), "AvgPointsPerGame"] = 99
    pool = parse_draftkings_classic_salary_csv(frame)
    shortlist = build_draftkings_classic_shortlist(pool)

    qb = shortlist[shortlist["Classic Position"].eq("QB")]
    assert qb.iloc[0]["Name"] == "QB Player 0"
    assert qb.iloc[0]["ProjectionSource"] == "draftkings_average_fppg"


def test_inactive_and_invalid_salary_rows_are_excluded():
    frame = _salary_rows()
    frame.loc[frame["Name"].eq("QB Player 5"), "Status"] = "OUT"
    frame.loc[frame["Name"].eq("RB Player 6"), "Salary"] = 0

    pool = parse_draftkings_classic_salary_csv(frame)

    assert "QB Player 5" not in set(pool["Name"])
    assert "RB Player 6" not in set(pool["Name"])


def test_classic_score_rewards_salary_value_without_ignoring_projection():
    frame = pd.DataFrame([
        {
            "Position": "QB", "Name": "Elite", "Salary": 8000,
            "Projected Points": 25, "AvgPointsPerGame": 25,
        },
        {
            "Position": "QB", "Name": "Value", "Salary": 5000,
            "Projected Points": 24, "AvgPointsPerGame": 24,
        },
        {
            "Position": "QB", "Name": "Cheap", "Salary": 4000,
            "Projected Points": 12, "AvgPointsPerGame": 12,
        },
    ])

    shortlist = build_draftkings_classic_shortlist(
        parse_draftkings_classic_salary_csv(frame)
    )

    assert shortlist.iloc[0]["Name"] == "Value"
    assert shortlist.iloc[-1]["Name"] == "Cheap"


def test_nfl_optimizer_returns_five_complete_valid_and_unique_lineups():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    lineups = build_draftkings_classic_lineups(pool, top_n=5)

    assert len(lineups) == 5
    assert lineups["Lineup Rank"].tolist() == [1, 2, 3, 4, 5]
    assert lineups["Lineup Key"].nunique() == 5
    assert lineups["Salary"].le(50_000).all()
    assert lineups["Unused Salary"].ge(0).all()
    assert lineups["Unique Players"].eq(9).all()
    assert lineups["Teams"].ge(2).all()
    assert lineups["Projected Points"].is_monotonic_decreasing
    assert lineups[list(DK_NFL_CLASSIC_ROSTER_SLOTS)].notna().all().all()


def test_nfl_optimizer_honors_slots_qb_stack_and_dst_conflict_guard():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    lineups = build_draftkings_classic_lineups(pool, top_n=5)
    by_name_id = pool.set_index("Name + ID")

    for _, lineup in lineups.iterrows():
        selected = {
            slot: by_name_id.loc[lineup[slot]]
            for slot in DK_NFL_CLASSIC_ROSTER_SLOTS
        }
        assert len({lineup[slot] for slot in DK_NFL_CLASSIC_ROSTER_SLOTS}) == 9
        assert selected["QB"]["Position"] == "QB"
        assert selected["DST"]["Position"] == "DST"
        assert all(selected[slot]["Position"] == "RB" for slot in ("RB1", "RB2"))
        assert all(selected[slot]["Position"] == "WR" for slot in ("WR1", "WR2", "WR3"))
        assert selected["TE"]["Position"] == "TE"
        assert selected["FLEX"]["Position"] in {"RB", "WR", "TE"}

        qb_team = selected["QB"]["Team"]
        assert any(
            player["Team"] == qb_team and player["Position"] in {"WR", "TE"}
            for player in selected.values()
        )
        dst_opponent = _nfl_opponent(selected["DST"]["Team"])
        assert all(
            player["Team"] != dst_opponent
            for slot, player in selected.items()
            if slot != "DST"
        )


def test_missing_required_salary_columns_are_explained():
    with pytest.raises(ValueError, match="salary"):
        parse_draftkings_classic_salary_csv(pd.DataFrame({
            "Position": ["QB"],
            "Name": ["Player"],
        }))
