import pandas as pd
import pytest

from app_core.draftkings_classic import (
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


def test_missing_required_salary_columns_are_explained():
    with pytest.raises(ValueError, match="salary"):
        parse_draftkings_classic_salary_csv(pd.DataFrame({
            "Position": ["QB"],
            "Name": ["Player"],
        }))
