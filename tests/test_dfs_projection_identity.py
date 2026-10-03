"""Offline counterexamples for projection state and DFS identity contracts."""
import csv
from io import StringIO
from itertools import combinations
import socket

import numpy as np
import pandas as pd
import pytest

from app_core.draftkings_classic import (
    DK_NFL_CLASSIC_ROSTER_SLOTS, DK_MLB_CLASSIC_ROSTER_SLOTS,
    attach_draftkings_projections, build_draftkings_classic_lineups,
    build_draftkings_classic_shortlist, build_draftkings_mlb_classic_lineups,
    export_draftkings_classic_position_csv, parse_draftkings_classic_salary_csv,
    parse_draftkings_mlb_classic_salary_csv,
)
from test_draftkings_classic import _nfl_lineup_rows
from test_draftkings_mlb_classic import _mlb_salary_rows


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("DFS regressions must run offline")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


SPORTS = [
    (_nfl_lineup_rows, parse_draftkings_classic_salary_csv,
     build_draftkings_classic_lineups, DK_NFL_CLASSIC_ROSTER_SLOTS, "NFL"),
    (_mlb_salary_rows, parse_draftkings_mlb_classic_salary_csv,
     build_draftkings_mlb_classic_lineups, DK_MLB_CLASSIC_ROSTER_SLOTS, "MLB"),
]


@pytest.mark.parametrize("rows,parse,build,slots,sport", SPORTS)
def test_inline_zero_missing_invalid_are_distinct(rows, parse, build, slots, sport):
    frame = rows().iloc[:7].copy()
    frame["Projected Points"] = [0, "", None, "bad", np.inf, -np.inf, "NaN"]
    frame["AvgPointsPerGame"] = 99
    pool = parse(frame)
    assert pool["ProjectionStatus"].tolist() == ["valid", "missing", "missing"] + ["invalid"] * 4
    assert pool["ProjectedPoints"].iloc[:3].tolist() == [0, 99, 99]
    assert pool["ProjectedPoints"].iloc[3:].isna().all()
    assert pool["ProjectionSource"].tolist() == ["uploaded_projection"] + ["draftkings_average_fppg"] * 2 + ["invalid_projection"] * 4


@pytest.mark.parametrize("rows,parse,build,slots,sport", SPORTS)
def test_all_zero_upload_overrides_historical_average_and_is_usable(rows, parse, build, slots, sport):
    pool = parse(rows().drop(columns="Projected Points"))
    projected = attach_draftkings_projections(pool, pd.DataFrame({
        "ID": pool["ID"], "Projected Points": 0,
    }))
    assert projected["ProjectedPoints"].eq(0).all()
    assert projected["ValuePer1000"].eq(0).all()
    assert projected.attrs["projection_match_count"] == len(pool)
    lineups = build(projected, top_n=1)
    assert len(lineups) == 1
    assert lineups.iloc[0]["Projected Points"] == 0
    assert lineups.iloc[0]["Projection Sources"] == "uploaded_projection"


@pytest.mark.parametrize("bad", ["bad", "NaN", "Infinity", "-Infinity", "1e999", np.inf, -np.inf])
def test_invalid_uploaded_value_cannot_resurrect_high_fppg(bad):
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    result = attach_draftkings_projections(pool, pd.DataFrame({
        "ID": pool["ID"].iloc[:2], "Projected Points": [bad, ""],
    }))
    assert pd.isna(result.iloc[0]["ProjectedPoints"])
    assert result.iloc[0]["ProjectionSource"] == "invalid_projection"
    assert result.iloc[0]["ProjectionStatus"] == "invalid"
    assert result.iloc[1]["ProjectedPoints"] == pool.iloc[1]["ProjectedPoints"]
    assert result.iloc[1]["ProjectionStatus"] == "missing"
    assert result.attrs["projection_match_count"] == 0
    assert result.attrs["projection_missing_count"] == result.attrs["projection_invalid_count"] == 1
    assert result.attrs["projection_unmatched_count"] == len(pool) - 2
    assert result.iloc[0]["Name"] not in set(build_draftkings_classic_shortlist(result)["Name"])


def test_csv_literal_nan_is_invalid_and_blank_is_missing():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    ids = pool["ID"].iloc[:2].tolist()
    projected = attach_draftkings_projections(pool, f"ID,Projected Points\n{ids[0]},NaN\n{ids[1]},\n")
    assert projected["ProjectionStatus"].iloc[:2].tolist() == ["invalid", "missing"]


def test_mixed_upload_preserves_zero_and_decimal_over_integer_inline_values():
    frame = _nfl_lineup_rows().iloc[:3].copy()
    frame["Projected Points"] = 99
    pool = parse_draftkings_classic_salary_csv(frame)
    result = attach_draftkings_projections(pool, pd.DataFrame({
        "ID": pool["ID"], "Projection": [0, 12.25, ""],
    }))
    assert result["ProjectedPoints"].tolist() == [0, 12.25, 99]
    assert result["ProjectionStatus"].tolist() == ["valid", "valid", "missing"]
    assert result.attrs["projection_match_count"] == 2


def test_signed_finite_projection_and_numeric_id_normalization():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows().iloc[:1])
    pool["ID"] = "123"
    pool["Name + ID"] = "Player (123)"
    projected = attach_draftkings_projections(pool, pd.DataFrame({"ID": [123.0], "Projection": [-1.5]}))
    assert projected.iloc[0]["ProjectedPoints"] == -1.5


@pytest.mark.parametrize("value", [0, 50, "", "bad", np.inf])
def test_duplicate_projection_ids_fail_before_value_filtering(value):
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    with pytest.raises(ValueError, match="duplicate player IDs"):
        attach_draftkings_projections(pool, pd.DataFrame({
            "ID": [pool.iloc[0]["ID"]] * 2, "Projection": [20, value],
        }))


def test_conflicting_known_id_cannot_fall_back_to_same_name():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    with pytest.raises(ValueError, match="conflicting known player IDs"):
        attach_draftkings_projections(pool, pd.DataFrame({
            "ID": ["wrong-id"], "Name": [pool.iloc[0]["Name"]], "Projection": [90],
        }))


def test_matching_id_with_conflicting_name_fails_closed():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    with pytest.raises(ValueError, match="conflicting ID/name"):
        attach_draftkings_projections(pool, pd.DataFrame({
            "ID": [pool.iloc[0]["ID"]], "Name": [pool.iloc[1]["Name"]], "Projection": [90],
        }))


@pytest.mark.parametrize("side", ["pool", "projections"])
def test_normalized_name_fallback_requires_uniqueness_on_both_sides(side):
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    pool.loc[0, "Name"] = "A.B. Player"
    projection = pd.DataFrame({"Name": ["AB Player"], "Projection": [90]})
    if side == "pool":
        pool.loc[1, "Name"] = "AB Player"
    else:
        projection = pd.concat([projection, pd.DataFrame({"Name": ["A-B Player"], "Projection": [""]})])
    with pytest.raises(ValueError, match="ambiguous player name"):
        attach_draftkings_projections(pool, projection)


def test_exact_ids_disambiguate_same_names_and_unique_name_can_fill_missing_id():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows().iloc[:3])
    pool.loc[:1, "Name"] = "Same Name"
    projected = attach_draftkings_projections(pool, pd.DataFrame({
        "ID": [pool.iloc[0]["ID"], pool.iloc[1]["ID"], ""],
        "Name": ["Same Name", "Same Name", pool.iloc[2]["Name"]],
        "Projection": [1, 2, 3],
    }))
    assert projected["ProjectedPoints"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("rows,parse,build,slots,sport", SPORTS)
def test_duplicate_salary_ids_do_not_bypass_uniqueness_or_diversity(rows, parse, build, slots, sport):
    pool = parse(rows())
    # Multiple high-scoring row aliases force the old row-index optimizer to
    # select the same player more than once and to swap aliases for diversity.
    target = pool[pool["Position"].eq("WR" if sport == "NFL" else "OF")].iloc[0].copy()
    target["ProjectedPoints"] = 150
    aliases = []
    for number in range(4):
        alias = target.copy()
        alias["Name + ID"] = f"Alias {number} ({target['ID']})"
        aliases.append(alias)
    pool = pd.concat([pool, pd.DataFrame(aliases)], ignore_index=True)
    lineups = build(pool, top_n=3, min_unique_players_between_lineups=3)
    assert len(lineups) == 3
    id_sets = []
    for _, lineup in lineups.iterrows():
        ids = [lineup[slot].rsplit("(", 1)[1].rstrip(")") for slot in slots]
        assert len(set(ids)) == len(slots)
        assert lineup["Unique Players"] == len(slots)
        assert lineup["Salary"] <= 50_000
        id_sets.append(set(ids))
    for a, b in combinations(id_sets, 2):
        assert len(a & b) <= len(slots) - 3
    exported = list(csv.reader(StringIO(export_draftkings_classic_position_csv(lineups, sport=sport))))
    assert len(exported) == 4
    assert all(len(row) == len(slots) for row in exported)


@pytest.mark.parametrize("rows,parse,build,slots,sport", SPORTS)
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_nonfinite_direct_pool_does_not_reach_solver(rows, parse, build, slots, sport, bad):
    pool = parse(rows())
    pool["Salary"] = pool["Salary"].astype(float)
    pool.loc[0, "ProjectedPoints"] = bad
    pool.loc[1, "Salary"] = bad
    lineups = build(pool, top_n=1)
    assert len(lineups) == 1
    assert np.isfinite(lineups.iloc[0]["Projected Points"])
    assert all(lineups.iloc[0][slot] not in set(pool.iloc[:2]["Name + ID"]) for slot in slots)


@pytest.mark.parametrize("rows,parse,build,slots,sport", SPORTS)
def test_nonfinite_salary_and_fallback_are_unusable(rows, parse, build, slots, sport):
    frame = rows().iloc[:3].drop(columns="Projected Points").copy()
    frame["Salary"] = [np.inf, -np.inf, 5000]
    frame["AvgPointsPerGame"] = np.inf
    pool = parse(frame)
    assert len(pool) == 1
    assert pd.isna(pool.iloc[0]["ProjectedPoints"])


def test_conflicting_embedded_draftkings_id_rejected():
    pool = parse_draftkings_classic_salary_csv(_nfl_lineup_rows())
    pool.loc[0, "Name + ID"] = "Wrong Export (other-id)"
    with pytest.raises(ValueError, match="Conflicting DraftKings ID"):
        build_draftkings_classic_lineups(pool, top_n=1)
