"""DraftKings Classic player-pool parsing, shortlists, and lineup optimization.

DraftKings Classic salaries are contest/slate specific, so the official contest
CSV is the source of truth. This module does not scrape or invent salaries. It
normalizes downloaded NFL and MLB player pools, retains the legacy NFL position
shortlist, and optimizes complete salary-cap-compliant lineups for both sports.
"""
from __future__ import annotations

import re
from io import BytesIO, StringIO
from typing import Any

import pandas as pd


DK_CLASSIC_SALARY_CAP = 50_000
DK_CLASSIC_POSITION_ORDER = ("QB", "RB", "WR", "TE", "FLEX", "DST")
DK_CLASSIC_FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})
DK_NFL_CLASSIC_ROSTER_SLOTS = (
    "QB",
    "RB1",
    "RB2",
    "WR1",
    "WR2",
    "WR3",
    "TE",
    "FLEX",
    "DST",
)
DK_MLB_CLASSIC_POSITION_ORDER = ("P", "C", "1B", "2B", "3B", "SS", "OF")
DK_MLB_CLASSIC_POSITIONS = frozenset(DK_MLB_CLASSIC_POSITION_ORDER)
DK_MLB_CLASSIC_ROSTER_SLOTS = (
    "P1",
    "P2",
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "OF1",
    "OF2",
    "OF3",
)

_COLUMN_ALIASES = {
    "position": ("position", "pos"),
    "name_id": ("name + id", "name+id", "name_id"),
    "name": ("name", "player", "player name"),
    "id": ("id", "player id", "player_id"),
    "roster_position": ("roster position", "roster_position"),
    "salary": ("salary", "dk salary", "draftkings salary"),
    "game_info": ("game info", "game_info", "matchup"),
    "team": ("teamabbrev", "team abbrev", "team", "team_abbrev"),
    "average_points": (
        "avgpointspergame",
        "avg points per game",
        "fppg",
        "avg_fppg",
    ),
    "projection": (
        "projected points",
        "projectedpoints",
        "projection",
        "proj points",
        "proj_points",
        "fantasy points projection",
    ),
    "status": ("status", "injury status", "injurystatus"),
    "starting": ("starting", "confirmed starter", "lineup spot", "batting order"),
}


def _column_key(value: object) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {_column_key(column): column for column in frame.columns}
    for alias in aliases:
        match = normalized.get(_column_key(alias))
        if match is not None:
            return match
    return None


def _read_salary_source(source: Any) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if hasattr(source, "getvalue"):
        source = source.getvalue()
    if isinstance(source, bytes):
        return pd.read_csv(BytesIO(source), encoding="utf-8-sig")
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return pd.read_csv(StringIO(source))
        return pd.read_csv(source, encoding="utf-8-sig")
    raise TypeError("DraftKings salary source must be a CSV, bytes, or DataFrame")


def _normalize_position(value: object) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"D", "D/ST", "DEF", "DEFENSE", "DST"}:
        return "DST"
    for position in ("QB", "RB", "WR", "TE", "DST"):
        if position in {part.strip() for part in raw.replace("-", "/").split("/")}:
            return position
    return raw


def _mlb_position_tokens(value: object) -> tuple[str, ...]:
    """Return normalized DraftKings MLB roster positions in source order."""

    raw = str(value or "").strip().upper()
    if not raw:
        return ()
    normalized: list[str] = []
    for part in raw.replace(",", "/").replace(";", "/").split("/"):
        token = part.strip()
        if token in {"P", "SP", "RP", "PITCHER"}:
            token = "P"
        elif token in {"LF", "CF", "RF", "OUTFIELD"}:
            token = "OF"
        elif token == "DH":
            token = "1B"
        if token in DK_MLB_CLASSIC_POSITIONS and token not in normalized:
            normalized.append(token)
    return tuple(normalized)


def _classic_score(ranked: pd.DataFrame) -> pd.Series:
    projection_span = ranked["ProjectedPoints"].max() - ranked["ProjectedPoints"].min()
    value_span = ranked["ValuePer1000"].max() - ranked["ValuePer1000"].min()
    projection_score = (
        (ranked["ProjectedPoints"] - ranked["ProjectedPoints"].min())
        / projection_span
        if projection_span > 0
        else pd.Series(1.0, index=ranked.index)
    )
    value_score = (
        (ranked["ValuePer1000"] - ranked["ValuePer1000"].min()) / value_span
        if value_span > 0
        else pd.Series(1.0, index=ranked.index)
    )
    return (0.70 * projection_score + 0.30 * value_score).round(4)


def parse_draftkings_classic_salary_csv(source: Any) -> pd.DataFrame:
    """Normalize an official DraftKings NFL Classic player-pool CSV.

    The standard DraftKings fields are supported along with common projection
    aliases used by exported research sheets. A projection is preferred when
    supplied; otherwise DraftKings' average fantasy points per game is retained
    as an explicitly labeled fallback rather than being presented as a forecast.
    """
    raw = _read_salary_source(source)
    columns = {
        name: _find_column(raw, aliases)
        for name, aliases in _COLUMN_ALIASES.items()
    }
    missing = [name for name in ("position", "name", "salary") if not columns[name]]
    if missing:
        raise ValueError(
            "DraftKings salary CSV is missing required column(s): " + ", ".join(missing)
        )

    index = raw.index
    out = pd.DataFrame(index=index)
    out["Position"] = raw[columns["position"]].map(_normalize_position)
    out["Name"] = raw[columns["name"]].fillna("").astype(str).str.strip()
    out["Name + ID"] = (
        raw[columns["name_id"]].fillna("").astype(str).str.strip()
        if columns["name_id"]
        else out["Name"]
    )
    out["ID"] = (
        raw[columns["id"]].fillna("").astype(str).str.strip()
        if columns["id"]
        else ""
    )
    out["Roster Position"] = (
        raw[columns["roster_position"]].fillna("").astype(str).str.strip()
        if columns["roster_position"]
        else out["Position"]
    )
    out["Salary"] = pd.to_numeric(raw[columns["salary"]], errors="coerce")
    out["Game Info"] = (
        raw[columns["game_info"]].fillna("").astype(str).str.strip()
        if columns["game_info"]
        else ""
    )
    out["Team"] = (
        raw[columns["team"]].fillna("").astype(str).str.strip().str.upper()
        if columns["team"]
        else ""
    )
    out["AvgPointsPerGame"] = (
        pd.to_numeric(raw[columns["average_points"]], errors="coerce")
        if columns["average_points"]
        else pd.Series(float("nan"), index=index)
    )
    supplied_projection = (
        pd.to_numeric(raw[columns["projection"]], errors="coerce")
        if columns["projection"]
        else pd.Series(float("nan"), index=index)
    )
    out["ProjectedPoints"] = supplied_projection.fillna(out["AvgPointsPerGame"])
    out["ProjectionSource"] = "unavailable"
    out.loc[supplied_projection.notna(), "ProjectionSource"] = "uploaded_projection"
    out.loc[
        supplied_projection.isna() & out["AvgPointsPerGame"].notna(),
        "ProjectionSource",
    ] = "draftkings_average_fppg"
    out["Status"] = (
        raw[columns["status"]].fillna("").astype(str).str.strip().str.upper()
        if columns["status"]
        else ""
    )

    active = ~out["Status"].isin({"O", "OUT", "IR", "INACTIVE", "SUSPENDED"})
    valid_position = out["Position"].isin({"QB", "RB", "WR", "TE", "DST"})
    valid_identity = out["Name"].ne("")
    valid_salary = out["Salary"].notna() & out["Salary"].gt(0)
    out = out[active & valid_position & valid_identity & valid_salary].copy()
    out["Salary"] = out["Salary"].astype(int)
    out["ValuePer1000"] = (
        out["ProjectedPoints"] * 1000.0 / out["Salary"]
    ).round(3)
    return out.reset_index(drop=True)


def build_draftkings_classic_shortlist(
    player_pool: pd.DataFrame,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top ``top_n`` options for every NFL Classic roster bucket.

    Ranking weights projection at 70% and salary-adjusted value at 30% within
    each roster bucket. Projection, rather than bargain salary alone, therefore
    remains the primary signal while Classic's salary cap still affects order.
    FLEX is ranked independently from eligible RB/WR/TE players. The output is a
    shortlist, not a claim that all listed players fit together under the cap.
    """
    if player_pool is None or player_pool.empty:
        return pd.DataFrame()
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")

    required = {"Position", "Name", "Salary", "ProjectedPoints", "ValuePer1000"}
    missing = sorted(required.difference(player_pool.columns))
    if missing:
        raise ValueError("Normalized player pool is missing: " + ", ".join(missing))

    frames: list[pd.DataFrame] = []
    for bucket in DK_CLASSIC_POSITION_ORDER:
        if bucket == "FLEX":
            eligible = player_pool["Position"].isin(DK_CLASSIC_FLEX_POSITIONS)
        else:
            eligible = player_pool["Position"].eq(bucket)
        ranked = player_pool[eligible & player_pool["ProjectedPoints"].notna()].copy()
        ranked["ClassicScore"] = _classic_score(ranked)
        ranked = ranked.sort_values(
            ["ClassicScore", "ProjectedPoints", "ValuePer1000", "Salary", "Name"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        ).head(int(top_n))
        if ranked.empty:
            continue
        ranked.insert(0, "Classic Position", bucket)
        ranked.insert(1, "Rank", range(1, len(ranked) + 1))
        frames.append(ranked)

    if not frames:
        return pd.DataFrame()
    columns = [
        "Classic Position",
        "Rank",
        "Position",
        "Name",
        "Team",
        "Salary",
        "ProjectedPoints",
        "ValuePer1000",
        "ClassicScore",
        "ProjectionSource",
        "Game Info",
        "ID",
        "Name + ID",
        "Roster Position",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def _nfl_slot_positions(slot: str) -> frozenset[str]:
    if slot.startswith("RB"):
        return frozenset({"RB"})
    if slot.startswith("WR"):
        return frozenset({"WR"})
    if slot == "FLEX":
        return DK_CLASSIC_FLEX_POSITIONS
    return frozenset({slot})


def build_draftkings_classic_lineups(
    player_pool: pd.DataFrame,
    *,
    top_n: int = 5,
    salary_cap: int = DK_CLASSIC_SALARY_CAP,
    require_qb_pass_catcher: bool = True,
    avoid_offense_against_dst: bool = True,
) -> pd.DataFrame:
    """Optimize the top complete DraftKings NFL Classic lineups.

    Lineups contain QB, two RBs, three WRs, TE, RB/WR/TE FLEX, and DST. Every
    lineup stays under the salary cap, uses each player once, and draws from at
    least two teams. The default tournament construction requires a same-team
    WR/TE with the quarterback and excludes offensive players facing the DST.
    """

    if player_pool is None or player_pool.empty:
        return pd.DataFrame()
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    if int(salary_cap) <= 0:
        raise ValueError("salary_cap must be positive")

    required = {
        "Position",
        "Name",
        "Salary",
        "ProjectedPoints",
        "ValuePer1000",
        "Team",
        "Game Info",
    }
    missing = sorted(required.difference(player_pool.columns))
    if missing:
        raise ValueError("Normalized NFL player pool is missing: " + ", ".join(missing))

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError(
            "NFL Classic lineup optimization requires scipy."
        ) from exc

    pool = player_pool.copy()
    pool["Position"] = pool["Position"].fillna("").astype(str).str.upper()
    pool["Salary"] = pd.to_numeric(pool["Salary"], errors="coerce")
    pool["ProjectedPoints"] = pd.to_numeric(
        pool["ProjectedPoints"], errors="coerce"
    )
    pool["Team"] = pool["Team"].fillna("").astype(str).str.strip().str.upper()
    pool = pool[
        pool["Position"].isin({"QB", "RB", "WR", "TE", "DST"})
        & pool["Salary"].notna()
        & pool["Salary"].gt(0)
        & pool["ProjectedPoints"].notna()
        & pool["Team"].ne("")
    ].reset_index(drop=True)
    if pool.empty:
        return pd.DataFrame()

    assignments: list[tuple[int, str]] = []
    for player_index, player in pool.iterrows():
        position = str(player["Position"])
        for slot in DK_NFL_CLASSIC_ROSTER_SLOTS:
            if position in _nfl_slot_positions(slot):
                assignments.append((int(player_index), slot))
    if not assignments:
        return pd.DataFrame()

    assignment_count = len(assignments)
    slot_variables = {
        slot: [
            i
            for i, (_, assignment_slot) in enumerate(assignments)
            if assignment_slot == slot
        ]
        for slot in DK_NFL_CLASSIC_ROSTER_SLOTS
    }
    if any(not indices for indices in slot_variables.values()):
        return pd.DataFrame()
    player_variables = {
        player_index: [
            i
            for i, (candidate, _) in enumerate(assignments)
            if candidate == player_index
        ]
        for player_index in pool.index
    }

    objective = np.array(
        [
            -float(pool.at[player_index, "ProjectedPoints"])
            + float(pool.at[player_index, "Salary"]) * 1e-9
            + player_index * 1e-12
            for player_index, _ in assignments
        ],
        dtype=float,
    )
    base_rows: list[tuple[dict[int, float], float, float]] = []
    for indices in slot_variables.values():
        base_rows.append(({index: 1.0 for index in indices}, 1.0, 1.0))
    for indices in player_variables.values():
        base_rows.append(({index: 1.0 for index in indices}, -np.inf, 1.0))
    base_rows.append(
        (
            {
                variable_index: float(pool.at[player_index, "Salary"])
                for variable_index, (player_index, _) in enumerate(assignments)
            },
            -np.inf,
            float(salary_cap),
        )
    )

    # DraftKings Classic requires players from at least two teams. With nine
    # roster spots, limiting any one team to eight is the equivalent linear rule.
    for team in sorted(pool["Team"].unique()):
        team_indices = [
            variable_index
            for variable_index, (player_index, _) in enumerate(assignments)
            if pool.at[player_index, "Team"] == team
        ]
        if team_indices:
            base_rows.append(({index: 1.0 for index in team_indices}, -np.inf, 8.0))

    if require_qb_pass_catcher:
        for qb_index in pool.index[pool["Position"].eq("QB")]:
            qb_indices = player_variables[qb_index]
            pass_catcher_indices = [
                variable_index
                for variable_index, (player_index, _) in enumerate(assignments)
                if pool.at[player_index, "Team"] == pool.at[qb_index, "Team"]
                and pool.at[player_index, "Position"] in {"WR", "TE"}
            ]
            coefficients = {index: 1.0 for index in qb_indices}
            for index in pass_catcher_indices:
                coefficients[index] = coefficients.get(index, 0.0) - 1.0
            base_rows.append((coefficients, -np.inf, 0.0))

    if avoid_offense_against_dst:
        maximum_offensive_players = len(DK_NFL_CLASSIC_ROSTER_SLOTS) - 1
        for dst_index in pool.index[pool["Position"].eq("DST")]:
            opponent = _mlb_opponent_team(
                pool.at[dst_index, "Team"], pool.at[dst_index, "Game Info"]
            )
            if not opponent:
                continue
            offense_indices = [
                variable_index
                for variable_index, (player_index, _) in enumerate(assignments)
                if pool.at[player_index, "Team"] == opponent
                and pool.at[player_index, "Position"] != "DST"
            ]
            if not offense_indices:
                continue
            coefficients = {index: 1.0 for index in offense_indices}
            coefficients.update(
                {
                    index: float(maximum_offensive_players)
                    for index in player_variables[dst_index]
                }
            )
            base_rows.append(
                (coefficients, -np.inf, float(maximum_offensive_players))
            )

    lineup_rows: list[dict[str, object]] = []
    prior_player_sets: list[frozenset[int]] = []
    for lineup_rank in range(1, int(top_n) + 1):
        rows = list(base_rows)
        for prior_players in prior_player_sets:
            no_good_indices = [
                variable_index
                for variable_index, (player_index, _) in enumerate(assignments)
                if player_index in prior_players
            ]
            rows.append(
                (
                    {index: 1.0 for index in no_good_indices},
                    -np.inf,
                    len(DK_NFL_CLASSIC_ROSTER_SLOTS) - 1.0,
                )
            )

        matrix = np.zeros((len(rows), assignment_count), dtype=float)
        lower = np.empty(len(rows), dtype=float)
        upper = np.empty(len(rows), dtype=float)
        for row_index, (coefficients, lower_bound, upper_bound) in enumerate(rows):
            if coefficients:
                indices = list(coefficients)
                matrix[row_index, indices] = [coefficients[index] for index in indices]
            lower[row_index] = lower_bound
            upper[row_index] = upper_bound

        result = milp(
            c=objective,
            integrality=np.ones(assignment_count, dtype=int),
            bounds=Bounds(np.zeros(assignment_count), np.ones(assignment_count)),
            constraints=LinearConstraint(csr_matrix(matrix), lower, upper),
            options={"time_limit": 30.0, "mip_rel_gap": 0.0},
        )
        if result.x is None or not result.success:
            break

        selected_variables = [
            variable_index
            for variable_index, value in enumerate(result.x)
            if value > 0.5
        ]
        selected_players = frozenset(
            assignments[variable_index][0] for variable_index in selected_variables
        )
        if len(selected_players) != len(DK_NFL_CLASSIC_ROSTER_SLOTS):
            break
        prior_player_sets.append(selected_players)

        selected_by_slot = {
            assignments[variable_index][1]: assignments[variable_index][0]
            for variable_index in selected_variables
        }
        # Repeated RB and WR slots are symmetric; sort them for stable exports.
        for prefix in ("RB", "WR"):
            slots = [slot for slot in DK_NFL_CLASSIC_ROSTER_SLOTS if slot.startswith(prefix)]
            indices = sorted(
                (selected_by_slot[slot] for slot in slots),
                key=lambda index: (
                    -float(pool.at[index, "ProjectedPoints"]),
                    str(pool.at[index, "Name"]),
                ),
            )
            for slot, player_index in zip(slots, indices):
                selected_by_slot[slot] = player_index

        lineup: dict[str, object] = {"Lineup Rank": lineup_rank}
        ordered_players: list[int] = []
        for slot in DK_NFL_CLASSIC_ROSTER_SLOTS:
            player_index = selected_by_slot[slot]
            ordered_players.append(player_index)
            name_id = str(pool.at[player_index, "Name + ID"] or "").strip()
            lineup[slot] = name_id or str(pool.at[player_index, "Name"])

        salaries = pool.loc[ordered_players, "Salary"].astype(float)
        projections = pool.loc[ordered_players, "ProjectedPoints"].astype(float)
        lineup["Salary"] = int(salaries.sum())
        lineup["Unused Salary"] = int(salary_cap - salaries.sum())
        lineup["Projected Points"] = round(float(projections.sum()), 3)
        lineup["Teams"] = int(pool.loc[ordered_players, "Team"].nunique())
        lineup["Unique Players"] = len(selected_players)
        lineup["QB Stack Team"] = str(pool.at[selected_by_slot["QB"], "Team"])
        lineup["Projection Sources"] = ", ".join(
            sorted(pool.loc[ordered_players, "ProjectionSource"].astype(str).unique())
        )
        lineup["Lineup Key"] = "|".join(
            sorted(
                str(pool.at[index, "ID"] or pool.at[index, "Name"]).strip()
                for index in selected_players
            )
        )
        lineup_rows.append(lineup)

    return pd.DataFrame(lineup_rows)


def parse_draftkings_mlb_classic_salary_csv(source: Any) -> pd.DataFrame:
    """Normalize an official DraftKings MLB Classic player-pool CSV.

    Multi-position hitters remain eligible for every roster position listed in
    either DraftKings' Position or Roster Position field. SP/RP normalize to P,
    and LF/CF/RF normalize to OF. Uploaded projections are preferred; DraftKings
    average fantasy points per game is retained only as a labeled fallback.
    """

    raw = _read_salary_source(source)
    columns = {
        name: _find_column(raw, aliases)
        for name, aliases in _COLUMN_ALIASES.items()
    }
    missing = [name for name in ("name", "salary") if not columns[name]]
    if not columns["position"] and not columns["roster_position"]:
        missing.append("position")
    if missing:
        raise ValueError(
            "DraftKings MLB salary CSV is missing required column(s): "
            + ", ".join(missing)
        )

    index = raw.index
    source_positions = (
        raw[columns["position"]].fillna("").astype(str)
        if columns["position"]
        else pd.Series("", index=index)
    )
    roster_positions = (
        raw[columns["roster_position"]].fillna("").astype(str)
        if columns["roster_position"]
        else source_positions
    )

    def eligible_positions(position: object, roster: object) -> tuple[str, ...]:
        combined: list[str] = []
        for value in (roster, position):
            for token in _mlb_position_tokens(value):
                if token not in combined:
                    combined.append(token)
        return tuple(combined)

    eligible = pd.Series(
        [
            eligible_positions(position, roster)
            for position, roster in zip(source_positions, roster_positions)
        ],
        index=index,
    )
    out = pd.DataFrame(index=index)
    out["Position"] = eligible.map(lambda values: values[0] if values else "")
    out["EligiblePositions"] = eligible.map(lambda values: "/".join(values))
    out["Name"] = raw[columns["name"]].fillna("").astype(str).str.strip()
    out["Name + ID"] = (
        raw[columns["name_id"]].fillna("").astype(str).str.strip()
        if columns["name_id"]
        else out["Name"]
    )
    out["ID"] = (
        raw[columns["id"]].fillna("").astype(str).str.strip()
        if columns["id"]
        else ""
    )
    out["Roster Position"] = roster_positions.str.strip()
    out["Salary"] = pd.to_numeric(raw[columns["salary"]], errors="coerce")
    out["Game Info"] = (
        raw[columns["game_info"]].fillna("").astype(str).str.strip()
        if columns["game_info"]
        else ""
    )
    out["Team"] = (
        raw[columns["team"]].fillna("").astype(str).str.strip().str.upper()
        if columns["team"]
        else ""
    )
    out["AvgPointsPerGame"] = (
        pd.to_numeric(raw[columns["average_points"]], errors="coerce")
        if columns["average_points"]
        else pd.Series(float("nan"), index=index)
    )
    supplied_projection = (
        pd.to_numeric(raw[columns["projection"]], errors="coerce")
        if columns["projection"]
        else pd.Series(float("nan"), index=index)
    )
    out["ProjectedPoints"] = supplied_projection.fillna(out["AvgPointsPerGame"])
    out["ProjectionSource"] = "unavailable"
    out.loc[supplied_projection.notna(), "ProjectionSource"] = "uploaded_projection"
    out.loc[
        supplied_projection.isna() & out["AvgPointsPerGame"].notna(),
        "ProjectionSource",
    ] = "draftkings_average_fppg"
    out["Status"] = (
        raw[columns["status"]].fillna("").astype(str).str.strip().str.upper()
        if columns["status"]
        else ""
    )
    out["Starting"] = (
        raw[columns["starting"]].fillna("").astype(str).str.strip().str.upper()
        if columns["starting"]
        else ""
    )

    active = ~out["Status"].isin(
        {
            "O",
            "OUT",
            "IR",
            "IL",
            "INACTIVE",
            "SUSPENDED",
            "PPD",
            "POSTPONED",
            "CANCELLED",
        }
    )
    valid_position = out["EligiblePositions"].ne("")
    valid_identity = out["Name"].ne("")
    valid_salary = out["Salary"].notna() & out["Salary"].gt(0)
    out = out[active & valid_position & valid_identity & valid_salary].copy()
    out["Salary"] = out["Salary"].astype(int)
    out["ValuePer1000"] = (
        out["ProjectedPoints"] * 1000.0 / out["Salary"]
    ).round(3)
    return out.reset_index(drop=True)


def _mlb_slot_position(slot: str) -> str:
    if slot.startswith("P"):
        return "P"
    if slot.startswith("OF"):
        return "OF"
    return slot


def _mlb_opponent_team(team: object, game_info: object) -> str:
    matchup = str(game_info or "").strip().upper().split(" ", 1)[0]
    parts = re.split(r"@|VS\.?", matchup)
    if len(parts) != 2:
        return ""
    away, home = (part.strip() for part in parts)
    normalized_team = str(team or "").strip().upper()
    if normalized_team == away:
        return home
    if normalized_team == home:
        return away
    return ""


def build_draftkings_mlb_classic_lineups(
    player_pool: pd.DataFrame,
    *,
    top_n: int = 5,
    salary_cap: int = DK_CLASSIC_SALARY_CAP,
    max_hitters_per_team: int = 5,
    avoid_pitcher_hitter_conflicts: bool = True,
) -> pd.DataFrame:
    """Optimize the top complete DraftKings MLB Classic lineups.

    Every lineup contains two pitchers and eight hitters in the standard Classic
    roster, stays at or below the supplied salary cap, uses each player once,
    and includes no more than five hitters from one team. By default, a lineup
    cannot roster a hitter opposing one of its pitchers. Repeated solves add a
    no-good constraint on the player set, yielding the next-best unique lineup.
    """

    if player_pool is None or player_pool.empty:
        return pd.DataFrame()
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    if int(salary_cap) <= 0:
        raise ValueError("salary_cap must be positive")
    if int(max_hitters_per_team) <= 0:
        raise ValueError("max_hitters_per_team must be positive")

    required = {
        "Position",
        "EligiblePositions",
        "Name",
        "Salary",
        "ProjectedPoints",
        "ValuePer1000",
        "Team",
        "Game Info",
    }
    missing = sorted(required.difference(player_pool.columns))
    if missing:
        raise ValueError("Normalized MLB player pool is missing: " + ", ".join(missing))

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError(
            "MLB Classic lineup optimization requires scipy."
        ) from exc

    pool = player_pool.copy()
    pool["Salary"] = pd.to_numeric(pool["Salary"], errors="coerce")
    pool["ProjectedPoints"] = pd.to_numeric(
        pool["ProjectedPoints"], errors="coerce"
    )
    pool["Team"] = pool["Team"].fillna("").astype(str).str.strip().str.upper()
    pool = pool[
        pool["Salary"].notna()
        & pool["Salary"].gt(0)
        & pool["ProjectedPoints"].notna()
        & pool["Team"].ne("")
    ].reset_index(drop=True)
    # The official MLB export includes every reliever, but identifies the
    # slate's probable starters in its Starting column. Once DraftKings has
    # populated at least two probable starters, only those players may fill P;
    # otherwise the optimizer safely retains the full P pool for early slates.
    if "Starting" in pool.columns:
        probable_pitcher = (
            pool["Starting"].fillna("").astype(str).str.upper().isin({"P", "SP"})
        )
        if int(probable_pitcher.sum()) >= 2:
            pitcher = pool["EligiblePositions"].fillna("").astype(str).map(
                lambda value: "P" in _mlb_position_tokens(value)
            )
            pool = pool[~pitcher | probable_pitcher].reset_index(drop=True)
    if pool.empty:
        return pd.DataFrame()

    eligibility = pool["EligiblePositions"].fillna("").astype(str).map(
        lambda value: frozenset(_mlb_position_tokens(value))
    )
    assignments: list[tuple[int, str]] = []
    for player_index, positions in eligibility.items():
        for slot in DK_MLB_CLASSIC_ROSTER_SLOTS:
            if _mlb_slot_position(slot) in positions:
                assignments.append((int(player_index), slot))
    if not assignments:
        return pd.DataFrame()

    assignment_count = len(assignments)
    slot_variables = {
        slot: [
            i
            for i, (_, assignment_slot) in enumerate(assignments)
            if assignment_slot == slot
        ]
        for slot in DK_MLB_CLASSIC_ROSTER_SLOTS
    }
    if any(not indices for indices in slot_variables.values()):
        return pd.DataFrame()
    player_variables = {
        player_index: [
            i
            for i, (candidate, _) in enumerate(assignments)
            if candidate == player_index
        ]
        for player_index in pool.index
    }

    objective = np.array(
        [
            -float(pool.at[player_index, "ProjectedPoints"])
            + float(pool.at[player_index, "Salary"]) * 1e-9
            + player_index * 1e-12
            for player_index, _ in assignments
        ],
        dtype=float,
    )

    base_rows: list[tuple[dict[int, float], float, float]] = []
    for indices in slot_variables.values():
        base_rows.append(({index: 1.0 for index in indices}, 1.0, 1.0))
    for indices in player_variables.values():
        base_rows.append(({index: 1.0 for index in indices}, -np.inf, 1.0))
    base_rows.append(
        (
            {
                variable_index: float(pool.at[player_index, "Salary"])
                for variable_index, (player_index, _) in enumerate(assignments)
            },
            -np.inf,
            float(salary_cap),
        )
    )

    teams = sorted(pool["Team"].unique())
    for team in teams:
        hitter_indices = [
            variable_index
            for variable_index, (player_index, slot) in enumerate(assignments)
            if pool.at[player_index, "Team"] == team
            and _mlb_slot_position(slot) != "P"
        ]
        if hitter_indices:
            base_rows.append(
                (
                    {index: 1.0 for index in hitter_indices},
                    -np.inf,
                    float(max_hitters_per_team),
                )
            )

    if avoid_pitcher_hitter_conflicts:
        for pitcher_index in pool.index:
            pitcher_vars = [
                variable_index
                for variable_index in player_variables[pitcher_index]
                if _mlb_slot_position(assignments[variable_index][1]) == "P"
            ]
            if not pitcher_vars:
                continue
            opponent = _mlb_opponent_team(
                pool.at[pitcher_index, "Team"], pool.at[pitcher_index, "Game Info"]
            )
            if not opponent:
                continue
            opposing_hitter_vars = [
                variable_index
                for variable_index, (player_index, slot) in enumerate(assignments)
                if pool.at[player_index, "Team"] == opponent
                and _mlb_slot_position(slot) != "P"
            ]
            if opposing_hitter_vars:
                maximum_opposing_hitters = len(
                    [
                        slot
                        for slot in DK_MLB_CLASSIC_ROSTER_SLOTS
                        if _mlb_slot_position(slot) != "P"
                    ]
                )
                coefficients = {
                    index: 1.0 for index in opposing_hitter_vars
                }
                coefficients.update(
                    {
                        index: float(maximum_opposing_hitters)
                        for index in pitcher_vars
                    }
                )
                base_rows.append(
                    (coefficients, -np.inf, float(maximum_opposing_hitters))
                )

    lineup_rows: list[dict[str, object]] = []
    prior_player_sets: list[frozenset[int]] = []

    for lineup_rank in range(1, int(top_n) + 1):
        rows = list(base_rows)
        for prior_players in prior_player_sets:
            indices = [
                variable_index
                for variable_index, (player_index, _) in enumerate(assignments)
                if player_index in prior_players
            ]
            rows.append(
                (
                    {index: 1.0 for index in indices},
                    -np.inf,
                    len(DK_MLB_CLASSIC_ROSTER_SLOTS) - 1.0,
                )
            )

        matrix = np.zeros((len(rows), assignment_count), dtype=float)
        lower = np.empty(len(rows), dtype=float)
        upper = np.empty(len(rows), dtype=float)
        for row_index, (coefficients, lower_bound, upper_bound) in enumerate(rows):
            if coefficients:
                indices = list(coefficients)
                matrix[row_index, indices] = [coefficients[index] for index in indices]
            lower[row_index] = lower_bound
            upper[row_index] = upper_bound

        result = milp(
            c=objective,
            integrality=np.ones(assignment_count, dtype=int),
            bounds=Bounds(np.zeros(assignment_count), np.ones(assignment_count)),
            constraints=LinearConstraint(csr_matrix(matrix), lower, upper),
            options={"time_limit": 30.0, "mip_rel_gap": 0.0},
        )
        if result.x is None or not result.success:
            break

        selected_variables = [
            variable_index
            for variable_index, value in enumerate(result.x)
            if value > 0.5
        ]
        selected_players = frozenset(
            assignments[variable_index][0] for variable_index in selected_variables
        )
        selected_hitter_players = frozenset(
            assignments[variable_index][0]
            for variable_index in selected_variables
            if _mlb_slot_position(assignments[variable_index][1]) != "P"
        )
        if len(selected_players) != len(DK_MLB_CLASSIC_ROSTER_SLOTS):
            break
        prior_player_sets.append(selected_players)

        selected_by_position: dict[str, list[int]] = {}
        for variable_index in selected_variables:
            player_index, slot = assignments[variable_index]
            selected_by_position.setdefault(_mlb_slot_position(slot), []).append(
                player_index
            )
        for player_indices in selected_by_position.values():
            player_indices.sort(
                key=lambda index: (
                    -float(pool.at[index, "ProjectedPoints"]),
                    str(pool.at[index, "Name"]),
                )
            )

        lineup: dict[str, object] = {"Lineup Rank": lineup_rank}
        position_offsets = {"P": 0, "OF": 0}
        ordered_players: list[int] = []
        for slot in DK_MLB_CLASSIC_ROSTER_SLOTS:
            position = _mlb_slot_position(slot)
            offset = position_offsets.get(position, 0)
            player_index = selected_by_position[position][offset]
            if position in position_offsets:
                position_offsets[position] += 1
            ordered_players.append(player_index)
            name_id = str(pool.at[player_index, "Name + ID"] or "").strip()
            lineup[slot] = name_id or str(pool.at[player_index, "Name"])

        salaries = pool.loc[ordered_players, "Salary"].astype(float)
        projections = pool.loc[ordered_players, "ProjectedPoints"].astype(float)
        hitter_teams = pool.loc[list(selected_hitter_players), "Team"]
        team_counts = hitter_teams.value_counts()
        lineup["Salary"] = int(salaries.sum())
        lineup["Unused Salary"] = int(salary_cap - salaries.sum())
        lineup["Projected Points"] = round(float(projections.sum()), 3)
        lineup["Teams"] = int(pool.loc[ordered_players, "Team"].nunique())
        lineup["Max Hitters / Team"] = int(team_counts.max()) if not team_counts.empty else 0
        lineup["Unique Players"] = len(selected_players)
        lineup["Projection Sources"] = ", ".join(
            sorted(pool.loc[ordered_players, "ProjectionSource"].astype(str).unique())
        )
        lineup["Lineup Key"] = "|".join(
            sorted(
                str(pool.at[index, "ID"] or pool.at[index, "Name"]).strip()
                for index in selected_players
            )
        )
        lineup_rows.append(lineup)

    return pd.DataFrame(lineup_rows)
