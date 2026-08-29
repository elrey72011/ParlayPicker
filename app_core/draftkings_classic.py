"""DraftKings NFL Classic player-pool parsing and positional shortlists.

DraftKings Classic salaries are contest/slate specific, so the official contest
CSV is the source of truth. This module does not scrape or invent salaries. It
normalizes the downloaded player pool and ranks the five strongest options at
each roster position, including a separate RB/WR/TE FLEX shortlist.
"""
from __future__ import annotations

from io import BytesIO, StringIO
from typing import Any

import pandas as pd


DK_CLASSIC_SALARY_CAP = 50_000
DK_CLASSIC_POSITION_ORDER = ("QB", "RB", "WR", "TE", "FLEX", "DST")
DK_CLASSIC_FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})

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
        projection_span = (
            ranked["ProjectedPoints"].max() - ranked["ProjectedPoints"].min()
        )
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
        ranked["ClassicScore"] = (
            0.70 * projection_score + 0.30 * value_score
        ).round(4)
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
