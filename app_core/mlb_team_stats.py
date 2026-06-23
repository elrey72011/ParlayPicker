"""Genuine MLB team features for the live model.

Until now MLB games hit the model with league-default stats (4.5 runs, .500) because no real
MLB team feed was wired in — so the model had little independent signal and largely echoed
the market. This module populates the model's existing team-stat features (win%, runs
scored/allowed per game, streak, recent form) from the FREE MLB StatsAPI standings endpoint
(same source the strikeout-props slice already uses; no API key required).

It fills the EXISTING VERTEX feature columns the XGBoost model was trained on — it does NOT
invent new features (the model can't consume those). Every fetch degrades to "leave the row
as-is" (keeping the prior default) so a feed hiccup can never break prediction.

StatsAPI standings teamRecord shape (the fields we use):
  team.name, wins, losses, gamesPlayed, runsScored, runsAllowed,
  streak.streakType ('wins'|'losses') + streak.streakNumber,
  records.splitRecords[type=='lastTen'].wins/losses
"""
from __future__ import annotations

import re
from typing import Any, Callable

import requests

_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10
_REG_SEASON_LEAGUES = "103,104"  # AL, NL


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def parse_standings(payload: dict) -> dict[str, dict]:
    """Parse a StatsAPI /standings payload into {normalized_team_name: stats}.

    Pure. Defensive: a team missing runs/streak/last-ten yields None for that field (the
    caller leaves the corresponding feature at its default) rather than guessing.
    """
    out: dict[str, dict] = {}
    for record in (payload or {}).get("records", []) or []:
        for tr in record.get("teamRecords", []) or []:
            team = tr.get("team") or {}
            name = team.get("name")
            if not name:
                continue
            wins = tr.get("wins")
            losses = tr.get("losses")
            gp = tr.get("gamesPlayed")
            if not gp:
                gp = (wins or 0) + (losses or 0)

            def _per_game(key):
                v = tr.get(key)
                try:
                    return float(v) / gp if v is not None and gp else None
                except (TypeError, ValueError, ZeroDivisionError):
                    return None

            win_pct = None
            try:
                if (wins is not None) and gp:
                    win_pct = float(wins) / gp
            except (TypeError, ValueError, ZeroDivisionError):
                win_pct = None

            streak = None
            sk = tr.get("streak") or {}
            if sk.get("streakNumber") is not None:
                n = int(sk["streakNumber"])
                streak = n if str(sk.get("streakType", "")).lower().startswith("w") else -n

            last10 = None
            for sr in (tr.get("records") or {}).get("splitRecords", []) or []:
                if str(sr.get("type", "")).lower() == "lastten":
                    lw, ll = sr.get("wins"), sr.get("losses")
                    if lw is not None and (lw + (ll or 0)) > 0:
                        last10 = float(lw) / (lw + ll)
                    break

            out[_norm(name)] = {
                "name": name,
                "win_pct": win_pct,
                "ppg": _per_game("runsScored"),
                "oppg": _per_game("runsAllowed"),
                "streak": streak,
                "last10_pct": last10,
            }
    return out


def fetch_mlb_team_stats(season: int, *, http_get: Callable = requests.get) -> dict[str, dict]:
    """Fetch + parse current MLB standings into {normalized_team_name: stats}. {} on failure."""
    try:
        resp = http_get(
            f"{_BASE}/standings",
            params={"leagueId": _REG_SEASON_LEAGUES, "season": season, "standingsTypes": "regularSeason"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return parse_standings(resp.json())
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return {}


def _resolve(team_str: str, stats: dict[str, dict]) -> dict | None:
    """Match a df team string to a standings team.

    StatsAPI returns nicknames only ("Red Sox", "Reds"), so we match the df name (often the
    full "Boston Red Sox") by containment. Critically we take the LONGEST containment match,
    not the first: "bostonredsox" contains BOTH "redsox" and "reds", and dict order could
    otherwise pick the wrong club. Exact match always wins first.
    """
    key = _norm(team_str)
    if not key:
        return None
    if key in stats:
        return stats[key]
    best, best_len = None, 0
    for sk, sv in stats.items():
        if not sk:
            continue
        if sk in key:
            m = len(sk)
        elif key in sk:
            m = len(key)
        else:
            continue
        if m > best_len:
            best, best_len = sv, m
    return best


# Feature columns this module populates (subset of VERTEX_FEATURE_COLUMNS), by signal.
def enrich_mlb_model_features(
    df,
    *,
    stats: dict[str, dict] | None = None,
    season: int | None = None,
    fetch: Callable = fetch_mlb_team_stats,
):
    """Fill the model's team-stat features for MLB rows from real standings.

    Operates on the dataframe that feeds predict_batch. MLB rows whose home AND away teams
    resolve get real win%/ppg/oppg/streak/form (and diffs); everything else is untouched, so
    non-MLB rows and unresolved teams keep their existing values. Returns the same df. Never
    raises — on any error the df is returned unchanged (model keeps its prior features).
    """
    try:
        import pandas as pd  # local import keeps module import-light

        if df is None or len(df) == 0:
            return df
        league_col = "league" if "league" in df.columns else ("League" if "League" in df.columns else None)
        home_col = "home_team" if "home_team" in df.columns else ("Home" if "Home" in df.columns else None)
        away_col = "away_team" if "away_team" in df.columns else ("Away" if "Away" in df.columns else None)
        if not (league_col and home_col and away_col):
            return df

        mlb_mask = df[league_col].astype(str).str.upper().eq("MLB")
        if not mlb_mask.any():
            return df

        if stats is None:
            if season is None:
                season = _infer_season(df)
            stats = fetch(season)
        if not stats:
            return df

        resolved = 0
        for idx in df.index[mlb_mask]:
            h = _resolve(df.at[idx, home_col], stats)
            a = _resolve(df.at[idx, away_col], stats)
            if not h or not a:
                continue
            _set(df, idx, "feature_home_win_pct", h["win_pct"])
            _set(df, idx, "feature_away_win_pct", a["win_pct"])
            _set(df, idx, "feature_home_ppg", h["ppg"])
            _set(df, idx, "feature_away_ppg", a["ppg"])
            _set(df, idx, "feature_home_oppg", h["oppg"])
            _set(df, idx, "feature_away_oppg", a["oppg"])
            _set(df, idx, "feature_home_streak", h["streak"])
            _set(df, idx, "feature_away_streak", a["streak"])
            _set_diff(df, idx, "feature_diff_win_pct", h["win_pct"], a["win_pct"])
            _set_diff(df, idx, "feature_diff_ppg", h["ppg"], a["ppg"])
            _set_diff(df, idx, "feature_diff_oppg", h["oppg"], a["oppg"])
            _set_diff(df, idx, "feature_diff_streak", h["streak"], a["streak"])
            _set_diff(df, idx, "feature_diff_last5", h["last10_pct"], a["last10_pct"])
            df.at[idx, "stats_source"] = "mlb_statsapi"
            resolved += 1
        return df
    except Exception:
        return df


def _set(df, idx, col, val):
    if val is not None:
        df.at[idx, col] = float(val)


def _set_diff(df, idx, col, hv, av):
    if hv is not None and av is not None:
        df.at[idx, col] = float(hv) - float(av)


def _infer_season(df) -> int:
    import pandas as pd
    for c in ("game_date", "Local Date", "Commence (Local)"):
        if c in df.columns:
            yr = pd.to_datetime(df[c], errors="coerce").dt.year.dropna()
            if not yr.empty:
                return int(yr.mode().iloc[0])
    from datetime import datetime
    return datetime.now().year
