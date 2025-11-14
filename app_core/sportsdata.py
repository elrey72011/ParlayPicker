"""Thin client for SportsData.io NFL feeds used by the Streamlit app."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class SportsDataTeamInsight:
    """Snapshot of team-level metrics returned by SportsData.io."""

    name: str
    alias: Optional[str]
    record: Optional[str]
    streak: Optional[str]
    trend: Optional[str]
    points_for_per_game: Optional[float]
    points_against_per_game: Optional[float]
    net_points_per_game: Optional[float]
    turnover_margin: Optional[float]
    power_index: Optional[float]


@dataclass
class SportsDataGameInsight:
    """Game metadata paired with home/away team insights."""

    game_key: Optional[str]
    season: Optional[str]
    season_type: Optional[str]
    week: Optional[int]
    status: Optional[str]
    kickoff: Optional[str]
    stadium: Optional[str]
    home: SportsDataTeamInsight
    away: SportsDataTeamInsight


class SportsDataNFLClient:
    """Lightweight helper that wraps the SportsData.io NFL API."""

    BASE_URL = "https://api.sportsdata.io/v3/nfl"
    SECRET_ENV_PRIORITY: Tuple[str, ...] = (
        "NFL_SPORTSDATA_API_KEY",
        "SPORTSDATA_NFL_KEY",
        "SPORTSDATA_API_KEY",
        "SPORTSDATA_KEY",
    )
    DEFAULT_SEASON_TYPE = "REG"
    RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        key_source: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or ""
        self.key_source: Optional[str] = key_source

        if not resolved_key:
            for env_name in self.SECRET_ENV_PRIORITY:
                candidate = os.environ.get(env_name)
                if candidate:
                    resolved_key = candidate
                    self.key_source = f"env:{env_name}"
                    break

        if resolved_key and not self.key_source:
            self.key_source = "runtime"

        self.api_key = resolved_key or ""
        self.session = session or requests.Session()
        if self.api_key:
            self.session.headers.update({"Ocp-Apim-Subscription-Key": self.api_key})

        self.timeout = 10
        self.max_retries = 3
        self.backoff_seconds = 1.5

        self._scores_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._team_stats_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def key_origin(self) -> Optional[str]:
        return self.key_source

    def update_api_key(self, api_key: Optional[str], source: Optional[str] = None) -> None:
        self.api_key = api_key or ""
        if self.api_key:
            self.key_source = source or "runtime"
            self.session.headers.update({"Ocp-Apim-Subscription-Key": self.api_key})
        else:
            self.key_source = None
            self.session.headers.pop("Ocp-Apim-Subscription-Key", None)
        self._scores_cache.clear()
        self._team_stats_cache.clear()
        self.last_error = None

    # ------------------------------------------------------------------
    # Networking helpers
    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if not self.api_key:
            self.last_error = "missing_api_key"
            return None

        url = f"{self.BASE_URL}{path}"
        params = params.copy() if params else {}
        attempt = 0
        while attempt <= self.max_retries:
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in self.RETRY_STATUS and attempt < self.max_retries:
                    delay = self.backoff_seconds * (attempt + 1)
                    time.sleep(delay)
                    attempt += 1
                    continue
                resp.raise_for_status()
                data = resp.json()
                self.last_error = None
                return data
            except requests.HTTPError as exc:
                self.last_error = f"http_{exc.response.status_code if exc.response else 'error'}"
                if (
                    exc.response is not None
                    and exc.response.status_code in self.RETRY_STATUS
                    and attempt < self.max_retries
                ):
                    delay = self.backoff_seconds * (attempt + 1)
                    time.sleep(delay)
                    attempt += 1
                    continue
                return None
            except Exception:
                self.last_error = "request_failed"
                return None
        return None

    # ------------------------------------------------------------------
    # Data fetchers
    def get_scores_by_date(self, target_date: date) -> List[Dict[str, Any]]:
        date_key = target_date.isoformat()
        if date_key in self._scores_cache:
            return self._scores_cache[date_key]

        payload = self._request(f"/scores/json/ScoresByDate/{date_key}")
        games: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            games = payload
        self._scores_cache[date_key] = games
        return games

    def _season_tokens(self, season: Optional[str], season_type: Optional[str]) -> List[str]:
        tokens: List[str] = []
        base = str(season) if season else ""
        season_type = (season_type or self.DEFAULT_SEASON_TYPE or "").upper()
        if base:
            tokens.append(base)
            tokens.append(f"{base}{season_type}")
            tokens.append(f"{base}-{season_type}")
        return [token for token in tokens if token]

    def _fetch_team_stats(
        self, season: Optional[str], season_type: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        key = (str(season or ""), (season_type or self.DEFAULT_SEASON_TYPE or "").upper())
        if key in self._team_stats_cache:
            return self._team_stats_cache[key]

        season_tokens = self._season_tokens(*key)
        stats_payload: Optional[List[Dict[str, Any]]] = None
        for token in season_tokens or [None]:
            if not token:
                continue
            candidate = self._request(f"/stats/json/TeamSeasonStats/{token}")
            if isinstance(candidate, list) and candidate:
                stats_payload = candidate
                break

        stats_map: Dict[str, Dict[str, Any]] = {}
        if stats_payload:
            for entry in stats_payload:
                if not isinstance(entry, dict):
                    continue
                identifiers = {
                    entry.get("Team"),
                    entry.get("Name"),
                    entry.get("Key"),
                    entry.get("GlobalTeamID"),
                }
                for ident in identifiers:
                    norm = self._normalize_name(ident)
                    if norm:
                        stats_map[norm] = entry
        self._team_stats_cache[key] = stats_map
        return stats_map

    # ------------------------------------------------------------------
    # Matching helpers
    @staticmethod
    def _normalize_name(value: Optional[str]) -> str:
        if value is None:
            return ""
        return re.sub(r"[^A-Z]", "", str(value).upper())

    def match_game(
        self,
        games: List[Dict[str, Any]],
        home: str,
        away: str,
    ) -> Optional[Dict[str, Any]]:
        if not games:
            return None

        home_norm = self._normalize_name(home)
        away_norm = self._normalize_name(away)
        if not home_norm or not away_norm:
            return None

        for game in games:
            if not isinstance(game, dict):
                continue
            game_home = self._normalize_name(game.get("HomeTeam") or game.get("HomeTeamName"))
            game_away = self._normalize_name(game.get("AwayTeam") or game.get("AwayTeamName"))
            if game_home == home_norm and game_away == away_norm:
                return game
        return None

    # ------------------------------------------------------------------
    # Insight builders
    def build_game_insight(
        self,
        game: Dict[str, Any],
        team_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[SportsDataGameInsight]:
        if not isinstance(game, dict):
            return None

        season = game.get("Season")
        season_type = game.get("SeasonType") or self.DEFAULT_SEASON_TYPE
        week = game.get("Week")
        kickoff = game.get("Date") or game.get("DateTime")
        status = game.get("Status")
        stadium = None
        venue = game.get("StadiumDetails")
        if isinstance(venue, dict):
            stadium = venue.get("Name") or venue.get("Stadium")
        else:
            stadium = game.get("Stadium")

        stats_map = team_stats or self._fetch_team_stats(season, season_type)

        home_team = self._build_team_insight(game, stats_map, side="Home")
        away_team = self._build_team_insight(game, stats_map, side="Away")
        if not home_team or not away_team:
            return None

        return SportsDataGameInsight(
            game_key=str(game.get("GameKey") or game.get("GlobalGameID") or ""),
            season=str(season) if season else None,
            season_type=str(season_type) if season_type else None,
            week=int(week) if isinstance(week, (int, float)) else None,
            status=status if isinstance(status, str) else None,
            kickoff=str(kickoff) if kickoff else None,
            stadium=stadium if isinstance(stadium, str) else None,
            home=home_team,
            away=away_team,
        )

    def _build_team_insight(
        self,
        game: Dict[str, Any],
        stats_map: Dict[str, Dict[str, Any]],
        side: str,
    ) -> Optional[SportsDataTeamInsight]:
        team_key = f"{side}Team"
        team_name = game.get(team_key) or game.get(f"{team_key}Name")
        if not team_name:
            return None

        alias = game.get(f"{side}TeamID") or game.get(f"{side}TeamKey")
        normalized = self._normalize_name(team_name)
        stats_entry = stats_map.get(normalized)

        record = None
        streak_text = None
        trend = None
        points_for_pg = None
        points_against_pg = None
        net_points_pg = None
        turnover_margin = None
        power_index = None

        if stats_entry:
            wins = stats_entry.get("Wins") or stats_entry.get("Win")
            losses = stats_entry.get("Losses") or stats_entry.get("Loss")
            ties = stats_entry.get("Ties")
            if any(isinstance(val, (int, float)) for val in (wins, losses, ties)):
                wins = int(wins or 0)
                losses = int(losses or 0)
                ties = int(ties or 0)
                record = f"{wins}-{losses}"
                if ties:
                    record += f"-{ties}"

            streak_text = stats_entry.get("StreakDescription") or stats_entry.get("Streak")
            if isinstance(streak_text, (int, float)):
                streak_val = int(streak_text)
                if streak_val > 0:
                    streak_text = f"Won {streak_val}" if streak_val > 1 else "Won 1"
                elif streak_val < 0:
                    streak_text = f"Lost {abs(streak_val)}"
                else:
                    streak_text = "Even"

            if isinstance(streak_text, str):
                lower = streak_text.lower()
                if "win" in lower:
                    trend = "hot"
                elif "lose" in lower or "loss" in lower:
                    trend = "cold"
                else:
                    trend = "neutral"

            games_played = stats_entry.get("Games") or stats_entry.get("GamesPlayed")
            if isinstance(games_played, (int, float)) and games_played:
                pf = stats_entry.get("PointsFor") or stats_entry.get("PointsScored")
                pa = stats_entry.get("PointsAgainst") or stats_entry.get("PointsAllowed")
                if isinstance(pf, (int, float)):
                    points_for_pg = float(pf) / float(games_played)
                if isinstance(pa, (int, float)):
                    points_against_pg = float(pa) / float(games_played)
                if points_for_pg is not None and points_against_pg is not None:
                    net_points_pg = points_for_pg - points_against_pg

            turnover_margin_val = stats_entry.get("TurnoverMargin")
            if isinstance(turnover_margin_val, (int, float)):
                turnover_margin = float(turnover_margin_val)

            # Simple power index derived from scoring margin and win percentage.
            win_pct = None
            if isinstance(stats_entry.get("Wins"), (int, float)) and isinstance(stats_entry.get("Games"), (int, float)):
                games = float(stats_entry.get("Games") or 0)
                if games:
                    win_pct = float(stats_entry.get("Wins") or 0) / games
            margin_component = net_points_pg if net_points_pg is not None else 0.0
            win_component = (win_pct * 10.0) if win_pct is not None else 0.0
            power_index = margin_component + win_component

        return SportsDataTeamInsight(
            name=str(team_name),
            alias=str(alias) if alias else None,
            record=record,
            streak=streak_text,
            trend=trend,
            points_for_per_game=points_for_pg,
            points_against_per_game=points_against_pg,
            net_points_per_game=net_points_pg,
            turnover_margin=turnover_margin,
            power_index=power_index,
        )

    # ------------------------------------------------------------------
    def find_game_insight(
        self,
        game_date: date,
        home_team: str,
        away_team: str,
    ) -> Optional[SportsDataGameInsight]:
        games = self.get_scores_by_date(game_date)
        matched = self.match_game(games, home_team, away_team)
        if not matched:
            return None

        season = matched.get("Season")
        season_type = matched.get("SeasonType") or self.DEFAULT_SEASON_TYPE
        stats_map = self._fetch_team_stats(season, season_type)
        return self.build_game_insight(matched, team_stats=stats_map)


__all__ = [
    "SportsDataNFLClient",
    "SportsDataGameInsight",
    "SportsDataTeamInsight",
]
