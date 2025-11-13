"""Lightweight client for API-Sports basketball data.

The Streamlit app uses this module to pull live games and team statistics so
odds analysis can be enriched with real-world league context.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import pytz
import requests


@dataclass
class TeamSummary:
    """Minimal statistics for a basketball team."""

    id: Optional[int]
    name: str
    record: Optional[str]
    form: Optional[str]
    average_points: Optional[float]
    trend: Optional[str]


@dataclass
class GameSummary:
    """Relevant details for an NBA game fetched from API-Sports."""

    id: Optional[int]
    league: Optional[str]
    season: Optional[str]
    status: Optional[str]
    stage: Optional[str]
    tipoff_local: Optional[str]
    arena: Optional[str]
    home: TeamSummary
    away: TeamSummary


class APISportsBasketballClient:
    """Small wrapper around the API-Sports basketball endpoints."""

    BASE_URL = "https://v1.basketball.api-sports.io"
    NBA_LEAGUE_ID = 12

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key or os.environ.get("APISPORTS_API_KEY") or os.environ.get("API_SPORTS_KEY")
        self.session = session or requests.Session()
        self.timeout = 10
        self._games_cache: Dict[tuple, List[Dict]] = {}
        self._team_cache: Dict[tuple, Dict] = {}
        self.last_error: Optional[str] = None

        if self.api_key:
            self.session.headers.update({"x-apisports-key": self.api_key})

    # ------------------------------------------------------------------
    # General helpers
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def update_api_key(self, api_key: Optional[str]) -> None:
        self.api_key = api_key or ""
        if api_key:
            self.session.headers.update({"x-apisports-key": api_key})
        else:
            self.session.headers.pop("x-apisports-key", None)
        self._games_cache.clear()
        self._team_cache.clear()
        self.last_error = None

    @staticmethod
    def current_season_for_date(target: Optional[date] = None) -> str:
        """Return the NBA season string (e.g. "2023-2024") for a given date."""

        target = target or datetime.utcnow().date()
        if target.month >= 7:
            start_year = target.year
        else:
            start_year = target.year - 1
        return f"{start_year}-{start_year + 1}"

    # ------------------------------------------------------------------
    # Networking helpers
    def _request(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        if not self.is_configured():
            self.last_error = "Missing API-Sports key"
            return None

        url = f"{self.BASE_URL}{path}"
        try:
            resp = self.session.get(url, params=params or {}, timeout=self.timeout)
            if resp.status_code == 401:
                self.last_error = "Invalid API-Sports key"
                return None
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("errors"):
                self.last_error = "; ".join(payload.get("errors", {}).values()) or "Unknown API-Sports error"
                return None
            self.last_error = None
            return payload
        except requests.exceptions.Timeout:
            self.last_error = "API-Sports timeout"
        except requests.RequestException as exc:
            self.last_error = f"API-Sports request failed: {exc}"
        except ValueError:
            self.last_error = "Invalid JSON from API-Sports"
        return None

    def get_games_by_date(
        self,
        target_date: date,
        timezone: str = "America/New_York",
        league_id: int = NBA_LEAGUE_ID,
        season: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch NBA games for a given date and timezone."""

        if not self.is_configured():
            return []

        season = season or self.current_season_for_date(target_date)
        cache_key = (target_date.isoformat(), timezone, league_id, season)
        if cache_key in self._games_cache:
            return self._games_cache[cache_key]

        payload = self._request(
            "/games",
            {
                "date": target_date.isoformat(),
                "timezone": timezone,
                "league": league_id,
                "season": season,
            },
        )
        games = (payload or {}).get("response", []) if payload else []
        self._games_cache[cache_key] = games
        return games

    def get_team_statistics(
        self,
        team_id: Optional[int],
        season: str,
        league_id: int = NBA_LEAGUE_ID,
    ) -> Dict:
        """Fetch team statistics for the specified season."""

        if not self.is_configured() or not team_id:
            return {}

        cache_key = (team_id, season, league_id)
        if cache_key in self._team_cache:
            return self._team_cache[cache_key]

        payload = self._request(
            "/teams/statistics",
            {
                "team": team_id,
                "league": league_id,
                "season": season,
            },
        )
        stats = (payload or {}).get("response", {}) if payload else {}
        self._team_cache[cache_key] = stats
        return stats

    # ------------------------------------------------------------------
    # Formatting helpers
    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"[^A-Z]", "", (name or "").upper())

    def match_game(self, games: List[Dict], home: str, away: str) -> Optional[Dict]:
        """Match an odds snapshot game to an API-Sports game response."""

        if not games:
            return None

        home_norm = self._normalize_name(home)
        away_norm = self._normalize_name(away)

        for game in games:
            teams = game.get("teams") or {}
            home_team = teams.get("home") or {}
            away_team = teams.get("away") or {}
            if (
                self._normalize_name(home_team.get("name", "")) == home_norm
                and self._normalize_name(away_team.get("name", "")) == away_norm
            ):
                return game
        return None

    @staticmethod
    def _format_record(stats: Dict) -> Optional[str]:
        games = (stats or {}).get("games") or {}
        wins = (games.get("wins") or {}).get("total")
        losses = (games.get("loses") or {}).get("total")
        if wins is None or losses is None:
            return None
        return f"{wins}-{losses}"

    @staticmethod
    def _determine_trend(form: Optional[str]) -> Optional[str]:
        if not form:
            return None
        wins = form.count("W")
        losses = form.count("L")
        if wins - losses >= 2:
            return "hot"
        if losses - wins >= 2:
            return "cold"
        return "neutral"

    @staticmethod
    def _average_points(stats: Dict) -> Optional[float]:
        points = (stats or {}).get("points") or {}
        avg = points.get("average") or {}
        # Some responses use "total", others "points"
        for key in ("total", "points"):
            value = avg.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def build_game_summary(
        self,
        game: Dict,
        tz_name: str = "America/New_York",
        season: Optional[str] = None,
    ) -> GameSummary:
        """Convert a raw API response into a structured summary."""

        season = season or game.get("season") or self.current_season_for_date()
        teams = game.get("teams") or {}
        status = (game.get("status") or {}).get("long") or (game.get("status") or {}).get("short")
        stage = game.get("stage")
        arena = (game.get("arena") or {}).get("name")

        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.timezone("UTC")
        tip_iso = (game.get("date") or {}).get("start") or (game.get("date") or {}).get("utc")
        if tip_iso:
            try:
                tipoff_local = (
                    datetime.fromisoformat(tip_iso.replace("Z", "+00:00"))
                    .astimezone(tz)
                    .strftime("%Y-%m-%d %I:%M %p %Z")
                )
            except ValueError:
                tipoff_local = None
        else:
            tipoff_local = None

        home_team = teams.get("home") or {}
        away_team = teams.get("away") or {}

        home_stats = self.get_team_statistics(home_team.get("id"), season)
        away_stats = self.get_team_statistics(away_team.get("id"), season)

        home_summary = TeamSummary(
            id=home_team.get("id"),
            name=home_team.get("name", ""),
            record=self._format_record(home_stats),
            form=home_stats.get("form"),
            average_points=self._average_points(home_stats),
            trend=self._determine_trend(home_stats.get("form")),
        )
        away_summary = TeamSummary(
            id=away_team.get("id"),
            name=away_team.get("name", ""),
            record=self._format_record(away_stats),
            form=away_stats.get("form"),
            average_points=self._average_points(away_stats),
            trend=self._determine_trend(away_stats.get("form")),
        )

        return GameSummary(
            id=game.get("id"),
            league=(game.get("league") or {}).get("name"),
            season=season,
            status=status,
            stage=stage,
            tipoff_local=tipoff_local,
            arena=arena,
            home=home_summary,
            away=away_summary,
        )
