"""Lightweight clients for API-Sports live sports data."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pytz
import requests


@dataclass
class TeamSummary:
    """Minimal statistics for a team returned by API-Sports."""

    id: Optional[int]
    name: str
    record: Optional[str]
    form: Optional[str]
    average_points_for: Optional[float]
    average_points_against: Optional[float]
    trend: Optional[str]


@dataclass
class GameSummary:
    """Relevant details for a game fetched from API-Sports."""

    id: Optional[int]
    league: Optional[str]
    season: Optional[str]
    status: Optional[str]
    stage: Optional[str]
    kickoff_local: Optional[str]
    venue: Optional[str]
    home: TeamSummary
    away: TeamSummary
    sport_key: Optional[str] = None
    sport_name: Optional[str] = None
    scoring_metric: Optional[str] = None


class _APISportsBaseClient:
    """Shared helper for sport-specific API-Sports clients."""

    BASE_URL: str = ""
    DEFAULT_LEAGUE_ID: int = 0
    SECRET_ENV_PRIORITY: Sequence[str] = ()
    SPORT_KEY: Optional[str] = None
    SPORT_NAME: Optional[str] = None
    STAT_CATEGORY_OFFENSE: str = "points"
    STAT_CATEGORY_DEFENSE: str = "points"
    SCORING_METRIC_LABEL: str = "points"
    SEASON_CUTOFF_MONTH: int = 1
    SEASON_FORMAT: str = "single"  # "single" -> "2024", "split" -> "2023-2024"

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        key_source: Optional[str] = None,
    ) -> None:
        self.key_source: Optional[str] = key_source
        resolved_key = api_key or ""
        if resolved_key:
            self.key_source = key_source or "runtime"
        else:
            for env_name in self.SECRET_ENV_PRIORITY:
                candidate = os.environ.get(env_name)
                if candidate:
                    resolved_key = candidate
                    self.key_source = f"env:{env_name}"
                    break

        self.api_key = resolved_key or ""
        self.session = session or requests.Session()
        self.timeout = 10
        self._games_cache: Dict[Tuple[str, str, int, str], List[Dict]] = {}
        self._team_cache: Dict[Tuple[int, str, int], Dict] = {}
        self.last_error: Optional[str] = None

        if self.api_key:
            self.session.headers.update({"x-apisports-key": self.api_key})

    # ------------------------------------------------------------------
    # General helpers
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def update_api_key(self, api_key: Optional[str], source: Optional[str] = None) -> None:
        self.api_key = api_key or ""
        if self.api_key:
            self.key_source = source or "runtime"
        else:
            self.key_source = None
        if self.api_key:
            self.session.headers.update({"x-apisports-key": self.api_key})
        else:
            self.session.headers.pop("x-apisports-key", None)
        self._games_cache.clear()
        self._team_cache.clear()
        self.last_error = None

    def key_origin(self) -> Optional[str]:
        """Return a short description of where the API key came from."""

        return self.key_source

    @classmethod
    def _season_start_year(cls, target: Optional[date] = None) -> int:
        target = target or datetime.utcnow().date()
        if target.month >= cls.SEASON_CUTOFF_MONTH:
            return target.year
        return target.year - 1

    @classmethod
    def season_candidates_for_date(cls, target: Optional[date] = None) -> List[str]:
        """Return possible season identifiers for the target date."""

        start_year = cls._season_start_year(target)
        if cls.SEASON_FORMAT == "split":
            # API-Sports tends to use "2023-2024" for leagues that span years.
            return [
                f"{start_year}-{start_year + 1}",
                str(start_year),
                str(start_year + 1),
            ]
        return [str(start_year)]

    @classmethod
    def current_season_for_date(cls, target: Optional[date] = None) -> str:
        """Return the primary season label (first candidate) for the given date."""

        return cls.season_candidates_for_date(target)[0]

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
        league_id: Optional[int] = None,
        season: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch games for a given date and timezone."""

        if not self.is_configured():
            return []

        league_id = league_id or self.DEFAULT_LEAGUE_ID
        season_candidates = (
            [season]
            if season
            else self.season_candidates_for_date(target_date)
        )

        last_games: List[Dict] = []
        for season_label in season_candidates:
            cache_key = (target_date.isoformat(), timezone, league_id, season_label)
            if cache_key in self._games_cache:
                games = self._games_cache[cache_key]
            else:
                payload = self._request(
                    "/games",
                    {
                        "date": target_date.isoformat(),
                        "timezone": timezone,
                        "league": league_id,
                        "season": season_label,
                    },
                )
                games = (payload or {}).get("response", []) if payload else []
                self._games_cache[cache_key] = games

            last_games = games
            if games:
                return games

        return last_games

    def get_team_statistics(
        self,
        team_id: Optional[int],
        season: Optional[str],
        league_id: Optional[int] = None,
    ) -> Dict:
        """Fetch team statistics for the specified season."""

        if not self.is_configured() or not team_id:
            return {}

        league_id = league_id or self.DEFAULT_LEAGUE_ID
        season_str = str(season) if season is not None else None
        season_candidates = (
            [season_str]
            if season_str
            else self.season_candidates_for_date()
        )

        for season_label in season_candidates:
            if season_label is None:
                continue
            cache_key = (team_id, season_label, league_id)
            if cache_key in self._team_cache:
                stats = self._team_cache[cache_key]
            else:
                payload = self._request(
                    "/teams/statistics",
                    {
                        "team": team_id,
                        "league": league_id,
                        "season": season_label,
                    },
                )
                stats = (payload or {}).get("response", {}) if payload else {}
                self._team_cache[cache_key] = stats
            if stats:
                return stats

        return {}

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
        draws = (games.get("draws") or {}).get("total")
        if wins is None or losses is None:
            return None
        if draws is not None and draws > 0:
            return f"{wins}-{losses}-{draws}"
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

    def _average_stat(self, stats: Dict, direction: str = "for") -> Optional[float]:
        metrics = (stats or {}).get(self.STAT_CATEGORY_OFFENSE if direction == "for" else self.STAT_CATEGORY_DEFENSE) or {}
        bucket = metrics.get(direction) or {}
        avg = bucket.get("average") or {}
        for key in ("total", "points", "value"):
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

        raw_season = season or game.get("season") or self.current_season_for_date()
        if isinstance(raw_season, (int, float)):
            season_label = str(int(raw_season))
        else:
            season_label = str(raw_season) if raw_season is not None else None

        season = season_label or self.current_season_for_date()
        teams = game.get("teams") or {}
        status = (game.get("status") or {}).get("long") or (game.get("status") or {}).get("short")

        stage = game.get("stage")
        if not stage:
            week = game.get("week")
            if isinstance(week, dict):
                stage = week.get("name") or week.get("number")
            elif isinstance(week, str):
                stage = week

        venue_info = game.get("game") or {}
        venue_raw = venue_info.get("venue") or (game.get("venue") or {}).get("name") or (game.get("arena") or {}).get("name")
        if isinstance(venue_raw, dict):
            venue = venue_raw.get("name") or venue_raw.get("city")
        else:
            venue = venue_raw

        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.timezone("UTC")
        tip_iso = venue_info.get("date") or (game.get("date") or {}).get("start") or (game.get("date") or {}).get("utc")
        if tip_iso:
            try:
                kickoff_local = (
                    datetime.fromisoformat(tip_iso.replace("Z", "+00:00"))
                    .astimezone(tz)
                    .strftime("%Y-%m-%d %I:%M %p %Z")
                )
            except ValueError:
                kickoff_local = None
        else:
            kickoff_local = None

        home_team = teams.get("home") or {}
        away_team = teams.get("away") or {}

        home_stats = self.get_team_statistics(home_team.get("id"), season)
        away_stats = self.get_team_statistics(away_team.get("id"), season)

        home_summary = TeamSummary(
            id=home_team.get("id"),
            name=home_team.get("name", ""),
            record=self._format_record(home_stats),
            form=home_stats.get("form"),
            average_points_for=self._average_stat(home_stats, "for"),
            average_points_against=self._average_stat(home_stats, "against"),
            trend=self._determine_trend(home_stats.get("form")),
        )
        away_summary = TeamSummary(
            id=away_team.get("id"),
            name=away_team.get("name", ""),
            record=self._format_record(away_stats),
            form=away_stats.get("form"),
            average_points_for=self._average_stat(away_stats, "for"),
            average_points_against=self._average_stat(away_stats, "against"),
            trend=self._determine_trend(away_stats.get("form")),
        )

        return GameSummary(
            id=game.get("id"),
            league=(game.get("league") or {}).get("name"),
            season=season,
            status=status,
            stage=stage,
            kickoff_local=kickoff_local,
            venue=venue,
            home=home_summary,
            away=away_summary,
            sport_key=self.SPORT_KEY,
            sport_name=self.SPORT_NAME,
            scoring_metric=self.SCORING_METRIC_LABEL,
        )


class APISportsFootballClient(_APISportsBaseClient):
    """Small wrapper around the API-Sports American football endpoints."""

    BASE_URL = "https://v1.american-football.api-sports.io"
    DEFAULT_LEAGUE_ID = 1  # NFL
    SECRET_ENV_PRIORITY = ("NFL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY")
    SPORT_KEY = "americanfootball_nfl"
    SPORT_NAME = "NFL"
    STAT_CATEGORY_OFFENSE = "points"
    STAT_CATEGORY_DEFENSE = "points"
    SCORING_METRIC_LABEL = "points"
    SEASON_CUTOFF_MONTH = 3
    SEASON_FORMAT = "single"


class APISportsHockeyClient(_APISportsBaseClient):
    """Wrapper around the API-Sports ice hockey endpoints (NHL focus)."""

    BASE_URL = "https://v1.hockey.api-sports.io"
    DEFAULT_LEAGUE_ID = 57  # NHL
    SECRET_ENV_PRIORITY = ("NHL_APISPORTS_API_KEY", "APISPORTS_API_KEY", "API_SPORTS_KEY")
    SPORT_KEY = "icehockey_nhl"
    SPORT_NAME = "NHL"
    STAT_CATEGORY_OFFENSE = "goals"
    STAT_CATEGORY_DEFENSE = "goals"
    SCORING_METRIC_LABEL = "goals"
    SEASON_CUTOFF_MONTH = 7
    SEASON_FORMAT = "split"
