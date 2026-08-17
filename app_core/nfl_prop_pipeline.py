"""Research-first NFL player-prop pipeline.

NFL props intentionally do not inherit MLB calibration or production eligibility.
The first rollout covers repeatable volume markets, estimates a conservative recent-
form distribution, exports every prediction for grading, and assigns zero stake until
each league/market/direction has enough settled out-of-sample evidence.
"""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Any, Callable

import pandas as pd

from app_core.prop_odds_ingest import (
    NFL_PLAYER_PROP_MARKET_KEYS,
    fetch_nfl_player_props,
)


NFL_SPORT_KEYS = ("americanfootball_nfl_preseason", "americanfootball_nfl")
NFL_PROP_LOOKBACK_GAMES = 8
NFL_PROP_MIN_FORM_GAMES = 3
NFL_PROP_MODEL_WEIGHT = 0.35
NFL_PROP_UNCERTAINTY_Z = 0.35

NFL_PROP_SPECS = {
    "player_pass_yds": {
        "stat": "passing_yards",
        "label": "Pass Yards",
        "sigma_floor": 35.0,
    },
    "player_rush_yds": {
        "stat": "rushing_yards",
        "label": "Rush Yards",
        "sigma_floor": 12.0,
    },
    "player_reception_yds": {
        "stat": "receiving_yards",
        "label": "Receiving Yards",
        "sigma_floor": 12.0,
    },
    "player_receptions": {
        "stat": "receptions",
        "label": "Receptions",
        "sigma_floor": 1.2,
    },
}


def _norm_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _american_decimal(value: object) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def _american_implied(value: object) -> float | None:
    decimal = _american_decimal(value)
    return (1.0 / decimal) if decimal and decimal > 1.0 else None


def _devig_pair(over_odds: object, under_odds: object) -> tuple[float, float] | None:
    over = _american_implied(over_odds)
    under = _american_implied(under_odds)
    if over is None or under is None or over + under <= 0:
        return None
    total = over + under
    return over / total, under / total


def _normal_over_probability(mean: float, sigma: float, line: float) -> float:
    if sigma <= 0:
        return 0.5
    return float(1.0 - NormalDist(mu=float(mean), sigma=float(sigma)).cdf(float(line)))


def _completed_regular_week(schedules: pd.DataFrame, date: str) -> int:
    if schedules is None or schedules.empty or "week" not in schedules.columns:
        return 0
    date_col = next(
        (name for name in ("gameday", "game_date", "date") if name in schedules.columns),
        None,
    )
    if date_col is None:
        return 0
    game_dates = pd.to_datetime(schedules[date_col], errors="coerce", utc=True)
    cutoff = pd.to_datetime(date, errors="coerce", utc=True)
    if pd.isna(cutoff):
        return 0
    completed = schedules.loc[game_dates.lt(cutoff), "week"]
    completed = pd.to_numeric(completed, errors="coerce").dropna()
    return int(completed.max()) if not completed.empty else 0


def load_nfl_player_forms(
    season: int,
    date: str,
    *,
    weekly_fetch: Callable | None = None,
    schedule_fetch: Callable | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build leak-safe recent-form profiles from completed NFL weekly data.

    Prior-season games provide a veteran baseline during preseason. Current-season
    rows are included only through the latest week whose scheduled date precedes
    the requested slate date.
    """
    if weekly_fetch is None or schedule_fetch is None:
        try:
            import nfl_data_py as nfl
        except ImportError:
            return {}
        weekly_fetch = weekly_fetch or nfl.import_weekly_data
        schedule_fetch = schedule_fetch or nfl.import_schedules
    try:
        weekly = weekly_fetch([int(season) - 1, int(season)])
        schedules = schedule_fetch([int(season)])
    except Exception:
        return {}
    if not isinstance(weekly, pd.DataFrame) or weekly.empty:
        return {}

    current_week = _completed_regular_week(schedules, date)
    weekly = weekly.copy()
    season_values = pd.to_numeric(
        weekly.get("season", pd.Series(int(season) - 1, index=weekly.index)),
        errors="coerce",
    )
    week_values = pd.to_numeric(
        weekly.get("week", pd.Series(0, index=weekly.index)), errors="coerce"
    ).fillna(0)
    allowed = season_values.lt(int(season)) | (
        season_values.eq(int(season)) & week_values.le(current_week)
    )
    weekly = weekly[allowed].copy()
    if weekly.empty:
        return {}

    name_col = next(
        (
            name
            for name in ("player_display_name", "player_name", "display_name")
            if name in weekly.columns
        ),
        None,
    )
    if name_col is None:
        return {}
    weekly["_player_key"] = weekly[name_col].map(_norm_name)
    weekly["_season"] = season_values.loc[weekly.index].fillna(0)
    weekly["_week"] = week_values.loc[weekly.index]
    weekly = weekly[weekly["_player_key"].ne("")]

    forms: dict[str, dict[str, dict[str, float]]] = {}
    for player_key, player_rows in weekly.groupby("_player_key", sort=False):
        player_rows = player_rows.sort_values(["_season", "_week"])
        profile: dict[str, dict[str, float]] = {}
        for spec in NFL_PROP_SPECS.values():
            stat = spec["stat"]
            if stat not in player_rows.columns:
                continue
            values = pd.to_numeric(player_rows[stat], errors="coerce").dropna()
            if len(values) < NFL_PROP_MIN_FORM_GAMES:
                continue
            recent = values.tail(NFL_PROP_LOOKBACK_GAMES)
            weights = pd.Series(range(1, len(recent) + 1), index=recent.index, dtype=float)
            recent_mean = float((recent * weights).sum() / weights.sum())
            season_mean = float(values.tail(17).mean())
            expected = 0.70 * recent_mean + 0.30 * season_mean
            sigma = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
            sigma = max(float(spec["sigma_floor"]), sigma)
            profile[stat] = {
                "expected": expected,
                "sigma": sigma,
                "games": float(len(recent)),
            }
        if profile:
            forms[player_key] = profile
    return forms


def score_nfl_prop(
    prop: dict,
    forms: dict[str, dict[str, dict[str, float]]],
) -> dict | None:
    """Score one NFL over/under quote into an explicitly unfunded research row."""
    market_key = str(prop.get("market_key") or "")
    spec = NFL_PROP_SPECS.get(market_key)
    player = str(prop.get("player") or "").strip()
    if spec is None or not player:
        return None
    form = forms.get(_norm_name(player), {}).get(spec["stat"])
    try:
        line = float(prop["line"])
        over_odds = float(prop["over_odds"])
        under_odds = float(prop["under_odds"])
    except (KeyError, TypeError, ValueError):
        return None
    market_pair = _devig_pair(over_odds, under_odds)
    if market_pair is None:
        return None
    market_over, market_under = market_pair
    has_form = bool(form)
    model_over = (
        min(
            0.97,
            max(
                0.03,
                _normal_over_probability(form["expected"], form["sigma"], line),
            ),
        )
        if has_form
        else market_over
    )
    raw_probabilities = {"over": model_over, "under": 1.0 - model_over}
    market_probabilities = {"over": market_over, "under": market_under}
    raw_edges = {
        side: raw_probabilities[side] - market_probabilities[side]
        for side in ("over", "under")
    }
    side = max(
        ("over", "under"),
        key=(
            (lambda value: raw_edges[value])
            if has_form
            else (lambda value: market_probabilities[value])
        ),
    )
    raw_probability = raw_probabilities[side]
    market_probability = market_probabilities[side]
    # With no settled NFL calibration, use the model as a rank signal only and
    # let the market carry most of the decision probability.
    blended = market_probability + NFL_PROP_MODEL_WEIGHT * (
        raw_probability - market_probability
    )
    uncertainty = (
        NFL_PROP_UNCERTAINTY_Z
        * sqrt(
            max(0.01, blended * (1.0 - blended))
            / max(1.0, form["games"])
        )
        if has_form
        else 0.025
    )
    conservative = min(raw_probability, max(0.05, blended - uncertainty))
    odds = over_odds if side == "over" else under_odds
    decimal = _american_decimal(odds)
    if decimal is None:
        return None
    matchup = f"{prop.get('away_team', '')} @ {prop.get('home_team', '')}".strip(" @")
    reason = (
        "Research only: NFL player-prop model is uncalibrated; collect settled "
        "league/market/direction results before production eligibility"
        if has_form
        else "Research only: no completed NFL player form was available; market-baseline "
        "row retained for grading and future calibration"
    )
    return {
        "league": "NFL",
        "game_date": str(prop.get("slate_date") or prop.get("commence_time") or "")[:10],
        "event_id": prop.get("event_id"),
        "player": player,
        "participant_type": "nfl_player",
        "matchup": matchup,
        "market_type": f"{market_key}_{side}",
        "best_pick": f"{player} {side.capitalize()} {line:g} {spec['label']}",
        "line": line,
        "expected_stat": spec["stat"],
        "expected_count": round(float(form["expected"]), 3) if has_form else line,
        "projection_sigma": round(float(form["sigma"]), 3) if has_form else pd.NA,
        "FormSampleSize": int(form["games"]) if has_form else 0,
        "RawWinProbability": round(float(raw_probability), 4),
        "MarketProbability": round(float(market_probability), 4),
        "CalibratedProbability": round(float(blended), 4),
        "ConservativeWinProbability": round(float(conservative), 4),
        "WinProbability": round(float(conservative), 4),
        "CalibrationSource": (
            "nfl_research_only_uncalibrated"
            if has_form else "nfl_market_only_no_form"
        ),
        "CalibrationSampleSize": 0,
        "CalibrationProfileSampleSize": 0,
        "DirectionalCalibrationSampleSize": 0,
        "expected_value": round(float(conservative * decimal - 1.0), 4),
        "edge": round(float(conservative - market_probability), 4),
        "raw_model_edge": round(float(raw_edges[side]), 4),
        "odds_american": int(odds),
        "book": prop.get("book"),
        "Pick_Status": "Research",
        "Stake_Status": "Research / No Stake",
        "Kelly_Bet_Size": 0.0,
        "Market_Probation": True,
        "Prop_Tier": "Research",
        "production_eligible": False,
        "production_gate_reason": reason,
        "Status_Reason": reason,
        "Prop_Precision_Shortlist": False,
        "Prop_Precision_Rank": pd.NA,
        "Wager_Instruction": "DO NOT BET - NFL PROP RESEARCH / $0",
    }


def build_nfl_prop_card(
    odds_client: Any,
    date: str,
    season: int,
    *,
    sport_keys: tuple[str, ...] = NFL_SPORT_KEYS,
    diagnostics: dict | None = None,
    list_events: Callable | None = None,
    props_fetch: Callable = fetch_nfl_player_props,
    form_loader: Callable = load_nfl_player_forms,
) -> pd.DataFrame:
    """Fetch, score, and rank NFL props without assigning a production stake."""
    if list_events is None:
        def list_events(client, sport_key, slate_date):  # noqa: ANN001
            return client.get_odds(sport_key, date=slate_date) or []
    try:
        forms = form_loader(int(season), date)
    except Exception:
        forms = {}
    if diagnostics is not None:
        diagnostics["nfl_prop_form_player_count"] = int(len(forms))
    props: list[dict] = []
    event_count = 0
    fetch_errors = 0
    for sport_key in sport_keys:
        try:
            events = list_events(odds_client, sport_key, date) or []
        except Exception:
            fetch_errors += 1
            continue
        for event in events:
            if not isinstance(event, dict) or not event.get("id"):
                continue
            event_count += 1
            try:
                rows = props_fetch(odds_client, sport_key, event["id"]) or []
            except Exception:
                fetch_errors += 1
                continue
            for row in rows:
                merged = dict(row)
                merged.setdefault("event_id", event.get("id"))
                merged.setdefault("commence_time", event.get("commence_time"))
                merged.setdefault("home_team", event.get("home_team"))
                merged.setdefault("away_team", event.get("away_team"))
                merged["sport_key"] = sport_key
                merged["slate_date"] = str(date)[:10]
                props.append(merged)
    if diagnostics is not None:
        diagnostics["nfl_prop_event_count"] = event_count
        diagnostics["nfl_prop_event_fetch_errors"] = fetch_errors
        diagnostics["nfl_prop_raw_count"] = len(props)
    if not props:
        if diagnostics is not None:
            diagnostics["nfl_prop_feed_status"] = (
                "event_fetch_failed" if fetch_errors else "no_prop_markets"
            )
        return pd.DataFrame()

    rows = [score_nfl_prop(prop, forms) for prop in props]
    card = pd.DataFrame([row for row in rows if row is not None])
    if card.empty:
        if diagnostics is not None:
            diagnostics["nfl_prop_feed_status"] = "no_matched_player_form"
        return card
    card = card.drop_duplicates(
        ["event_id", "player", "market_type", "line"], keep="first"
    )
    card = card.sort_values(
        ["WinProbability", "raw_model_edge", "expected_value"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    card["NFL_Research_Rank"] = range(1, len(card) + 1)
    if diagnostics is not None:
        diagnostics["nfl_prop_feed_status"] = "ready"
        diagnostics["nfl_prop_scored_count"] = int(len(card))
    return card


def _number(value: object) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _category_value(labels: list[str], stats: list[object], label: str) -> float | None:
    normalized = [str(item).strip().upper() for item in labels]
    try:
        return _number(stats[normalized.index(label.upper())])
    except (ValueError, IndexError):
        return None


def fetch_nfl_actuals(
    game_date: str,
    *,
    http_get: Callable | None = None,
) -> dict[str, dict[str, float]]:
    """Fetch NFL player box-score stats from ESPN for automatic prop grading."""
    if http_get is None:
        import requests

        http_get = requests.get
    compact_date = str(game_date).replace("-", "")[:8]
    scoreboard_url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    )
    try:
        response = http_get(
            scoreboard_url, params={"dates": compact_date, "limit": 100}, timeout=15
        )
        response.raise_for_status()
        events = response.json().get("events", [])
    except Exception:
        return {}

    actuals: dict[str, dict[str, float]] = {}
    summary_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
    for event in events:
        event_id = event.get("id") if isinstance(event, dict) else None
        if not event_id:
            continue
        try:
            response = http_get(summary_url, params={"event": event_id}, timeout=15)
            response.raise_for_status()
            teams = response.json().get("boxscore", {}).get("players", [])
        except Exception:
            continue
        for team in teams:
            for category in team.get("statistics", []) if isinstance(team, dict) else []:
                category_name = str(category.get("name") or "").strip().lower()
                labels = list(category.get("labels") or [])
                for athlete_row in category.get("athletes", []) or []:
                    athlete = athlete_row.get("athlete", {})
                    name = athlete.get("displayName") or athlete.get("shortName")
                    key = _norm_name(name)
                    if not key:
                        continue
                    stats = list(athlete_row.get("stats") or [])
                    player = actuals.setdefault(key, {})
                    if category_name == "passing":
                        yards = _category_value(labels, stats, "YDS")
                        completions_attempts = None
                        try:
                            completions_attempts = stats[
                                [str(x).strip().upper() for x in labels].index("C/ATT")
                            ]
                        except (ValueError, IndexError):
                            pass
                        if yards is not None:
                            player["passing_yards"] = yards
                        if completions_attempts and "/" in str(completions_attempts):
                            completed, attempted = str(completions_attempts).split("/", 1)
                            if _number(completed) is not None:
                                player["completions"] = float(_number(completed))
                            if _number(attempted) is not None:
                                player["pass_attempts"] = float(_number(attempted))
                    elif category_name == "rushing":
                        attempts = _category_value(labels, stats, "CAR")
                        yards = _category_value(labels, stats, "YDS")
                        if attempts is not None:
                            player["rush_attempts"] = attempts
                        if yards is not None:
                            player["rushing_yards"] = yards
                    elif category_name == "receiving":
                        receptions = _category_value(labels, stats, "REC")
                        yards = _category_value(labels, stats, "YDS")
                        if receptions is not None:
                            player["receptions"] = receptions
                        if yards is not None:
                            player["receiving_yards"] = yards
    return actuals
