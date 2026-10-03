"""Controlled provider diagnostics; these facts never supply wager authority."""
from __future__ import annotations

import requests


SPORT_KEYS = frozenset({
    "basketball_ncaab", "basketball_nba", "basketball_wnba", "icehockey_nhl",
    "americanfootball_nfl", "americanfootball_nfl_preseason", "americanfootball_ncaaf",
    "baseball_mlb", "baseball_mlb_preseason",
})
OUTCOMES = frozenset({
    "SUCCESS", "SUCCESS_EMPTY", "AUTHENTICATION_FAILURE", "RATE_LIMIT_EXHAUSTED",
    "HTTP_FAILURE", "TIMEOUT", "TRANSPORT_FAILURE", "PROVIDER_FAILURE",
    "INVALID_RESPONSE", "NOT_CONFIGURED", "NOT_RECORDED",
})
PROCESSING = frozenset({"NOT_RUN", "EMPTY", "FILTERED_EMPTY", "SUCCESS", "FAILED"})
FALLBACKS = frozenset({"nfl_novig", "college_novig", "ncaaf_fcs"})


def verified_status(exc) -> int | None:
    """Use only a typed response status, never parse exception/body/URL text."""
    try:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    except Exception:
        return None
    return value if type(value) is int and 100 <= value <= 599 else None


def failure(exc) -> dict:
    status = verified_status(exc)
    if isinstance(exc, requests.exceptions.HTTPError):
        category = ("AUTHENTICATION_FAILURE" if status in (401, 403) else
                    "RATE_LIMIT_EXHAUSTED" if status == 429 else "HTTP_FAILURE")
    elif isinstance(exc, requests.exceptions.Timeout):
        category = "TIMEOUT"
    elif isinstance(exc, requests.exceptions.RequestException):
        category = "TRANSPORT_FAILURE"
    else:
        category = "PROVIDER_FAILURE"
    return {"outcome": category, "http_status": status}


def outcome(category: str, received_games: int = 0, http_status: int | None = None) -> dict:
    return {"outcome": category, "http_status": http_status,
            "received_games": received_games, "processing": "NOT_RUN", "processing_error": None,
            "fallback_errors": {}, "fallback_outcomes": {}}


def _count(value) -> int:
    return value if type(value) is int and value >= 0 else 0


def _choice(value, allowed, default):
    return value if isinstance(value, str) and value in allowed else default


def sanitized_outcome(raw) -> dict:
    """Project a helper receipt without retaining exceptions, bodies or URLs."""
    raw = raw if isinstance(raw, dict) else {}
    status = raw.get("http_status")
    return {"outcome": _choice(raw.get("outcome"), OUTCOMES, "NOT_RECORDED"),
            "http_status": status if type(status) is int and 100 <= status <= 599 else None}


class ProviderGames(list):
    """A normal game list with additive, sanitized fetch-outcome metadata."""
    def __init__(self, games, detail):
        super().__init__(games)
        self.provider_outcome = sanitized_outcome(detail)


def sanitized_health(raw) -> dict:
    """Re-project even stored diagnostics; arbitrary provider text is never rendered."""
    if not isinstance(raw, dict) or not isinstance(raw.get("sports"), dict):
        return {}
    sports = {}
    for sport, value in raw["sports"].items():
        if sport not in SPORT_KEYS or not isinstance(value, dict):
            continue
        status = value.get("http_status")
        item = outcome(_choice(value.get("outcome"), OUTCOMES, "NOT_RECORDED"),
                       _count(value.get("received_games")),
                       status if type(status) is int and 100 <= status <= 599 else None)
        item["processing"] = _choice(value.get("processing"), PROCESSING, "NOT_RUN")
        error = value.get("processing_error")
        if isinstance(error, dict):
            code = error.get("http_status")
            item["processing_error"] = {
                "outcome": _choice(error.get("outcome"), OUTCOMES, "NOT_RECORDED"),
                "http_status": code if type(code) is int and 100 <= code <= 599 else None,
            }
        errors = value.get("fallback_errors")
        if isinstance(errors, dict):
            for name, error in errors.items():
                if name in FALLBACKS and isinstance(error, dict):
                    code = error.get("http_status")
                    item["fallback_errors"][name] = {
                        "outcome": _choice(error.get("outcome"), OUTCOMES, "NOT_RECORDED"),
                        "http_status": code if type(code) is int and 100 <= code <= 599 else None,
                    }
        outcomes = value.get("fallback_outcomes")
        if isinstance(outcomes, dict):
            for name, detail in outcomes.items():
                if name in FALLBACKS and isinstance(detail, dict):
                    projected = sanitized_outcome(detail)
                    item["fallback_outcomes"][name] = projected
                    if projected["outcome"] not in {"SUCCESS", "SUCCESS_EMPTY", "NOT_RECORDED"}:
                        item["fallback_errors"][name] = projected
        sports[sport] = item
    good = sum(v["outcome"] in {"SUCCESS", "SUCCESS_EMPTY"} and
               v["processing"] != "FAILED" and not v["fallback_errors"] for v in sports.values())
    bad = len(sports) - good
    status = ("NOT_REQUESTED" if not sports else "PARTIAL_FAILURE" if good and bad else
              "FAILED" if bad else "SUCCESS_EMPTY" if all(v["outcome"] == "SUCCESS_EMPTY" for v in sports.values())
              else "SUCCESS")
    return {"schema_version": 1, "status": status, "sports": sports,
            "returned_games": _count(raw.get("returned_games"))}


def health_report(sports: dict, returned_games: int) -> dict:
    return sanitized_health({"sports": sports, "returned_games": returned_games})
