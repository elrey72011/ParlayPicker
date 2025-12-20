"""Lightweight sentiment pipeline using NewsAPI with keyword lexicon scoring."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

_POSITIVE = {
    "win",
    "wins",
    "victory",
    "dominant",
    "elite",
    "excellent",
    "great",
    "strong",
    "hot",
    "streak",
    "improve",
    "improving",
    "surge",
    "surging",
    "secure",
    "clinched",
    "confident",
    "healthy",
}

_NEGATIVE = {
    "loss",
    "lose",
    "loses",
    "injury",
    "injured",
    "cold",
    "slump",
    "struggle",
    "struggling",
    "issue",
    "issues",
    "problem",
    "problems",
    "out",
    "doubt",
    "doubtful",
    "illness",
    "sick",
    "ankle",
    "knee",
    "hamstring",
}


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        v = float(val)
    except Exception:
        return 0.0
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def score_text_simple(text: str) -> float:
    """Score text using a tiny lexicon; normalized per token and clamped to [-1, 1]."""
    if not text:
        return 0.0
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0
    raw = 0
    for tok in tokens:
        if tok in _POSITIVE:
            raw += 1
        elif tok in _NEGATIVE:
            raw -= 1
    norm = raw / float(len(tokens))
    return _clamp(norm)


def league_label(league: str) -> str:
    mapping = {
        "NBA": "NBA basketball",
        "NFL": "NFL football",
        "NCAAF": "college football",
        "NCAAB": "college basketball",
        "MLB": "MLB baseball",
        "NHL": "NHL hockey",
        "WNBA": "WNBA basketball",
    }
    return mapping.get((league or "").upper(), league or "")


def _newsapi_query(team: str, league: str, league_query: Optional[str] = None) -> str:
    league_fragment = (league_query or league_label(league)).strip()
    return f'"{team}" {league_fragment}'.strip()


@st.cache_data(ttl=300)
def fetch_team_news(news_api_key: str, team: str, league: str, league_query: Optional[str] = None, *, max_retries: int = 2, retry_delay: float = 0.75) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch recent articles for a team; returns (articles, info) where info contains status/error."""
    if not news_api_key:
        return [], {
            "error": "missing_key",
            "status": None,
            "status_code": None,
            "league_query": league_query or league,
            "totalResults": None,
            "q": None,
            "rate_limited": False,
            "auth_error": False,
        }
    league_query = league_query or league_label(league)
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=3)
    url = "https://newsapi.org/v2/everything"
    q = _newsapi_query(team, league, league_query)
    params = {
        "q": q,
        "sortBy": "relevancy",
        "pageSize": 20,
        "language": "en",
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "apiKey": news_api_key,
    }
    attempts = 0
    last_error: Optional[str] = None
    while attempts <= max_retries:
        attempts += 1
        try:
            resp = requests.get(url, params=params, timeout=8)
            status = resp.status_code
            data: Dict[str, Any] = {}
            try:
                data = resp.json() if hasattr(resp, "json") else {}
            except Exception:
                data = {}
            articles = data.get("articles", []) if isinstance(data, dict) else []
            total_results = data.get("totalResults") if isinstance(data, dict) else None
            rate_limited = status == 429
            auth_error = status in {401, 403}
            if status != 200:
                error_key = "rate_limited" if rate_limited else ("bad_key" if auth_error else "http_error")
                if rate_limited and attempts <= max_retries and not articles:
                    time.sleep(retry_delay * attempts)
                    last_error = error_key
                    continue
                return articles, {
                    "error": error_key,
                    "status": status,
                    "status_code": status,
                    "league_query": league_query,
                    "totalResults": total_results,
                    "q": q,
                    "attempts": attempts,
                    "rate_limited": rate_limited,
                    "auth_error": auth_error,
                    "response_text_snippet": (resp.text or "")[:200] if hasattr(resp, "text") else None,
                }
            return articles, {
                "error": None,
                "status": status,
                "status_code": status,
                "league_query": league_query,
                "totalResults": total_results,
                "q": q,
                "attempts": attempts,
                "rate_limited": rate_limited,
                "auth_error": auth_error,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempts <= max_retries:
                time.sleep(retry_delay * attempts)
                continue
            return [], {
                "error": last_error,
                "status": None,
                "status_code": None,
                "league_query": league_query,
                "totalResults": None,
                "q": q,
                "attempts": attempts,
                "rate_limited": False,
                "auth_error": False,
            }

    return [], {
        "error": last_error or "unknown_error",
        "status": None,
        "status_code": None,
        "league_query": league_query,
        "totalResults": None,
        "q": q,
        "attempts": attempts,
        "rate_limited": False,
        "auth_error": False,
    }


def team_sentiment_from_articles(articles: List[Dict[str, Any]]) -> float:
    if not articles:
        return 0.0
    scores: List[float] = []
    for art in articles:
        title = art.get("title") or ""
        desc = art.get("description") or ""
        combined = f"{title}. {desc}".strip()
        scores.append(score_text_simple(combined))
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    return _clamp(avg)


def build_team_sentiment_map(
    news_api_key: str, games: List[Dict[str, Any]], league: str
) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    teams = set()
    for g in games or []:
        if g.get("home_team"):
            teams.add(str(g.get("home_team")))
        if g.get("away_team"):
            teams.add(str(g.get("away_team")))

    sentiment_map: Dict[str, Optional[float]] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {
        "total_teams": len(teams),
        "article_counts": {},
        "missing_teams": [],
        "articles_total": 0,
        "error_count": 0,
        "errors_sample": [],
        "query_label_used": (league or "").upper(),
        "status_counts": {},
        "sample_calls": [],
        "league_label_used": league_label(league),
        "rate_limited": False,
        "auth_error": False,
    }

    def _record_status(fetch_info: Dict[str, Any]) -> Optional[int]:
        status_val = fetch_info.get("status")
        if status_val is None:
            status_val = fetch_info.get("status_code")
        try:
            status_int = int(status_val)
        except Exception:
            return None
        debug["status_counts"][status_int] = debug["status_counts"].get(status_int, 0) + 1
        return status_int

    for team in sorted(teams):
        try:
            query_label = league_label(league)
            articles, info = fetch_team_news(news_api_key, team, league, query_label)
            debug.setdefault("fetch_info", {})[team] = info
            error_reason = (info or {}).get("error")
            status_int = _record_status(info or {})
            if len(debug["sample_calls"]) < 10:
                debug["sample_calls"].append(
                    {
                        "team": team,
                        "league": league,
                        "q": (info or {}).get("q"),
                        "status": status_int,
                        "totalResults": (info or {}).get("totalResults"),
                        "error": error_reason,
                    }
                )
            if error_reason:
                debug["error_count"] += 1
                if len(debug["errors_sample"]) < 5:
                    debug["errors_sample"].append(
                        {"team": team, "error": error_reason, "status_code": (info or {}).get("status_code")}
                    )
            debug["article_counts"][team] = len(articles)
            debug["articles_total"] += len(articles)
            if not articles:
                sentiment_map[team] = None
                meta_map[team] = {
                    "sentiment_valid": False,
                    "articles": 0,
                    "sentiment_source": "newsapi",
                    "error": error_reason,
                }
                debug["missing_teams"].append(team)
                continue
            score = team_sentiment_from_articles(articles)
            sentiment_map[team] = score
            meta_map[team] = {
                "sentiment_valid": True,
                "articles": len(articles),
                "sentiment_source": "newsapi",
                "error": error_reason,
                "rate_limited": bool((info or {}).get("rate_limited")),
                "auth_error": bool((info or {}).get("auth_error")),
            }
        except Exception as exc:  # pragma: no cover - defensive
            sentiment_map[team] = None
            meta_map[team] = {
                "sentiment_valid": False,
                "articles": 0,
                "sentiment_source": "error",
                "error": str(exc),
            }
            debug["error_count"] += 1
            if len(debug["errors_sample"]) < 5:
                debug["errors_sample"].append({"team": team, "error": str(exc)})
            continue

    if sentiment_map:
        sorted_scores = sorted(sentiment_map.items(), key=lambda kv: kv[1] if kv[1] is not None else -999)
        debug["bottom_5"] = [kv for kv in sorted_scores if kv[1] is not None][:5]
        debug["top_5"] = [kv for kv in sorted_scores if kv[1] is not None][-5:]
    else:
        debug["bottom_5"] = []
        debug["top_5"] = []
    debug["rate_limited"] = bool(debug["status_counts"].get(429))
    debug["auth_error"] = bool(debug["status_counts"].get(401) or debug["status_counts"].get(403))

    return sentiment_map, meta_map, debug
