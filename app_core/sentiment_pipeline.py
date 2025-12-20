"""Lightweight sentiment pipeline using NewsAPI with keyword lexicon scoring."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
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


@st.cache_data(ttl=1800)
def fetch_team_news(news_api_key: str, team: str, league: str, league_query: Optional[str] = None, *, max_retries: int = 2, retry_delay: float = 0.75, date_bucket: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch recent articles for a team; returns (articles, info) where info contains status/error."""
    date_bucket = date_bucket or datetime.now(timezone.utc).date().isoformat()
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
                "date_bucket": date_bucket,
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
        "date_bucket": date_bucket,
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


MAX_SENTIMENT_CALLS = 12


def _normalize_team_key(name: Any) -> str:
    cleaned = re.sub(r"[#.,]", " ", str(name or ""))
    cleaned = re.sub(r"\b(st)\b", "state", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]", "", cleaned.lower())
    tokens = [t for t in cleaned.split() if t]
    return " ".join(tokens)


def build_team_sentiment_map(
    news_api_key: str,
    games: List[Dict[str, Any]],
    league: str,
    *,
    existing_map: Optional[Dict[str, Optional[float]]] = None,
    existing_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
    cooldown_until: Optional[Any] = None,
    max_calls: int = MAX_SENTIMENT_CALLS,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]], Dict[str, Any]]:

    existing_map = existing_map or {}
    existing_meta_map = existing_meta_map or {}
    existing_map_norm = {_normalize_team_key(k): v for k, v in existing_map.items()}
    existing_meta_norm = {_normalize_team_key(k): v for k, v in existing_meta_map.items()}

    seen_norm = set()
    ordered_teams: List[str] = []
    for g in games or []:
        for key in ("home_team", "away_team"):
            team_raw = g.get(key)
            team = str(team_raw) if team_raw is not None else ""
            norm = _normalize_team_key(team)
            if norm and norm not in seen_norm:
                seen_norm.add(norm)
                ordered_teams.append(team)

    now_utc = datetime.now(timezone.utc)
    cooldown_until_dt: Optional[datetime] = None
    if cooldown_until:
        try:
            if isinstance(cooldown_until, str):
                cooldown_until_dt = datetime.fromisoformat(cooldown_until)
                if cooldown_until_dt.tzinfo is None:
                    cooldown_until_dt = cooldown_until_dt.replace(tzinfo=timezone.utc)
            elif isinstance(cooldown_until, datetime):
                cooldown_until_dt = cooldown_until
            else:
                cooldown_until_dt = None
        except Exception:
            cooldown_until_dt = None
    cooldown_active = bool(cooldown_until_dt and now_utc < cooldown_until_dt)

    sentiment_map: Dict[str, Optional[float]] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {
        "total_teams": len(ordered_teams),
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
        "calls_made": 0,
        "calls_limit": max_calls,
        "calls_capped": False,
        "used_cached": False,
        "cached_teams": 0,
        "cooldown_active": cooldown_active,
        "cooldown_until": cooldown_until_dt.isoformat() if cooldown_until_dt else None,
    }
    if cooldown_active:
        debug["rate_limited"] = True

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

    date_bucket = now_utc.date().isoformat()
    analyzer = RealSentimentAnalyzer(news_api_key) if RealSentimentAnalyzer and news_api_key else None

    for team in ordered_teams:
        norm_key = _normalize_team_key(team)
        prev_meta = existing_meta_norm.get(norm_key) or {}
        prev_score = existing_map_norm.get(norm_key)
        fetch_info: Dict[str, Any] = {}
        raw_payload: Dict[str, Any] = {}

        if cooldown_active or debug["calls_made"] >= max_calls or not news_api_key:
            debug["calls_capped"] = debug["calls_capped"] or debug["calls_made"] >= max_calls
            if prev_meta:
                used_score = prev_score if prev_meta.get("sentiment_valid") else None
                sentiment_map[team] = used_score
                meta_map[team] = {**prev_meta, "cached": True}
                debug["article_counts"][team] = int(prev_meta.get("sources") or prev_meta.get("articles") or 0)
                debug["articles_total"] += debug["article_counts"][team]
                debug["cached_teams"] += 1
                debug["used_cached"] = True
            else:
                sentiment_map[team] = None
                meta_map[team] = {
                    "sentiment_valid": False,
                    "articles": 0,
                    "sentiment_source": "none",
                    "error": "cooldown_active" if cooldown_active else "calls_capped" if debug["calls_capped"] else "missing_key",
                }
                debug["missing_teams"].append(team)
            continue

        try:
            query_label = league_label(league)
            if analyzer:
                raw_payload = analyzer.get_team_sentiment(team, league) or {}
                fetch_info = raw_payload.get("fetch_info") or {}
                debug["calls_made"] += 1
            if not raw_payload:
                articles, fetch_info = fetch_team_news(
                    news_api_key, team, league, query_label, date_bucket=date_bucket
                )
                debug["calls_made"] += 1
                score = team_sentiment_from_articles(articles)
                raw_payload = {
                    "score": score,
                    "confidence": 0.6 if articles else 0.0,
                    "sources": len(articles),
                    "method": "newsapi_simple" if articles else "none",
                }
            status_int = _record_status(fetch_info or {})
            if len(debug["sample_calls"]) < 10:
                debug["sample_calls"].append(
                    {
                        "team": team,
                        "league": league,
                        "q": (fetch_info or {}).get("q"),
                        "status": status_int,
                        "totalResults": (fetch_info or {}).get("totalResults"),
                        "error": (fetch_info or {}).get("error"),
                    }
                )

            meta = sentiment_payload_to_meta(raw_payload)
            meta_map[team] = {**meta, "error": (fetch_info or {}).get("error"), "cached": False}
            debug["article_counts"][team] = meta.get("sources") or 0
            debug["articles_total"] += meta.get("sources") or 0
            if (fetch_info or {}).get("rate_limited"):
                debug["rate_limited"] = True
                cooldown_until_dt = now_utc + timedelta(minutes=20)
                debug["cooldown_until"] = cooldown_until_dt.isoformat()
                cooldown_active = True
            if (fetch_info or {}).get("auth_error"):
                debug["auth_error"] = True
            if (fetch_info or {}).get("error"):
                debug["error_count"] += 1
                if len(debug["errors_sample"]) < 5:
                    debug["errors_sample"].append(
                        {"team": team, "error": (fetch_info or {}).get("error"), "status_code": (fetch_info or {}).get("status_code")}
                    )
            if meta["sentiment_valid"]:
                sentiment_map[team] = meta["score"]
            elif prev_meta.get("sentiment_valid"):
                sentiment_map[team] = prev_score
                meta_map[team] = {**prev_meta, "cached": True}
                debug["cached_teams"] += 1
                debug["used_cached"] = True
                debug["article_counts"][team] = int(prev_meta.get("sources") or prev_meta.get("articles") or 0)
                debug["articles_total"] += debug["article_counts"][team]
            else:
                sentiment_map[team] = None
                debug["missing_teams"].append(team)

            if debug["rate_limited"]:
                # Stop additional fetches; remaining teams can only use cached values.
                continue
        except Exception as exc:  # pragma: no cover - defensive
            sentiment_map[team] = prev_score if prev_meta.get("sentiment_valid") else None
            meta_map[team] = {
                **prev_meta,
                "sentiment_valid": bool(prev_meta.get("sentiment_valid")),
                "articles": int(prev_meta.get("sources") or prev_meta.get("articles") or 0),
                "sentiment_source": prev_meta.get("sentiment_source") or ("newsapi" if prev_meta.get("sentiment_valid") else "none"),
                "error": str(exc),
                "cached": bool(prev_meta),
            }
            debug["error_count"] += 1
            if len(debug["errors_sample"]) < 5:
                debug["errors_sample"].append({"team": team, "error": str(exc)})
            if meta_map[team]["sentiment_valid"]:
                debug["cached_teams"] += 1
                debug["used_cached"] = True
                debug["article_counts"][team] = meta_map[team]["articles"]
                debug["articles_total"] += meta_map[team]["articles"]
            else:
                debug["missing_teams"].append(team)
            continue

    if sentiment_map:
        sorted_scores = sorted(sentiment_map.items(), key=lambda kv: kv[1] if kv[1] is not None else -999)
        debug["bottom_5"] = [kv for kv in sorted_scores if kv[1] is not None][:5]
        debug["top_5"] = [kv for kv in sorted_scores if kv[1] is not None][-5:]
    else:
        debug["bottom_5"] = []
        debug["top_5"] = []
    debug["rate_limited"] = bool(debug["rate_limited"] or debug["status_counts"].get(429))
    debug["auth_error"] = bool(debug["auth_error"] or debug["status_counts"].get(401) or debug["status_counts"].get(403))
    if debug["rate_limited"] and not debug["status_counts"].get(429):
        debug["status_counts"][429] = 1

    return sentiment_map, meta_map, debug
