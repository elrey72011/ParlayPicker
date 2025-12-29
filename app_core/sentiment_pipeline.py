"""Lightweight sentiment pipeline using NewsAPI with keyword lexicon scoring."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from app_core.reddit_sentiment import fetch_reddit_sentiment_map

# ... existing imports and constants ...

class SentimentPipeline:
    """Facade for sentiment analysis operations."""

    @staticmethod
    def build_team_sentiment_map(
        news_api_key: str,
        games: List[Dict[str, Any]],
        league: str,
        *,
        existing_map: Optional[Dict[str, Optional[float]]] = None,
        existing_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
        cooldown_until: Optional[Any] = None,
        max_calls: int = 6,
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        return build_team_sentiment_map(
            news_api_key,
            games,
            league,
            existing_map=existing_map,
            existing_meta_map=existing_meta_map,
            cooldown_until=cooldown_until,
            max_calls=max_calls
        )

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


def _normalize_sentiment_level(level: Any) -> str:
    try:
        lvl = str(level or "none").lower()
    except Exception:
        lvl = "none"
    return lvl if lvl in {"team", "game", "league", "none"} else "none"


def sentiment_strength_from_articles(level: str, articles_used: Any) -> str:
    lvl = _normalize_sentiment_level(level)
    try:
        count_val = int(articles_used or 0)
    except Exception:
        count_val = 0
    if lvl == "none" or count_val <= 0:
        return "NONE"
    if count_val >= 8:
        return "STRONG"
    if count_val >= 3:
        return "MEDIUM"
    return "WEAK"


def sentiment_badge_for(level: str, strength: str) -> str:
    lvl = _normalize_sentiment_level(level)
    strength_norm = str(strength or "").upper()
    mapping = {
        "team": {"STRONG": "TEAM_STRONG", "MEDIUM": "TEAM_MED", "WEAK": "TEAM_WEAK"},
        "game": {"STRONG": "GAME_STRONG", "MEDIUM": "GAME_MED", "WEAK": "GAME_WEAK"},
        "league": {"STRONG": "LEAGUE_MED", "MEDIUM": "LEAGUE_MED", "WEAK": "LEAGUE_WEAK"},
    }
    if lvl in mapping and strength_norm in mapping[lvl]:
        return mapping[lvl][strength_norm]
    return "NONE"


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
    # Use simple team name only to avoid 'restricted query' errors on free tier (User Request 3)
    return f'"{team}"'.strip()


@st.cache_data(ttl=21600)
def fetch_team_newsapi_cached(
    team: str,
    league: str,
    news_api_key: str,
    date_bucket: Optional[str] = None,
    league_query: Optional[str] = None,
    query_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch NewsAPI once for a team or custom query and return normalized debug payload without raising."""
    meta: Dict[str, Any] = {
        "status": "NO_CALL",
        "q_used": "",
        "totalResults": 0,
        "articles_count": 0,
        "sentiment": None,
        "error": None,
    }
    try:
        date_bucket = date_bucket or datetime.now(timezone.utc).date().isoformat()
        league_q = league_query or league_label(league)
        q_text = query_override or _newsapi_query(team, league, league_q)
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=3)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": q_text,
            "sortBy": "relevancy",
            "pageSize": 20,
            "language": "en",
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "apiKey": news_api_key,
        }
        resp = requests.get(url, params=params, timeout=8)
        status_val = resp.status_code
        try:
            data = resp.json() if hasattr(resp, "json") else {}
        except Exception:
            data = {}
        articles = data.get("articles", []) if isinstance(data, dict) else []
        total_results = data.get("totalResults") if isinstance(data, dict) else None
        meta["status"] = int(status_val) if str(status_val).isdigit() else str(status_val)
        meta["q_used"] = q_text
        meta["totalResults"] = int(total_results or 0)
        meta["articles_count"] = len(articles or [])
        if meta["articles_count"] > 0:
            meta["sentiment"] = float(team_sentiment_from_articles(articles))
        meta["error"] = None if status_val == 200 else data.get("message") if isinstance(data, dict) else None
    except Exception as exc:
        meta["status"] = "EXCEPTION"
        meta["error"] = str(exc)
        meta["q_used"] = meta.get("q_used") or _newsapi_query(team, league, league_query or league_label(league))
    return meta


@st.cache_data(ttl=21600)
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
            retry_after_hdr = None
            try:
                retry_after_hdr = resp.headers.get("Retry-After")
            except Exception:
                retry_after_hdr = None
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
                    "retry_after": retry_after_hdr,
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
                "retry_after": retry_after_hdr,
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


@st.cache_data(ttl=21600)
def fetch_team_sentiment_cached(
    news_api_key: str,
    team: str,
    league: str,
    *,
    league_query: Optional[str] = None,
    date_bucket: Optional[str] = None,
    max_retries: int = 2,
    retry_delay: float = 0.75,
) -> Dict[str, Any]:
    """Cached wrapper around NewsAPI fetch + simple scoring."""
    date_bucket = date_bucket or datetime.now(timezone.utc).date().isoformat()
    articles, info = fetch_team_news(
        news_api_key,
        team,
        league,
        league_query,
        max_retries=max_retries,
        retry_delay=retry_delay,
        date_bucket=date_bucket,
    )
    score = team_sentiment_from_articles(articles)
    sources = len(articles)
    status_val = info.get("status") or info.get("status_code")
    try:
        status_int = int(status_val) if status_val is not None else None
    except Exception:
        status_int = None
    return {
        "team": team,
        "league": league,
        "q": info.get("q"),
        "sentiment": score if sources > 0 else None,
        "articles_count": sources,
        "status": status_int,
        "totalResults": info.get("totalResults"),
        "error": info.get("error"),
        "rate_limited": bool(info.get("rate_limited")),
        "auth_error": bool(info.get("auth_error")),
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


MAX_SENTIMENT_CALLS = 6
COOLDOWN_HOURS = 12


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
    if len(ordered_teams) > max_calls:
        ordered_teams = ordered_teams[:max_calls]
        debug_trimmed = True
    else:
        debug_trimmed = False

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
        "reddit_posts_used": 0,
        "reddit_comments_used": 0,
        "teams_from_reddit": 0,
        "teams_blended": 0,
    }
    debug["calls_capped"] = debug_trimmed
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
    # Import RealSentimentAnalyzer dynamically or assume available via injection if integrated
    from app_core.sentiment import RealSentimentAnalyzer
    analyzer = RealSentimentAnalyzer(news_api_key) if RealSentimentAnalyzer and news_api_key else None
    stop_fetching = cooldown_active or False

    for team in ordered_teams:
        norm_key = _normalize_team_key(team)
        prev_meta = existing_meta_norm.get(norm_key) or {}
        prev_score = existing_map_norm.get(norm_key)

        if stop_fetching or debug["calls_made"] >= max_calls or not news_api_key:
            debug["calls_capped"] = debug["calls_capped"] or debug["calls_made"] >= max_calls or stop_fetching
            if prev_meta:
                used_score = prev_score if prev_meta.get("sentiment_valid") else None
                sentiment_map[team] = used_score
                meta_map[team] = {**prev_meta, "cached": True}
                sources_prev = int(prev_meta.get("sources") or prev_meta.get("articles") or 0)
                debug["article_counts"][team] = sources_prev
                debug["articles_total"] += sources_prev
                if prev_meta.get("sentiment_valid"):
                    debug["cached_teams"] += 1
                    debug["used_cached"] = True
                else:
                    debug["missing_teams"].append(team)
            else:
                sentiment_map[team] = None
                meta_map[team] = {
                    "sentiment_valid": False,
                    "articles": 0,
                    "sentiment_source": "none",
                    "sentiment_level": "none",
                    "sentiment_strength": "NONE",
                    "sentiment_badge": "NONE",
                    "sentiment_articles_used": 0,
                    "sentiment_query_used": None,
                    "error": "cooldown_active" if stop_fetching else "calls_capped" if debug["calls_capped"] else "missing_key",
                    "cached": False,
                }
                debug["missing_teams"].append(team)
            continue

        try:
            query_label = league_label(league)
            result_payload: Dict[str, Any] = {}
            fetch_info: Dict[str, Any] = {}
            if analyzer:
                result_payload = analyzer.get_team_sentiment(team, league) or {}
                fetch_info = result_payload.get("fetch_info") or {}
                debug["calls_made"] += 1
            if not result_payload:
                debug["calls_made"] += 1
                result_payload = fetch_team_sentiment_cached(
                    news_api_key,
                    team,
                    league,
                    league_query=query_label,
                    date_bucket=date_bucket,
                    max_retries=2,
                    retry_delay=0.75,
                )
                fetch_info = {
                    "status": result_payload.get("status"),
                    "status_code": result_payload.get("status"),
                    "q": result_payload.get("q"),
                    "totalResults": result_payload.get("totalResults"),
                    "rate_limited": result_payload.get("rate_limited"),
                    "auth_error": result_payload.get("auth_error"),
                    "error": result_payload.get("error"),
                }
            status_int = _record_status(fetch_info or {})
            if status_int is None and (fetch_info or {}).get("rate_limited"):
                status_int = 429
                debug["status_counts"][429] = debug["status_counts"].get(429, 0) + 1
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
            if status_int == 429:
                debug["rate_limited"] = True
                cooldown_until_dt = now_utc + timedelta(hours=COOLDOWN_HOURS)
                debug["cooldown_until"] = cooldown_until_dt.isoformat()
                try:
                    st.session_state["sentiment_cooldown_until"] = cooldown_until_dt.isoformat()
                except Exception:
                    pass
                stop_fetching = True
                break

            sources = int(result_payload.get("articles_count") or 0)
            score_val = result_payload.get("sentiment")
            meta = {
                "score": score_val if sources > 0 else None,
                "confidence": 0.6 if sources > 0 else 0.0,
                "sources": sources,
                "sentiment_valid": bool(sources > 0 and score_val is not None),
                "sentiment_source": "newsapi" if sources > 0 else "none",
                "sentiment_level": "team" if sources > 0 else "none",
                "sentiment_articles_used": sources,
                "sentiment_strength": sentiment_strength_from_articles("team", sources) if sources > 0 else "NONE",
                "sentiment_badge": sentiment_badge_for("team", sentiment_strength_from_articles("team", sources)) if sources > 0 else "NONE",
                "sentiment_query_used": (fetch_info or {}).get("q"),
                "status": status_int,
                "method": "newsapi_simple" if sources > 0 else "none",
                "reddit_used": False,
                "rate_limited": bool((fetch_info or {}).get("rate_limited")),
                "auth_error": bool((fetch_info or {}).get("auth_error")),
            }
            meta_map[team] = {**meta, "error": (fetch_info or {}).get("error"), "cached": False}
            debug["article_counts"][team] = sources
            debug["articles_total"] += sources
            if (fetch_info or {}).get("rate_limited"):
                debug["rate_limited"] = True
                cooldown_until_dt = now_utc + timedelta(minutes=20)
                debug["cooldown_until"] = cooldown_until_dt.isoformat()
                stop_fetching = True
            if (fetch_info or {}).get("auth_error"):
                debug["auth_error"] = True
            if (fetch_info or {}).get("error"):
                debug["error_count"] += 1
                if len(debug["errors_sample"]) < 5:
                    debug["errors_sample"].append(
                        {"team": team, "error": (fetch_info or {}).get("error"), "status_code": (fetch_info or {}).get("status_code") or status_int}
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
        except Exception as exc:  # pragma: no cover - defensive
            sentiment_map[team] = prev_score if prev_meta.get("sentiment_valid") else None
            meta_map[team] = {
                **prev_meta,
                "sentiment_valid": bool(prev_meta.get("sentiment_valid")),
                "articles": int(prev_meta.get("sources") or prev_meta.get("articles") or 0),
                "sentiment_source": prev_meta.get("sentiment_source") or ("newsapi" if prev_meta.get("sentiment_valid") else "none"),
                "sentiment_level": prev_meta.get("sentiment_level") or ("team" if prev_meta.get("sentiment_valid") else "none"),
                "sentiment_strength": prev_meta.get("sentiment_strength") or sentiment_strength_from_articles(prev_meta.get("sentiment_level") or "none", prev_meta.get("sources") or 0),
                "sentiment_badge": prev_meta.get("sentiment_badge") or sentiment_badge_for(prev_meta.get("sentiment_level") or "none", prev_meta.get("sentiment_strength") or "NONE"),
                "sentiment_articles_used": prev_meta.get("sentiment_articles_used") or prev_meta.get("sources") or 0,
                "sentiment_query_used": prev_meta.get("sentiment_query_used") or None,
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

    missing_teams = [t for t, v in sentiment_map.items() if v is None]
    news_valid_count = len([v for v in sentiment_map.values() if v is not None])
    need_reddit = debug.get("rate_limited") or news_valid_count == 0 or bool(missing_teams)
    if need_reddit and missing_teams:
        reddit_results = fetch_reddit_sentiment_map(missing_teams, league)
        for team in missing_teams:
            payload = reddit_results.get(team) or {}
            score_val = payload.get("score")
            try:
                confidence_val = float(payload.get("confidence") or 0.0)
            except Exception:
                confidence_val = 0.0
            try:
                sources_val = int(payload.get("source_count") or 0)
            except Exception:
                sources_val = 0
            posts_used = int(payload.get("posts_used") or 0)
            comments_used = int(payload.get("comments_used") or 0)
            if score_val is None or sources_val <= 0:
                continue
            existing_meta = meta_map.get(team) or {}
            existing_conf = existing_meta.get("confidence") or 0.0
            blended_score = score_val
            source_label = "reddit"
            if existing_meta.get("sentiment_valid") and confidence_val and confidence_val > existing_conf:
                blended_score = 0.7 * float(existing_meta.get("score") or 0.0) + 0.3 * float(score_val)
                source_label = "blended"
            elif existing_meta.get("sentiment_valid"):
                continue
            sentiment_map[team] = blended_score
            sentiment_strength = sentiment_strength_from_articles("team", sources_val)
            sentiment_badge = sentiment_badge_for("team", sentiment_strength)
            meta_map[team] = {
                **existing_meta,
                "sentiment_valid": True,
                "sentiment_source": source_label,
                "sentiment_level": "team",
                "sentiment_strength": sentiment_strength,
                "sentiment_badge": sentiment_badge,
                "sentiment_articles_used": sources_val,
                "sentiment_source_count": sources_val,
                "sentiment_query_used": payload.get("query"),
                "score": blended_score,
                "confidence": confidence_val,
                "sentiment_confidence": confidence_val,
                "reddit_used": True,
                "reddit_posts_used": posts_used,
                "reddit_comments_used": comments_used,
                "error": payload.get("error"),
            }
            debug["reddit_posts_used"] += posts_used
            debug["reddit_comments_used"] += comments_used
            if source_label == "reddit":
                debug["teams_from_reddit"] += 1
            if source_label == "blended":
                debug["teams_blended"] += 1
            debug["article_counts"][team] = sources_val
            debug["articles_total"] += sources_val
            if team in debug["missing_teams"]:
                try:
                    debug["missing_teams"].remove(team)
                except ValueError:
                    pass
            status_key = payload.get("status") if payload.get("status") is not None else "reddit"
            try:
                status_int = int(status_key)
            except Exception:
                status_int = None
            if status_int is not None:
                debug["status_counts"][status_int] = debug["status_counts"].get(status_int, 0) + 1
            if payload.get("error"):
                debug["error_count"] += 1
                if len(debug["errors_sample"]) < 5:
                    debug["errors_sample"].append({"team": team, "error": payload.get("error"), "status_code": status_key})

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
