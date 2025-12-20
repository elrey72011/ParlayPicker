"""Lightweight sentiment pipeline using NewsAPI with keyword lexicon scoring."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
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


@st.cache_data(ttl=300)
def fetch_team_news(news_api_key: str, team: str, league: str) -> List[Dict[str, Any]]:
    """Fetch recent articles for a team; returns empty list on failure."""
    if not news_api_key:
        return []
    league_label = {
        "NBA": "NBA basketball",
        "NFL": "NFL football",
        "NCAAF": "college football",
        "NCAAB": "college basketball",
        "NHL": "NHL hockey",
        "MLB": "MLB baseball",
    }.get((league or "").upper(), league or "")
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=3)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{team}" {league_label}',
        "sortBy": "relevancy",
        "pageSize": 20,
        "language": "en",
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "apiKey": news_api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("articles", []) if isinstance(data, dict) else []
    except Exception:
        return []


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
    }

    for team in sorted(teams):
        try:
            articles = fetch_team_news(news_api_key, team, league)
            debug["article_counts"][team] = len(articles)
            debug["articles_total"] += len(articles)
            if not articles:
                sentiment_map[team] = None
                meta_map[team] = {
                    "sentiment_valid": False,
                    "articles": 0,
                    "sentiment_source": "newsapi",
                    "error": None,
                }
                debug["missing_teams"].append(team)
                continue
            score = team_sentiment_from_articles(articles)
            sentiment_map[team] = score
            meta_map[team] = {
                "sentiment_valid": True,
                "articles": len(articles),
                "sentiment_source": "newsapi",
                "error": None,
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

    return sentiment_map, meta_map, debug
