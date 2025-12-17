"""Lightweight sentiment pipeline using NewsAPI with keyword lexicon scoring."""
from __future__ import annotations

import re
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
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{team}" AND {league}',
        "sortBy": "publishedAt",
        "pageSize": 10,
        "language": "en",
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
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    teams = set()
    for g in games or []:
        if g.get("home_team"):
            teams.add(str(g.get("home_team")))
        if g.get("away_team"):
            teams.add(str(g.get("away_team")))

    sentiment_map: Dict[str, float] = {}
    debug: Dict[str, Any] = {
        "total_teams": len(teams),
        "article_counts": {},
        "missing_teams": [],
        "articles_total": 0,
    }

    for team in sorted(teams):
        articles = fetch_team_news(news_api_key, team, league)
        score = team_sentiment_from_articles(articles)
        sentiment_map[team] = score
        debug["article_counts"][team] = len(articles)
        debug["articles_total"] += len(articles)
        if not articles:
            debug["missing_teams"].append(team)

    if sentiment_map:
        sorted_scores = sorted(sentiment_map.items(), key=lambda kv: kv[1])
        debug["bottom_5"] = sorted_scores[:5]
        debug["top_5"] = sorted_scores[-5:]
    else:
        debug["bottom_5"] = []
        debug["top_5"] = []

    return sentiment_map, debug
