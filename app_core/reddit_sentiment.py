"""Lightweight Reddit sentiment fallback to supplement NewsAPI."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_analyzer = SentimentIntensityAnalyzer()


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        v = float(val)
    except Exception:
        return 0.0
    return max(lo, min(hi, v))


def _score_texts(texts: Iterable[str]) -> Tuple[float, float]:
    scores: List[float] = []
    for text in texts:
        text_clean = (text or "").strip()
        if not text_clean:
            continue
        score = _analyzer.polarity_scores(text_clean).get("compound", 0.0)
        scores.append(_clamp(score))
    if not scores:
        return 0.0, 0.0
    avg_score = sum(scores) / len(scores)
    mean_abs = sum(abs(s) for s in scores) / len(scores)
    confidence = _clamp(0.25 + min(0.45, mean_abs * 0.65) + min(0.2, len(scores) / 50.0), 0.0, 0.95)
    return avg_score, confidence


def _default_headers() -> Dict[str, str]:
    ua = os.environ.get("REDDIT_USER_AGENT") or "parlaypicker-sentiment/1.0 (by user)"
    return {"User-Agent": ua}


def _filter_reddit_posts(children: List[Dict[str, Any]], team_name: str, league: str) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    team_kw = team_name.lower()
    league_kw = (league or "").lower()
    strong_keywords = {" vs", "vs.", "matchup", "preview", "game", "gameday", "odds"}
    for child in children:
        data = child.get("data") if isinstance(child, dict) else None
        if not isinstance(data, dict):
            continue
        title = (data.get("title") or "").lower()
        body = (data.get("selftext") or "").lower()
        blob = f"{title} {body}".strip()
        if team_kw not in blob:
            continue
        if league_kw and league_kw not in blob and league_kw.replace(" ", "") not in blob:
            if not any(k in blob for k in strong_keywords):
                continue
        if not any(k in blob for k in strong_keywords):
            continue
        filtered.append(data)
    return filtered


def _fetch_comments_for_post(post_id: str, limit: int, headers: Dict[str, str]) -> List[str]:
    try:
        url = f"https://www.reddit.com/comments/{post_id}.json"
        resp = requests.get(url, params={"limit": max(1, min(limit, 100))}, headers=headers, timeout=6)
        if resp.status_code != 200:
            return []
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        comments_data = payload[1].get("data", {}) if isinstance(payload[1], dict) else {}
        children = comments_data.get("children", []) if isinstance(comments_data, dict) else []
        bodies: List[str] = []
        for child in children:
            cdata = child.get("data") if isinstance(child, dict) else None
            if not isinstance(cdata, dict):
                continue
            body = cdata.get("body") or ""
            if not body or "removed" in body.lower() or "deleted" in body.lower():
                continue
            bodies.append(str(body))
            if len(bodies) >= limit:
                break
        return bodies
    except Exception:
        return []


def fetch_reddit_sentiment_for_team(team_name: str, league: str, max_posts: int = 20, max_comments: int = 100) -> Dict[str, Any]:
    """
    Fetch Reddit sentiment for a team using public JSON endpoints.
    Returns a payload with score/label/confidence/source_count and debug info.
    """
    headers = _default_headers()
    query = f'"{team_name}" ({league} OR game OR vs OR matchup)'
    posts_used = 0
    comments_used = 0
    rate_limited = False
    status = None

    try:
        resp = requests.get(
            "https://www.reddit.com/search.json",
            params={
                "q": query,
                "sort": "new",
                "limit": max(5, min(max_posts, 50)),
                "type": "link",
                "restrict_sr": False,
                "t": "week",
            },
            headers=headers,
            timeout=8,
        )
        status = resp.status_code
        if status == 429:
            rate_limited = True
            return {
                "score": 0.0,
                "label": None,
                "confidence": 0.0,
                "source_count": 0,
                "posts_used": 0,
                "comments_used": 0,
                "rate_limited": True,
                "status": status,
                "error": "rate_limited",
                "query": query,
            }
        if status != 200:
            return {
                "score": 0.0,
                "label": None,
                "confidence": 0.0,
                "source_count": 0,
                "posts_used": 0,
                "comments_used": 0,
                "rate_limited": rate_limited,
                "status": status,
                "error": f"http_{status}",
                "query": query,
            }
        data = resp.json()
        children = (data or {}).get("data", {}).get("children", []) if isinstance(data, dict) else []
        filtered = _filter_reddit_posts(children, team_name, league)
        texts: List[str] = []
        for post in filtered:
            title = post.get("title") or ""
            selftext = post.get("selftext") or ""
            texts.append(f"{title}\n{selftext}")
            posts_used += 1
            if comments_used >= max_comments:
                continue
            post_id = post.get("id") or ""
            remaining_comments = max(0, max_comments - comments_used)
            comments = _fetch_comments_for_post(post_id, remaining_comments, headers=headers) if post_id else []
            texts.extend(comments)
            comments_used += len(comments)

        score, confidence = _score_texts(texts)
        label = "Neutral"
        if score > 0.1:
            label = "Positive"
        elif score < -0.1:
            label = "Negative"
        source_count = posts_used + comments_used
        return {
            "score": score if source_count > 0 else 0.0,
            "label": label if source_count > 0 else None,
            "confidence": confidence if source_count > 0 else 0.0,
            "source_count": source_count,
            "posts_used": posts_used,
            "comments_used": comments_used,
            "rate_limited": rate_limited,
            "status": status,
            "query": query,
            "raw": {
                "posts_considered": len(children),
                "posts_used": posts_used,
                "comments_used": comments_used,
            },
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "label": None,
            "confidence": 0.0,
            "source_count": 0,
            "posts_used": posts_used,
            "comments_used": comments_used,
            "rate_limited": rate_limited,
            "status": status,
            "error": str(exc),
            "query": query,
        }


def fetch_reddit_sentiment_map(teams: List[str], league: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetch Reddit sentiment for a list of teams; returns a mapping per team.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for team in teams or []:
        if not team:
            continue
        results[team] = fetch_reddit_sentiment_for_team(team, league)
    return results
