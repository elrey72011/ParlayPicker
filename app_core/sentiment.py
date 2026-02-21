"""Sentiment analysis helpers shared by the Streamlit UI and CLI tools."""
from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests
from app_core.sentiment_pipeline import league_label
from app_core.sentiment_cache import (
    get_cache,
    is_cooldown_active,
    save_persistent_cooldown,
    get_cooldown_remaining_seconds,
)

logger = logging.getLogger(__name__)

# Track if we've already logged the rate limit message this session
_rate_limit_logged = False

# Module-level dedup cache (Fix 4)
_SENTIMENT_CACHE = {}   # {team_name_lower: result_dict}
_SENTIMENT_RATE_LIMITED = False

class RealSentimentAnalyzer:
    """Provide sentiment signals using NewsAPI articles or a neutral fallback."""

    def __init__(self, news_api_key: Optional[str] = None) -> None:
        self.news_api_key = news_api_key or os.environ.get("NEWS_API_KEY")
        self.sentiment_cache: Dict[str, Dict] = {}
        self.cache_duration = 1800  # 30 minutes

        self.positive_words = {
            "win",
            "wins",
            "won",
            "winning",
            "victory",
            "beat",
            "beats",
            "dominant",
            "strong",
            "excellent",
            "best",
            "great",
            "hot",
            "streak",
            "momentum",
            "comeback",
            "champion",
            "star",
            "explosive",
            "impressive",
            "outstanding",
            "stellar",
            "clutch",
            "elite",
            "record-breaking",
            "unstoppable",
            "phenomenal",
            "surging",
            "rolling",
        }

        self.negative_words = {
            "lose",
            "loses",
            "lost",
            "losing",
            "defeat",
            "beaten",
            "weak",
            "poor",
            "worst",
            "bad",
            "cold",
            "slump",
            "struggle",
            "injury",
            "injured",
            "hurt",
            "out",
            "questionable",
            "doubtful",
            "blow",
            "collapse",
            "disaster",
            "awful",
            "terrible",
            "embarrassing",
            "turnover",
            "frustrated",
            "disappointing",
            "concerning",
            "worry",
        }

    def get_team_sentiment(self, team_name: str, sport: str) -> Dict[str, float]:
        """Return a sentiment payload for the requested team."""
        global _rate_limit_logged, _SENTIMENT_RATE_LIMITED

        # ── FAST EXIT: if ANY prior call got a 403/rate-limit this run,
        #    skip ALL further network activity and return neutral instantly.
        if _SENTIMENT_RATE_LIMITED:
            if not _rate_limit_logged:
                logger.warning(
                    "NewsAPI rate limit already hit this run. "
                    "Skipping sentiment for all remaining teams (sentiment_weight=0)."
                )
                _rate_limit_logged = True
            return {
                **self._fallback_neutral(),
                "method": "Rate Limited (session)",
                "fetch_info": {"rate_limited": True, "skipped": True},
            }

        persistent_cache = get_cache()

        # Check if we're in a rate limit cooldown period
        # If so, use stale cache data as fallback to avoid hitting the API
        if is_cooldown_active():
            remaining_hours = get_cooldown_remaining_seconds() / 3600.0

            # Only log once per session to avoid spam
            if not _rate_limit_logged:
                logger.warning(
                    f"NewsAPI rate limit cooldown active ({remaining_hours:.1f}h remaining). "
                    f"Using cached sentiment data where available."
                )
                _rate_limit_logged = True

            # Try to get stale cache data as fallback
            cached_data = persistent_cache.get(team_name, allow_stale=True)
            if cached_data:
                return {
                    "score": cached_data.get("sentiment_score", 0.0),
                    "confidence": 0.5 if cached_data.get("is_stale") else 0.8,
                    "sources": 0,
                    "trend": cached_data.get("sentiment_label", "neutral").lower(),
                    "method": f"Cached (Stale)" if cached_data.get("is_stale") else "Cached",
                    "fetch_info": {
                        "cached": True,
                        "cached_at": cached_data.get("cached_at"),
                        "is_stale": cached_data.get("is_stale", False),
                        "age_hours": cached_data.get("age_hours", 0),
                        "rate_limited": True,
                    },
                    "cached": True,
                    "cached_at": cached_data.get("cached_at"),
                }
            else:
                # No cache available, return neutral without hitting API
                return {
                    **self._fallback_neutral(),
                    "method": "Rate Limited (No Cache)",
                    "fetch_info": {"rate_limited": True, "no_cache": True},
                }

        # Not rate limited - check persistent cache first (fresh data)
        cached_data = persistent_cache.get(team_name)
        if cached_data:
            # Cache hit - return cached sentiment
            return {
                "score": cached_data.get("sentiment_score", 0.0),
                "confidence": 0.8,  # High confidence for cached data
                "sources": 0,  # Unknown, but cached
                "trend": cached_data.get("sentiment_label", "neutral").lower(),
                "method": "Persistent Cache",
                "fetch_info": cached_data.get("fetch_info", {}),
                "cached": True,
                "cached_at": cached_data.get("cached_at"),
            }

        # Cache miss - check in-memory cache for backwards compatibility
        cache_key = f"{team_name}_{sport}_{datetime.now().date()}"

        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = (datetime.now() - cached["timestamp"]).seconds
            if age < self.cache_duration:
                return cached["data"]

        # No cache - fetch from NewsAPI
        if self.news_api_key:
            result = self._analyze_with_newsapi(team_name, sport)
        else:
            result = self._fallback_neutral()

        # Check if we hit a rate limit and need to set cooldown
        fetch_info = result.get("fetch_info", {})
        if fetch_info.get("rate_limited") or fetch_info.get("auth_error"):
            # Set 24-hour cooldown to avoid spamming the API
            cooldown_until = datetime.now() + timedelta(hours=24)
            save_persistent_cooldown(
                cooldown_until,
                reason=f"HTTP {fetch_info.get('status_code', 'unknown')}"
            )
            logger.warning(
                f"NewsAPI rate limit hit (HTTP {fetch_info.get('status_code')}). "
                f"Setting 24-hour cooldown until {cooldown_until.isoformat()}"
            )
            _rate_limit_logged = True

        # Store in in-memory cache
        self.sentiment_cache[cache_key] = {
            "data": result,
            "timestamp": datetime.now(),
        }

        # Store in persistent cache for 24-hour reuse
        # Only cache if we got a successful result (not rate limited)
        if not fetch_info.get("rate_limited", False) and not fetch_info.get("auth_error", False):
            sentiment_score = result.get("score", 0.0)
            sentiment_label = result.get("trend", "neutral").capitalize()
            persistent_cache.set(team_name, sentiment_score, sentiment_label, fetch_info)

        return result

    def _analyze_with_newsapi(self, team_name: str, sport: str) -> Dict:
        """Analyze sentiment using NewsAPI.org articles and capture HTTP details."""
        global _SENTIMENT_RATE_LIMITED

        # Check in-memory dedup cache first (Fix 4)
        cache_key = team_name.lower().strip()
        if cache_key in _SENTIMENT_CACHE:
            return _SENTIMENT_CACHE[cache_key]

        if _SENTIMENT_RATE_LIMITED:
            return {"score": None, "status": "rate_limited", "sources": 0}

        from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        sport_norm = (sport or "").upper()
        league_fragment = league_label(sport_norm)
        query = f'"{team_name}" {league_fragment}'.strip()

        fetch_info: Dict[str, Optional[str]] = {
            "q": query,
            "league_query": league_fragment,
            "rate_limited": False,
            "auth_error": False,
        }

        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "to": to_date,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 20,
                    "apiKey": self.news_api_key,
                },
                timeout=10,
            )
            fetch_info["status"] = response.status_code
            fetch_info["status_code"] = response.status_code
            # NewsAPI returns 429 for per-minute limits, but 403 for daily developer limits
            # Both should be treated as rate limiting for cooldown purposes
            fetch_info["rate_limited"] = response.status_code in {429, 403}
            fetch_info["auth_error"] = response.status_code == 401  # Only 401 is truly auth error
            if response.status_code != 200:
                error_key = "rate_limited" if fetch_info["rate_limited"] else ("bad_key" if fetch_info["auth_error"] else "http_error")
                fetch_info["error"] = error_key
                fetch_info["response_text_snippet"] = (response.text or "")[:200]
                try:
                    data_err = response.json() if response is not None else {}
                except Exception:
                    data_err = {}
                articles = data_err.get("articles", []) if isinstance(data_err, dict) else []
                fetch_info["totalResults"] = data_err.get("totalResults") if isinstance(data_err, dict) else None
                if fetch_info["rate_limited"] and articles:
                    # Partial success even while rate limited
                    response = None  # skip additional parsing
                else:
                    if fetch_info["rate_limited"]:
                        _SENTIMENT_RATE_LIMITED = True
                        result = {"score": None, "status": "rate_limited", "sources": 0}
                        _SENTIMENT_CACHE[cache_key] = result
                        return result

                    result = {
                        **self._fallback_neutral(),
                        "method": "newsapi_error",
                        "fetch_info": fetch_info,
                    }
                    _SENTIMENT_CACHE[cache_key] = result
                    return result

            data = response.json() if response is not None else data_err if "data_err" in locals() else {}
            articles = data.get("articles", []) if isinstance(data, dict) else []
            fetch_info["totalResults"] = data.get("totalResults") if isinstance(data, dict) else None
            if not articles:
                fetch_info["error"] = fetch_info.get("error") or "no_articles"
                return {
                    **self._fallback_neutral(),
                    "method": "newsapi_empty",
                    "fetch_info": fetch_info,
                }

            sentiment_scores = []
            for article in articles[:20]:
                text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                score = self._calculate_text_sentiment(text)
                sentiment_scores.append(score)

            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                score_variance = (
                    sum((s - avg_score) ** 2 for s in sentiment_scores) / len(sentiment_scores)
                )
                confidence = max(0.3, min(0.95, 1.0 - score_variance))

                trend = "positive" if avg_score > 0.15 else ("negative" if avg_score < -0.15 else "neutral")

                result = {
                    "score": avg_score,
                    "confidence": confidence,
                    "sources": len(sentiment_scores),
                    "trend": trend,
                    "method": "NewsAPI + NLP",
                    "fetch_info": fetch_info,
                }
                _SENTIMENT_CACHE[cache_key] = result
                return result

            fetch_info["error"] = fetch_info.get("error") or "no_articles"
            result = {
                **self._fallback_neutral(),
                "method": "newsapi_empty",
                "fetch_info": fetch_info,
            }
            _SENTIMENT_CACHE[cache_key] = result
            return result

        except Exception as exc:
            fetch_info["error"] = str(exc)
            if "429" in str(exc) or "403" in str(exc):
                _SENTIMENT_RATE_LIMITED = True
                result = {"score": None, "status": "rate_limited", "sources": 0}
                _SENTIMENT_CACHE[cache_key] = result
                return result

            result = {
                **self._fallback_neutral(),
                "method": "newsapi_exception",
                "fetch_info": fetch_info,
            }
            _SENTIMENT_CACHE[cache_key] = result
            return result

    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate a normalized sentiment score using keyword counts."""

        words = re.findall(r"\b\w+\b", text.lower())
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        sentiment_score = (positive_count - negative_count) / total * 0.7
        return max(-1.0, min(1.0, sentiment_score))

    def _fallback_neutral(self) -> Dict:
        """Return neutral sentiment when the API is unavailable."""

        return {
            "score": 0.0,
            "confidence": 0.2,
            "sources": 0,
            "trend": "neutral",
            "method": "No API key",
        }


SentimentAnalyzer = RealSentimentAnalyzer

__all__ = ["RealSentimentAnalyzer", "SentimentAnalyzer"]
