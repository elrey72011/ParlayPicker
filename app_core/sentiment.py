"""Sentiment analysis helpers shared by the Streamlit UI and CLI tools."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests


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

        cache_key = f"{team_name}_{sport}_{datetime.now().date()}"

        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = (datetime.now() - cached["timestamp"]).seconds
            if age < self.cache_duration:
                return cached["data"]

        if self.news_api_key:
            result = self._analyze_with_newsapi(team_name, sport)
        else:
            result = self._fallback_neutral()

        self.sentiment_cache[cache_key] = {
            "data": result,
            "timestamp": datetime.now(),
        }
        return result

    def _analyze_with_newsapi(self, team_name: str, sport: str) -> Dict:
        """Analyze sentiment using NewsAPI.org articles."""

        try:
            from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            query = f'"{team_name}"'
            sport_lower = sport.lower()
            if "nba" in sport_lower:
                query += " NBA basketball"
            elif "nfl" in sport_lower:
                query += " NFL football"
            elif "mlb" in sport_lower:
                query += " MLB baseball"
            elif "nhl" in sport_lower:
                query += " NHL hockey"

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

            if response.status_code != 200:
                return self._fallback_neutral()

            articles = response.json().get("articles", [])
            if not articles:
                return self._fallback_neutral()

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

                return {
                    "score": avg_score,
                    "confidence": confidence,
                    "sources": len(sentiment_scores),
                    "trend": trend,
                    "method": "NewsAPI + NLP",
                }

            return self._fallback_neutral()
        except Exception:
            return self._fallback_neutral()

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
