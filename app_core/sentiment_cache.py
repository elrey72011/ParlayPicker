"""
Sentiment Cache Module
Implements JSON-based caching with 12-hour TTL to reduce NewsAPI rate limit issues.

Created as part of data quality fixes - Issue #2.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SentimentCache:
    """
    Manages sentiment data caching with time-to-live (TTL) expiration.

    Cache structure:
    {
        "team_name": {
            "timestamp": "2026-01-22T10:30:00",
            "sentiment_score": 0.45,
            "sentiment_label": "Positive",
            "fetch_info": {...}
        }
    }
    """

    def __init__(self, cache_file: str = ".sentiment_cache.json", ttl_hours: int = 12):
        """
        Initialize the sentiment cache.

        Args:
            cache_file: Path to JSON cache file (default: .sentiment_cache.json)
            ttl_hours: Time-to-live in hours for cache entries (default: 12)
        """
        self.cache_file = cache_file
        self.ttl_hours = ttl_hours
        self.cache_data: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache_data = json.load(f)
                logger.info(f"Sentiment cache loaded: {len(self.cache_data)} teams cached")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load sentiment cache: {e}. Starting with empty cache.")
                self.cache_data = {}
        else:
            logger.info("No existing sentiment cache found. Starting fresh.")
            self.cache_data = {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
            logger.debug(f"Sentiment cache saved: {len(self.cache_data)} teams")
        except IOError as e:
            logger.error(f"Failed to save sentiment cache: {e}")

    def _is_expired(self, timestamp_str: str) -> bool:
        """
        Check if a cached entry has exceeded the TTL.

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            True if expired, False otherwise
        """
        try:
            cached_time = datetime.fromisoformat(timestamp_str)
            expiry_time = cached_time + timedelta(hours=self.ttl_hours)
            return datetime.now() > expiry_time
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid timestamp in cache: {timestamp_str}. {e}")
            return True  # Treat invalid timestamps as expired

    def get(self, team_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached sentiment data for a team if valid and not expired.

        Args:
            team_name: Name of the team

        Returns:
            Dict with sentiment data if cache hit, None if cache miss or expired
        """
        if not team_name:
            return None

        # Normalize team name for consistent cache keys
        team_key = str(team_name).strip().lower()

        if team_key not in self.cache_data:
            logger.debug(f"SENTIMENT CACHE MISS: {team_name} (not in cache)")
            return None

        entry = self.cache_data[team_key]
        timestamp = entry.get("timestamp")

        if not timestamp:
            logger.warning(f"Cache entry for {team_name} missing timestamp. Treating as expired.")
            return None

        if self._is_expired(timestamp):
            logger.debug(f"SENTIMENT CACHE EXPIRED: {team_name} (cached at {timestamp})")
            # Remove expired entry
            del self.cache_data[team_key]
            self._save_cache()
            return None

        logger.debug(f"SENTIMENT CACHE HIT: {team_name} (cached at {timestamp})")
        return {
            "sentiment_score": entry.get("sentiment_score"),
            "sentiment_label": entry.get("sentiment_label"),
            "fetch_info": entry.get("fetch_info", {}),
            "cached": True,
            "cached_at": timestamp,
        }

    def set(self, team_name: str, sentiment_score: float, sentiment_label: str,
            fetch_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Store sentiment data in cache with current timestamp.

        Args:
            team_name: Name of the team
            sentiment_score: Sentiment score (-1.0 to 1.0)
            sentiment_label: Sentiment label (e.g., "Positive", "Negative", "Neutral")
            fetch_info: Optional metadata about the fetch (API status, etc.)
        """
        if not team_name:
            logger.warning("Cannot cache sentiment: team_name is empty")
            return

        team_key = str(team_name).strip().lower()

        self.cache_data[team_key] = {
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "fetch_info": fetch_info or {},
        }

        self._save_cache()
        logger.debug(f"SENTIMENT CACHED: {team_name} (score={sentiment_score:.2f}, label={sentiment_label})")

    def clear(self) -> int:
        """
        Clear all cached sentiment data.

        Returns:
            Number of entries cleared
        """
        count = len(self.cache_data)
        self.cache_data = {}
        self._save_cache()
        logger.info(f"Sentiment cache cleared: {count} entries removed")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (total entries, expired entries, valid entries)
        """
        total = len(self.cache_data)
        expired = sum(1 for entry in self.cache_data.values()
                     if self._is_expired(entry.get("timestamp", "")))
        valid = total - expired

        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": expired,
            "ttl_hours": self.ttl_hours,
            "cache_file": self.cache_file,
        }

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        before_count = len(self.cache_data)

        # Find expired keys
        expired_keys = [
            key for key, entry in self.cache_data.items()
            if self._is_expired(entry.get("timestamp", ""))
        ]

        # Remove expired entries
        for key in expired_keys:
            del self.cache_data[key]

        if expired_keys:
            self._save_cache()
            removed = len(expired_keys)
            logger.info(f"Cleaned up {removed} expired sentiment cache entries")
            return removed

        return 0


# Global cache instance (singleton pattern)
_cache_instance: Optional[SentimentCache] = None


def get_cache() -> SentimentCache:
    """
    Get the global sentiment cache instance (singleton).

    Returns:
        SentimentCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SentimentCache()
    return _cache_instance
