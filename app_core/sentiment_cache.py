"""
Sentiment Cache Module
Implements JSON-based caching with configurable TTL to reduce NewsAPI rate limit issues.

Extended to support:
- 24+ hour TTL for better rate limit protection
- Staleness tracking for graceful degradation
- Persistent cooldown tracking across app restarts

Created as part of data quality fixes - Issue #2.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Persistent cooldown file path
COOLDOWN_FILE = ".sentiment_cooldown.json"


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

    def __init__(self, cache_file: str = ".sentiment_cache.json", ttl_hours: int = 24, stale_ttl_hours: int = 72):
        """
        Initialize the sentiment cache.

        Args:
            cache_file: Path to JSON cache file (default: .sentiment_cache.json)
            ttl_hours: Time-to-live in hours for fresh cache entries (default: 24)
            stale_ttl_hours: Maximum age in hours before cache entry is unusable (default: 72)
        """
        self.cache_file = cache_file
        self.ttl_hours = ttl_hours
        self.stale_ttl_hours = stale_ttl_hours
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
        Check if a cached entry has exceeded the TTL (considered not fresh).

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            True if expired (not fresh), False otherwise
        """
        try:
            cached_time = datetime.fromisoformat(timestamp_str)
            expiry_time = cached_time + timedelta(hours=self.ttl_hours)
            return datetime.now() > expiry_time
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid timestamp in cache: {timestamp_str}. {e}")
            return True  # Treat invalid timestamps as expired

    def _is_stale(self, timestamp_str: str) -> bool:
        """
        Check if a cached entry has exceeded the stale TTL (completely unusable).

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            True if stale (unusable), False otherwise
        """
        try:
            cached_time = datetime.fromisoformat(timestamp_str)
            stale_time = cached_time + timedelta(hours=self.stale_ttl_hours)
            return datetime.now() > stale_time
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid timestamp in cache: {timestamp_str}. {e}")
            return True  # Treat invalid timestamps as stale

    def _get_age_hours(self, timestamp_str: str) -> float:
        """
        Get the age of a cache entry in hours.

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            Age in hours, or float('inf') if invalid
        """
        try:
            cached_time = datetime.fromisoformat(timestamp_str)
            delta = datetime.now() - cached_time
            return delta.total_seconds() / 3600.0
        except (ValueError, TypeError):
            return float('inf')

    def get(self, team_name: str, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached sentiment data for a team if valid and not expired.

        Args:
            team_name: Name of the team
            allow_stale: If True, return stale data with staleness indicator when fresh data unavailable

        Returns:
            Dict with sentiment data if cache hit, None if cache miss or expired/stale
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

        age_hours = self._get_age_hours(timestamp)
        is_fresh = not self._is_expired(timestamp)
        is_usable = not self._is_stale(timestamp)

        if is_fresh:
            logger.debug(f"SENTIMENT CACHE HIT (fresh): {team_name} (cached at {timestamp}, age={age_hours:.1f}h)")
            return {
                "sentiment_score": entry.get("sentiment_score"),
                "sentiment_label": entry.get("sentiment_label"),
                "fetch_info": entry.get("fetch_info", {}),
                "cached": True,
                "cached_at": timestamp,
                "is_stale": False,
                "age_hours": age_hours,
            }

        if allow_stale and is_usable:
            logger.info(f"SENTIMENT CACHE HIT (stale): {team_name} (cached at {timestamp}, age={age_hours:.1f}h)")
            return {
                "sentiment_score": entry.get("sentiment_score"),
                "sentiment_label": entry.get("sentiment_label"),
                "fetch_info": entry.get("fetch_info", {}),
                "cached": True,
                "cached_at": timestamp,
                "is_stale": True,
                "age_hours": age_hours,
            }

        if not is_usable:
            logger.debug(f"SENTIMENT CACHE STALE: {team_name} (cached at {timestamp}, age={age_hours:.1f}h > {self.stale_ttl_hours}h)")
            # Remove completely stale entry
            del self.cache_data[team_key]
            self._save_cache()
            return None

        logger.debug(f"SENTIMENT CACHE EXPIRED: {team_name} (cached at {timestamp}, age={age_hours:.1f}h)")
        # Don't remove expired-but-not-stale entries; they can still be used as fallback
        return None

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
            Dict with cache stats (total entries, expired entries, valid entries, stale entries)
        """
        total = len(self.cache_data)
        fresh = 0
        expired = 0
        stale = 0

        for entry in self.cache_data.values():
            timestamp = entry.get("timestamp", "")
            if not self._is_expired(timestamp):
                fresh += 1
            elif not self._is_stale(timestamp):
                expired += 1  # Expired but usable as fallback
            else:
                stale += 1  # Completely stale/unusable

        return {
            "total_entries": total,
            "fresh_entries": fresh,
            "expired_entries": expired,
            "stale_entries": stale,
            "usable_entries": fresh + expired,
            "ttl_hours": self.ttl_hours,
            "stale_ttl_hours": self.stale_ttl_hours,
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


# =============================================================================
# Persistent Cooldown Functions
# =============================================================================

def load_persistent_cooldown() -> Optional[datetime]:
    """
    Load the sentiment API cooldown timestamp from persistent storage.

    Returns:
        datetime when cooldown expires, or None if no cooldown active
    """
    if not os.path.exists(COOLDOWN_FILE):
        return None

    try:
        with open(COOLDOWN_FILE, 'r') as f:
            data = json.load(f)

        cooldown_until_str = data.get("cooldown_until")
        if not cooldown_until_str:
            return None

        cooldown_until = datetime.fromisoformat(cooldown_until_str)

        # Check if cooldown has expired
        if datetime.now() > cooldown_until:
            logger.info("Persistent cooldown has expired. Clearing cooldown file.")
            clear_persistent_cooldown()
            return None

        remaining = cooldown_until - datetime.now()
        logger.info(f"Persistent cooldown active until {cooldown_until_str} ({remaining.total_seconds()/3600:.1f}h remaining)")
        return cooldown_until

    except (json.JSONDecodeError, IOError, ValueError) as e:
        logger.warning(f"Failed to load persistent cooldown: {e}")
        return None


def save_persistent_cooldown(cooldown_until: datetime, reason: str = "rate_limit") -> None:
    """
    Save the sentiment API cooldown timestamp to persistent storage.

    Args:
        cooldown_until: datetime when cooldown should expire
        reason: Reason for the cooldown (e.g., "rate_limit", "429_error")
    """
    try:
        data = {
            "cooldown_until": cooldown_until.isoformat(),
            "set_at": datetime.now().isoformat(),
            "reason": reason,
        }
        with open(COOLDOWN_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Persistent cooldown saved until {cooldown_until.isoformat()} (reason: {reason})")
    except IOError as e:
        logger.error(f"Failed to save persistent cooldown: {e}")


def clear_persistent_cooldown() -> None:
    """Remove the persistent cooldown file."""
    try:
        if os.path.exists(COOLDOWN_FILE):
            os.remove(COOLDOWN_FILE)
            logger.info("Persistent cooldown cleared")
    except IOError as e:
        logger.warning(f"Failed to clear persistent cooldown: {e}")


def is_cooldown_active() -> bool:
    """
    Check if sentiment API cooldown is currently active.

    Returns:
        True if cooldown is active, False otherwise
    """
    cooldown_until = load_persistent_cooldown()
    return cooldown_until is not None and datetime.now() < cooldown_until


def get_cooldown_remaining_seconds() -> float:
    """
    Get the remaining cooldown time in seconds.

    Returns:
        Remaining seconds, or 0 if no cooldown active
    """
    cooldown_until = load_persistent_cooldown()
    if cooldown_until is None:
        return 0.0
    remaining = (cooldown_until - datetime.now()).total_seconds()
    return max(0.0, remaining)
