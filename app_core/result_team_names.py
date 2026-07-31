"""Team-name normalization that does not require the prediction/ML runtime."""

from __future__ import annotations

from functools import lru_cache
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def _ascii_words(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _compact(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _ascii_words(value))


@lru_cache(maxsize=1)
def _production_normalizers():
    try:
        from app_core.feature_processing import robust_normalize_team
        from app_core.prediction_engine import clean_team_name

        return robust_normalize_team, clean_team_name
    except Exception as exc:
        logger.warning(
            "Prediction-stack team normalization is unavailable; using grading-only fallback: %s",
            exc,
        )
        return _ascii_words, _compact


def normalize_result_team(value: object) -> str:
    normalize, _ = _production_normalizers()
    return str(normalize(str(value or "")))


def clean_result_team(value: object) -> str:
    _, clean = _production_normalizers()
    return str(clean(str(value or "")))
