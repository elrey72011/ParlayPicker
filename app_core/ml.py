"""Machine learning helpers that blend API-Sports history with odds data."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ODDS_BASE_URL = "https://api.the-odds-api.com"


FEATURE_COLUMNS = [
    "home_avg_for",
    "home_avg_against",
    "home_form_pct",
    "home_trend_score",
    "home_record_pct",
    "away_avg_for",
    "away_avg_against",
    "away_form_pct",
    "away_trend_score",
    "away_record_pct",
    "sentiment_home",
    "sentiment_away",
    "sentiment_diff",
    "home_ml_implied",
    "away_ml_implied",
    "home_field_advantage",
]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_to_pct(record: Optional[str]) -> Optional[float]:
    if not record:
        return None
    parts = [p for p in str(record).replace(" ", "").split("-") if p]
    if not parts:
        return None
    try:
        wins = float(parts[0])
        losses = float(parts[1]) if len(parts) > 1 else 0.0
        draws = float(parts[2]) if len(parts) > 2 else 0.0
    except ValueError:
        return None
    total = wins + losses + draws
    if total <= 0:
        return None
    # Treat draws as half wins for win percentage
    return (wins + 0.5 * draws) / total


def _form_to_pct(form: Optional[str]) -> Optional[float]:
    if not form:
        return None
    sequence = [c for c in form.upper() if c in {"W", "L", "D"}]
    if not sequence:
        return None
    wins = sum(1 for c in sequence if c == "W")
    draws = sum(1 for c in sequence if c == "D")
    return (wins + 0.5 * draws) / len(sequence)


def _trend_to_score(trend: Optional[str]) -> Optional[float]:
    if not trend:
        return None
    trend_lower = trend.lower()
    if trend_lower == "hot":
        return 1.0
    if trend_lower == "cold":
        return -1.0
    if trend_lower == "neutral":
        return 0.0
    return None


def _implied_prob(odds: Optional[float]) -> Optional[float]:
    price = _safe_float(odds)
    if price is None or abs(price) < 100:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def _normalize_team_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return "".join(ch for ch in name.upper() if ch.isalnum())


def _extract_score(entry: Any) -> Optional[float]:
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for key in ("total", "score", "points"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _extract_commence_iso(game: Dict[str, Any]) -> Optional[str]:
    for key in ("game", "date"):
        container = game.get(key)
        if isinstance(container, dict):
            for candidate in ("utc", "start", "date"):
                value = container.get(candidate)
                if isinstance(value, str):
                    return value
        elif isinstance(container, str):
            return container
    return None


def _kickoff_date(iso_ts: Optional[str]) -> Optional[date]:
    if not iso_ts:
        return None
    try:
        return datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).date()
    except ValueError:
        return None


class HistoricalDataBuilder:
    """Build datasets that join API-Sports summaries with historical odds."""

    def __init__(
        self,
        odds_key_getter: Callable[[], Optional[str]],
        *,
        timezone: str = "America/New_York",
        days_back: int = 45,
    ) -> None:
        self._odds_key_getter = odds_key_getter
        self.timezone = timezone
        self.days_back = max(7, days_back)
        self._clients: Dict[str, Any] = {}
        self._dataset_cache: Dict[str, pd.DataFrame] = {}
        self._dataset_timestamp: Dict[str, datetime] = {}
        self._dataset_attempt_timestamp: Dict[str, datetime] = {}
        self._dataset_errors: Dict[str, Optional[str]] = {}
        self._odds_cache: Dict[str, Tuple[datetime, Dict[Tuple[str, str, date], Dict[str, Optional[float]]]]] = {}
        self._http = requests.Session()

    def register_client(self, sport_key: str, client: Any) -> None:
        if sport_key and client:
            self._clients[sport_key] = client

    # ------------------------------------------------------------------
    def get_dataset(self, sport_key: Optional[str], rebuild: bool = False) -> pd.DataFrame:
        if not sport_key:
            return pd.DataFrame()

        if (not rebuild) and sport_key in self._dataset_cache:
            ts = self._dataset_timestamp.get(sport_key)
            if ts and (datetime.utcnow() - ts) < timedelta(hours=6):
                return self._dataset_cache[sport_key]

        dataset = self._build_dataset(sport_key)
        self._dataset_attempt_timestamp[sport_key] = datetime.utcnow()
        if dataset.empty:
            # Avoid caching empty results so a newly provided API key can
            # trigger another fetch instead of being stuck with a cold cache.
            self._dataset_cache.pop(sport_key, None)
            self._dataset_timestamp.pop(sport_key, None)
            return dataset

        self._dataset_cache[sport_key] = dataset
        self._dataset_timestamp[sport_key] = datetime.utcnow()
        self._dataset_errors[sport_key] = None
        return dataset

    # ------------------------------------------------------------------
    def dataset_info(self, sport_key: Optional[str]) -> Dict[str, Optional[Any]]:
        """Return cached dataset metadata without forcing a rebuild."""

        info: Dict[str, Optional[Any]] = {
            "sport_key": sport_key,
            "rows": 0,
            "last_built": None,
            "last_attempt": None,
            "error": None,
        }

        if not sport_key:
            return info

        dataset = self._dataset_cache.get(sport_key)
        if dataset is not None:
            info["rows"] = int(len(dataset))
            info["last_built"] = self._dataset_timestamp.get(sport_key)
        else:
            info["rows"] = 0

        info["last_attempt"] = self._dataset_attempt_timestamp.get(sport_key)
        info["error"] = self._dataset_errors.get(sport_key)
        return info

    # ------------------------------------------------------------------
    def _build_dataset(self, sport_key: str) -> pd.DataFrame:
        client = self._clients.get(sport_key)
        if not client:
            self._dataset_errors[sport_key] = "unregistered_client"
            return pd.DataFrame(columns=FEATURE_COLUMNS + ["home_win"])

        if not getattr(client, "is_configured", lambda: False)():
            self._dataset_errors[sport_key] = "missing_api_key"
            return pd.DataFrame(columns=FEATURE_COLUMNS + ["home_win"])

        today = datetime.utcnow().date()
        rows = []
        odds_map = self._get_historical_odds_map(sport_key)

        for offset in range(1, self.days_back + 1):
            target_date = today - timedelta(days=offset)
            try:
                games = client.get_games_by_date(target_date, timezone=self.timezone)
            except Exception:
                self._dataset_errors[sport_key] = getattr(client, "last_error", "games_fetch_failed")
                continue

            for game in games or []:
                home_score = _extract_score((game.get("scores") or {}).get("home"))
                away_score = _extract_score((game.get("scores") or {}).get("away"))
                if home_score is None or away_score is None:
                    continue

                try:
                    summary = client.build_game_summary(game, tz_name=self.timezone)
                except Exception:
                    self._dataset_errors[sport_key] = getattr(client, "last_error", "summary_build_failed")
                    continue

                feature_row = self._summary_to_features(summary)
                commence_iso = _extract_commence_iso(game)
                odds_entry = self._lookup_odds_entry(summary, commence_iso, odds_map)
                if odds_entry:
                    feature_row.update(odds_entry)

                feature_row.setdefault("home_ml_implied", None)
                feature_row.setdefault("away_ml_implied", None)
                feature_row.setdefault("sentiment_home", 0.0)
                feature_row.setdefault("sentiment_away", 0.0)
                feature_row.setdefault("sentiment_diff", 0.0)
                feature_row.setdefault("home_field_advantage", 1.0)

                feature_row["home_win"] = 1 if home_score > away_score else 0
                rows.append(feature_row)

        if not rows:
            if sport_key not in self._dataset_errors:
                self._dataset_errors[sport_key] = getattr(client, "last_error", "no_historical_rows")
            return pd.DataFrame(columns=FEATURE_COLUMNS + ["home_win"])

        self._dataset_errors[sport_key] = None
        df = pd.DataFrame(rows)
        df = df.drop_duplicates()
        ordered_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        if "home_win" in df.columns:
            ordered_cols.append("home_win")
        return df[ordered_cols]

    # ------------------------------------------------------------------
    def _summary_to_features(self, summary: Any) -> Dict[str, Optional[float]]:
        features: Dict[str, Optional[float]] = {col: None for col in FEATURE_COLUMNS}

        home = getattr(summary, "home", None)
        away = getattr(summary, "away", None)

        features["home_avg_for"] = _safe_float(getattr(home, "average_points_for", None))
        features["home_avg_against"] = _safe_float(getattr(home, "average_points_against", None))
        features["home_form_pct"] = _form_to_pct(getattr(home, "form", None))
        features["home_trend_score"] = _trend_to_score(getattr(home, "trend", None))
        features["home_record_pct"] = _record_to_pct(getattr(home, "record", None))

        features["away_avg_for"] = _safe_float(getattr(away, "average_points_for", None))
        features["away_avg_against"] = _safe_float(getattr(away, "average_points_against", None))
        features["away_form_pct"] = _form_to_pct(getattr(away, "form", None))
        features["away_trend_score"] = _trend_to_score(getattr(away, "trend", None))
        features["away_record_pct"] = _record_to_pct(getattr(away, "record", None))

        # Leave odds/sentiment fields for later enrichment
        return features

    # ------------------------------------------------------------------
    def _get_historical_odds_map(
        self, sport_key: str
    ) -> Dict[Tuple[str, str, date], Dict[str, Optional[float]]]:
        cached = self._odds_cache.get(sport_key)
        if cached and (datetime.utcnow() - cached[0]) < timedelta(hours=6):
            return cached[1]

        odds_key = self._odds_key_getter() if self._odds_key_getter else None
        if not odds_key:
            # Don't pin an empty cache when credentials are missing; retry once
            # the user supplies a key.
            self._odds_cache.pop(sport_key, None)
            return {}

        params = {
            "apiKey": odds_key,
            "daysFrom": str(self.days_back),
            "dateFormat": "iso",
            "includeOdds": "1",
        }

        mapping: Dict[Tuple[str, str, date], Dict[str, Optional[float]]] = {}
        try:
            resp = self._http.get(
                f"{ODDS_BASE_URL}/v4/sports/{sport_key}/scores",
                params=params,
                timeout=12,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException:
            payload = []
        except ValueError:
            payload = []

        for event in payload or []:
            home = event.get("home_team")
            away = event.get("away_team")
            commence = event.get("commence_time")
            kickoff = _kickoff_date(commence)
            home_norm = _normalize_team_name(home)
            away_norm = _normalize_team_name(away)
            if not (home_norm and away_norm and kickoff):
                continue

            best_home = None
            best_away = None
            for bookmaker in event.get("bookmakers") or []:
                for market in bookmaker.get("markets") or []:
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes") or []:
                        name = outcome.get("name")
                        price = outcome.get("price")
                        if name == home:
                            best_home = price
                        elif name == away:
                            best_away = price
            mapping[(home_norm, away_norm, kickoff)] = {
                "home_ml": _safe_float(best_home),
                "away_ml": _safe_float(best_away),
                "home_ml_implied": _implied_prob(best_home),
                "away_ml_implied": _implied_prob(best_away),
            }

        if not mapping:
            # Allow quick retries when the API call fails or returns nothing.
            self._odds_cache.pop(sport_key, None)
            return mapping

        self._odds_cache[sport_key] = (datetime.utcnow(), mapping)
        return mapping

    # ------------------------------------------------------------------
    def _lookup_odds_entry(
        self,
        summary: Any,
        commence_iso: Optional[str],
        odds_map: Dict[Tuple[str, str, date], Dict[str, Optional[float]]],
    ) -> Optional[Dict[str, Optional[float]]]:
        if not odds_map:
            return None

        home_name = getattr(getattr(summary, "home", None), "name", None)
        away_name = getattr(getattr(summary, "away", None), "name", None)
        home_norm = _normalize_team_name(home_name)
        away_norm = _normalize_team_name(away_name)
        kickoff_date = _kickoff_date(commence_iso)
        if not (home_norm and away_norm and kickoff_date):
            return None

        return odds_map.get((home_norm, away_norm, kickoff_date))

    # ------------------------------------------------------------------
    def build_feature_vector(
        self,
        *,
        sport_key: Optional[str],
        home_payload: Optional[Dict[str, Any]],
        away_payload: Optional[Dict[str, Any]],
        home_odds: Optional[float],
        away_odds: Optional[float],
        sentiment_home: Optional[float],
        sentiment_away: Optional[float],
    ) -> Dict[str, Optional[float]]:
        features: Dict[str, Optional[float]] = {col: None for col in FEATURE_COLUMNS}

        def _populate(prefix: str, payload: Optional[Dict[str, Any]]) -> None:
            if not payload:
                return
            if prefix == "home":
                features["home_avg_for"] = _safe_float(payload.get("team_avg_points_for"))
                features["home_avg_against"] = _safe_float(payload.get("team_avg_points_against"))
                features["home_form_pct"] = _form_to_pct(payload.get("team_form"))
                features["home_trend_score"] = _trend_to_score(payload.get("trend"))
                features["home_record_pct"] = _record_to_pct(payload.get("team_record"))
            else:
                features["away_avg_for"] = _safe_float(payload.get("team_avg_points_for"))
                features["away_avg_against"] = _safe_float(payload.get("team_avg_points_against"))
                features["away_form_pct"] = _form_to_pct(payload.get("team_form"))
                features["away_trend_score"] = _trend_to_score(payload.get("trend"))
                features["away_record_pct"] = _record_to_pct(payload.get("team_record"))

        _populate("home", home_payload)
        _populate("away", away_payload)

        sentiment_home = _safe_float(sentiment_home) or 0.0
        sentiment_away = _safe_float(sentiment_away) or 0.0
        features["sentiment_home"] = sentiment_home
        features["sentiment_away"] = sentiment_away
        features["sentiment_diff"] = sentiment_home - sentiment_away

        features["home_ml_implied"] = _implied_prob(home_odds)
        features["away_ml_implied"] = _implied_prob(away_odds)
        features["home_field_advantage"] = 1.0

        return features


class HistoricalMLPredictor:
    """Train logistic models on the fly using historical datasets."""

    def __init__(self, data_builder: HistoricalDataBuilder) -> None:
        self.data_builder = data_builder
        self._models: Dict[str, Pipeline] = {}
        self._model_timestamp: Dict[str, datetime] = {}
        self._training_rows: Dict[str, int] = {}
        self.min_rows = 25

    def register_client(self, sport_key: str, client: Any) -> None:
        self.data_builder.register_client(sport_key, client)

    # ------------------------------------------------------------------
    def training_metadata(self, sport_key: Optional[str]) -> Dict[str, Optional[Any]]:
        """Return the current training status for the given sport."""

        info: Dict[str, Optional[Any]] = {
            "sport_key": sport_key,
            "dataset_rows": 0,
            "last_dataset_build": None,
            "model_ready": False,
            "last_trained": None,
            "training_rows": 0,
            "min_rows": self.min_rows,
            "error": None,
        }

        if not sport_key:
            return info

        dataset = self.data_builder.get_dataset(sport_key)
        info["dataset_rows"] = int(len(dataset))
        dataset_info = self.data_builder.dataset_info(sport_key)
        info["last_dataset_build"] = dataset_info.get("last_built")
        info["error"] = dataset_info.get("error")

        if dataset.empty or len(dataset) < self.min_rows or dataset["home_win"].nunique() < 2:
            info["training_rows"] = int(len(dataset))
            info["model_ready"] = False
            info["last_trained"] = self._model_timestamp.get(sport_key)
            return info

        model = self._ensure_model(sport_key)
        info["model_ready"] = model is not None and sport_key in self._models
        info["last_trained"] = self._model_timestamp.get(sport_key)
        info["training_rows"] = int(self._training_rows.get(sport_key, len(dataset)))
        return info

    # ------------------------------------------------------------------
    def _ensure_model(self, sport_key: Optional[str]) -> Optional[Pipeline]:
        if not sport_key:
            return None

        dataset = self.data_builder.get_dataset(sport_key)
        if dataset.empty or dataset["home_win"].nunique() < 2 or len(dataset) < self.min_rows:
            return None

        needs_retrain = False
        if sport_key not in self._models:
            needs_retrain = True
        else:
            ts = self._model_timestamp.get(sport_key)
            if not ts or (datetime.utcnow() - ts) > timedelta(hours=6):
                needs_retrain = True
            elif self._training_rows.get(sport_key) != len(dataset):
                needs_retrain = True

        if not needs_retrain:
            return self._models[sport_key]

        X = dataset.reindex(columns=FEATURE_COLUMNS).to_numpy(dtype=float)
        y = dataset["home_win"].to_numpy(dtype=int)

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)

        self._models[sport_key] = pipeline
        self._model_timestamp[sport_key] = datetime.utcnow()
        self._training_rows[sport_key] = len(dataset)
        return pipeline

    # ------------------------------------------------------------------
    def predict_game_outcome(
        self,
        home_team: str,
        away_team: str,
        home_odds: Optional[float],
        away_odds: Optional[float],
        sentiment_home: Optional[float],
        sentiment_away: Optional[float],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        context = context or {}
        sport_key = context.get("sport_key")

        home_payload = context.get("apisports_home") if isinstance(context.get("apisports_home"), dict) else context.get("apisports_home")
        away_payload = context.get("apisports_away") if isinstance(context.get("apisports_away"), dict) else context.get("apisports_away")

        feature_vector = self.data_builder.build_feature_vector(
            sport_key=sport_key,
            home_payload=home_payload if isinstance(home_payload, dict) else None,
            away_payload=away_payload if isinstance(away_payload, dict) else None,
            home_odds=home_odds,
            away_odds=away_odds,
            sentiment_home=sentiment_home,
            sentiment_away=sentiment_away,
        )

        model = self._ensure_model(sport_key)

        market_home = feature_vector.get("home_ml_implied")
        market_away = feature_vector.get("away_ml_implied")
        if market_home is None and market_away is not None:
            market_home = 1.0 - market_away
        if market_home is None:
            market_home = _implied_prob(home_odds) or 0.5
        if market_away is None:
            market_away = 1.0 - market_home

        sentiment_diff = feature_vector.get("sentiment_diff") or 0.0

        if model is not None:
            try:
                probs = model.predict_proba(pd.DataFrame([feature_vector], columns=FEATURE_COLUMNS))[0]
                home_prob_model = float(probs[1])
            except Exception:
                model = None
            else:
                sentiment_adjustment = float(np.tanh(sentiment_diff) * 0.08)
                blended_home = (
                    home_prob_model * 0.65
                    + market_home * 0.25
                    + (0.5 + sentiment_adjustment) * 0.10
                )
                home_prob = float(np.clip(blended_home, 0.02, 0.98))
                away_prob = 1.0 - home_prob
                agreement = 1.0 - abs(home_prob_model - market_home)
                confidence = float(np.clip(0.55 + 0.35 * agreement + 0.1 * (1 - abs(sentiment_adjustment) / 0.08), 0.45, 0.97))
                edge = abs(home_prob - market_home)
                return {
                    "home_prob": home_prob,
                    "away_prob": away_prob,
                    "confidence": confidence,
                    "edge": edge,
                    "recommendation": "home" if home_prob >= away_prob else "away",
                    "model_used": "historical-logistic",
                    "training_rows": float(self._training_rows.get(sport_key, 0)),
                }

        # Fallback heuristic when no model is available
        sentiment_adjustment = float(np.tanh(sentiment_diff) * 0.06)
        base_home = market_home
        home_prob = float(np.clip(base_home + sentiment_adjustment, 0.05, 0.95))
        away_prob = 1.0 - home_prob
        edge = abs(home_prob - market_home)
        confidence = float(np.clip(0.45 + 0.1 * (1 - abs(sentiment_adjustment) / 0.06), 0.4, 0.7))

        return {
            "home_prob": home_prob,
            "away_prob": away_prob,
            "confidence": confidence,
            "edge": edge,
            "recommendation": "home" if home_prob >= away_prob else "away",
            "model_used": "heuristic",
            "training_rows": float(self._training_rows.get(sport_key, 0)),
        }


# Backwards-compatible alias for the legacy import path
MLPredictor = HistoricalMLPredictor

