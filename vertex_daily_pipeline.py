"""
Vertex AI daily pipeline utilities.

This module provides two entry points for running a simple daily batch job on
Vertex AI:

- ``train_daily_model``: trains or refreshes the model using the aggregated
  training CSV, then uploads the serialized estimator to GCS.
- ``score_daily_slates``: loads the latest model from GCS, scores today's
  games, blends Kalshi sentiment into the ranking, and saves a CSV that matches
  the ParlayPicker app schema (including Game Time + Kalshi columns).

Both functions are intentionally lightweight and rely on existing feature
builders (CSV exports or API-backed scripts) that already exist in the
codebase/notebooks. The pipeline simply stitches those pieces together for a
Vertex AI scheduled run.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import GradientBoostingClassifier

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _parse_gcs_uri(gcs_path: str) -> Tuple[str, str]:
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    path = gcs_path.replace("gs://", "", 1)
    bucket, *parts = path.split("/", 1)
    if not bucket:
        raise ValueError(f"Missing bucket in GCS path: {gcs_path}")
    blob_name = parts[0] if parts else ""
    return bucket, blob_name


def _upload_bytes_to_gcs(data: bytes, gcs_path: str) -> None:
    bucket_name, blob_name = _parse_gcs_uri(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)


def _upload_file_to_gcs(local_path: str, gcs_path: str) -> None:
    bucket_name, blob_name = _parse_gcs_uri(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def _download_file_from_gcs(gcs_path: str, dest_path: str) -> None:
    bucket_name, blob_name = _parse_gcs_uri(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dest_path)


def _format_game_time_local(dt_utc: str, user_tz: str = "US/Eastern") -> str:
    import pytz

    if not dt_utc:
        return "TBD"
    try:
        dt = pd.to_datetime(dt_utc, utc=True)
        tz = pytz.timezone(user_tz or "US/Eastern")
        dt_local = dt.tz_convert(tz)
        return dt_local.strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception:
        return "TBD"


def _pick_best_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _infer_target_column(df: pd.DataFrame) -> str:
    candidates = ["label", "target", "home_win", "win_label", "y"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Training data must include one of: label, target, home_win, win_label, y")


def _infer_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    env_features = os.getenv("ACTIVE_FEATURE_NAMES")
    if env_features:
        names = [c.strip() for c in env_features.split(",") if c.strip()]
        missing = [c for c in names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return names

    excluded = {target_col}
    excluded.update({"league", "game", "home_team", "away_team", "commence_time", "game_time"})
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found for training")
    return numeric_cols


def _ensure_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = 0.0
    return df[feature_cols].fillna(0.0)


def _load_today_features(run_date: str) -> pd.DataFrame:
    """Load today's feature table.

    Preference order:
    1) Environment variable TODAY_FEATURE_CSV pointing to a local path
    2) Local file named today_features_{run_date}.csv

    Users can swap this loader with their API-backed builder before running in
    production; the pipeline keeps the interface simple for scheduling.
    """

    feature_path = os.getenv("TODAY_FEATURE_CSV")
    if feature_path and os.path.exists(feature_path):
        return pd.read_csv(feature_path)

    fallback_name = f"today_features_{run_date}.csv"
    if os.path.exists(fallback_name):
        return pd.read_csv(fallback_name)

    raise FileNotFoundError(
        "No daily feature table found. Provide TODAY_FEATURE_CSV or create "
        f"{fallback_name} before running the scorer."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_daily_model(train_csv_path: str, model_gcs_path: str) -> None:
    """
    Load the aggregated training data CSV, train the sports model, and upload
    the pickled model to GCS (model_gcs_path).
    """

    df = pd.read_csv(train_csv_path)
    target_col = _infer_target_column(df)
    feature_cols = _infer_feature_columns(df, target_col)

    X = _ensure_feature_matrix(df.copy(), feature_cols)
    y = df[target_col].values

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_names": feature_cols,
        "target": target_col,
        "trained_at": datetime.utcnow().isoformat(),
        "training_rows": len(df),
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(artifact, tmp)
        tmp.flush()
        _upload_file_to_gcs(tmp.name, model_gcs_path)

    metadata = {
        "model_gcs_path": model_gcs_path,
        "trained_at": artifact["trained_at"],
        "feature_names": feature_cols,
        "target": target_col,
    }
    _upload_bytes_to_gcs(json.dumps(metadata, indent=2).encode("utf-8"), f"{model_gcs_path}.metadata.json")


def score_daily_slates(
    model_gcs_path: str,
    pred_gcs_path: str,
    run_date: Optional[str] = None,
) -> None:
    """
    Load the model from GCS, build features for all games on run_date (default:
    today), compute predictions + blended_score, and write a CSV of best picks
    to pred_gcs_path in GCS.
    """

    run_date = run_date or datetime.utcnow().strftime("%Y%m%d")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        _download_file_from_gcs(model_gcs_path, tmp.name)
        tmp.flush()
        artifact = pickle.load(open(tmp.name, "rb"))

    model: GradientBoostingClassifier = artifact["model"]
    feature_cols: List[str] = artifact.get("feature_names") or []

    daily_df = _load_today_features(run_date)

    # Pick key columns for display
    home_col = _pick_best_column(daily_df, ["home_team", "HomeTeam", "home"])
    away_col = _pick_best_column(daily_df, ["away_team", "AwayTeam", "away"])
    league_col = _pick_best_column(daily_df, ["league", "League", "sport_key"])
    commence_col = _pick_best_column(daily_df, ["commence_time", "game_time", "start_time"])
    kalshi_col = _pick_best_column(daily_df, ["kalshi_prob", "kalshi_probability", "kalshi_pct", "Kalshi %"])

    X_daily = _ensure_feature_matrix(daily_df.copy(), feature_cols)
    probs = model.predict_proba(X_daily)[:, 1]

    user_tz = os.getenv("USER_TIMEZONE", "US/Eastern")

    results: List[Dict[str, object]] = []
    for idx, prob in enumerate(probs):
        row = daily_df.iloc[idx]
        home_team = row.get(home_col, "HOME") if home_col else "HOME"
        away_team = row.get(away_col, "AWAY") if away_col else "AWAY"
        league_val = row.get(league_col, "Unknown") if league_col else "Unknown"
        commence_raw = row.get(commence_col, "") if commence_col else ""

        kalshi_prob_raw = None
        if kalshi_col:
            kalshi_prob_raw = row.get(kalshi_col)
            if isinstance(kalshi_prob_raw, str) and kalshi_prob_raw.endswith("%"):
                try:
                    kalshi_prob_raw = float(kalshi_prob_raw.strip("%")) / 100.0
                except Exception:
                    kalshi_prob_raw = None

        model_prob = float(prob)
        kalshi_prob = None
        if kalshi_prob_raw is not None and not pd.isna(kalshi_prob_raw):
            try:
                kalshi_prob = float(kalshi_prob_raw)
                if kalshi_prob > 1:
                    kalshi_prob = kalshi_prob / 100.0
            except Exception:
                kalshi_prob = None

        kalshi_edge = None if kalshi_prob is None else model_prob - kalshi_prob
        blended_score = model_prob if kalshi_prob is None else 0.5 * model_prob + 0.5 * (kalshi_edge or 0)

        the_pick = home_team if model_prob >= 0.5 else away_team
        game_label = f"{away_team} @ {home_team}"
        game_time_fmt = _format_game_time_local(commence_raw, user_tz)

        results.append(
            {
                "league": league_val,
                "game": game_label,
                "Game Time": game_time_fmt,
                "the_pick": the_pick,
                "Win %": round(model_prob * 100, 2),
                "Favorite": row.get("favorite", ""),
                "Novig": row.get("novig", ""),
                "TheOver": row.get("theover_pick", row.get("TheOver", "")),
                "Sentiment": row.get("sentiment", ""),
                "Kalshi": row.get("kalshi_signal", ""),
                "Kalshi %": round(kalshi_prob * 100, 2) if kalshi_prob is not None else "",
                "EV": row.get("ev", ""),
                "kalshi_prob": kalshi_prob,
                "kalshi_edge": kalshi_edge if kalshi_edge is not None else 0.0,
                "blended_score": blended_score,
                "commence_time": commence_raw,
            }
        )

    pred_df = pd.DataFrame(results)
    pred_df["Rank"] = range(1, len(pred_df) + 1)
    pred_df = pred_df.sort_values("blended_score", ascending=False)

    cols = [
        "Rank",
        "league",
        "game",
        "Game Time",
        "the_pick",
        "Win %",
        "Favorite",
        "Novig",
        "TheOver",
        "Sentiment",
        "Kalshi",
        "Kalshi %",
        "EV",
        "kalshi_prob",
        "kalshi_edge",
        "blended_score",
    ]
    export_df = pred_df[[c for c in cols if c in pred_df.columns]]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        export_df.to_csv(tmp_csv.name, index=False)
        tmp_csv.flush()
        _upload_file_to_gcs(tmp_csv.name, pred_gcs_path)

    manifest = {
        "predictions_gcs_path": pred_gcs_path,
        "generated_at": datetime.utcnow().isoformat(),
        "run_date": run_date,
        "rows": len(export_df),
    }
    _upload_bytes_to_gcs(json.dumps(manifest, indent=2).encode("utf-8"), f"{pred_gcs_path}.manifest.json")

