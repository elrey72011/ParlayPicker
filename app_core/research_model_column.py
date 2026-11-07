"""The research models' probability for each board candidate, as a research column (owner 2026-11-07).

The app's own ML model has no spread or total model for NBA, NCAAB, NCAAF or NHL ("No
market-specific model configured"), so every win estimate on the board is the market's
no-vig price plus Kalshi. The research loop does fit score-distribution models for NBA,
NCAAB and NHL spreads/puck lines and totals (``app_core.prospective_research_models``) and
freezes a pregame prediction for each upcoming game in the canonical evidence store.

This shows that model's view next to the market, for comparison only:

* ``research_model_probability``: the model's win probability for the candidate's own side
  at its own line (same normal score distribution, ``score_probabilities``), from the
  latest frozen prediction for that game: mean score from the prediction's frozen
  features and model artifact, the artifact's residual SD.
* ``research_model_push_probability`` and ``research_model_ev`` (EV of the candidate's own
  price under that win/push/loss split).
* ``research_model_predicted_at``: when the prediction was frozen.

It changes no probability, pick, EV, gate or stake. The models are unvalidated: their
graded record is what the research loop and the Pinnacle calibration measure.

The app reads the canonical store the qualified bridge already restores read-only from the
evidence remote (``app_core.qualified_bridge.refresh_store``: throttled, background, never
writes). That restore otherwise runs only when a market is activated; this column starts it
in the background too. The first Refresh after an app restart may show the column empty
until the restore finishes. ``PARLAYPICKER_RESEARCH_MODEL_COLUMN=false`` turns it all off.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
import re

import numpy as np
import pandas as pd

ENV = "PARLAYPICKER_RESEARCH_MODEL_COLUMN"
COLUMNS = ("research_model_probability", "research_model_push_probability", "research_model_ev",
           "research_model_predicted_at")
FAMILIES = {"NBA": {"spread": "SPREAD", "total": "TOTAL"}, "NCAAB": {"spread": "SPREAD", "total": "TOTAL"},
            "NHL": {"spread": "PUCK_LINE", "total": "TOTAL"}}
START_TOLERANCE = timedelta(hours=3)
LOOKBACK = timedelta(hours=12)
_CACHE: dict = {}


def enabled(setting=os.environ.get) -> bool:
    return str(setting(ENV, "true") or "true").strip().lower() != "false"


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def _name(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _same_team(board, stored) -> bool:
    """Board names can be short ("Oklahoma City") where the provider is full ("Oklahoma City Thunder")."""
    board, stored = _name(board), _name(stored)
    return bool(board and stored) and (board == stored or (len(board) >= 4 and stored.startswith(board))
                                       or (len(stored) >= 4 and board.startswith(stored)))


def start_store_refresh(setting=os.environ.get):
    """Start the bridge's read-only store restore in the background (one per process)."""
    import threading
    from app_core import qualified_bridge as bridge
    if not str(setting("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "") or "").strip():
        return {"status": "MISSING_CREDENTIALS"}  # no remote to read; no thread
    with bridge._BACKGROUND_LOCK:
        thread = bridge._BACKGROUND["thread"]
        if thread is not None and thread.is_alive():
            return {"status": "IN_PROGRESS_BACKGROUND", "last": bridge._BACKGROUND["last"]}

        def run():
            bridge._BACKGROUND["last"] = bridge.refresh_store(setting=setting)  # never raises
        thread = threading.Thread(target=run, name="research-model-store-refresh", daemon=True)
        bridge._BACKGROUND["thread"] = thread
        thread.start()
    return {"status": "STARTED_IN_BACKGROUND", "last": bridge._BACKGROUND["last"]}


def load(path=None, *, now=None) -> list[dict]:
    """Latest frozen prediction per (sport, family, game) for games starting after now - 12h.

    Each entry: sport, family, home_team, away_team, start, mean, sd, predicted_at.
    """
    from app_core import prospective_evidence as evidence
    from app_core import prospective_research_models as models
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in FAMILIES.items():
        for family in set(families.values()):
            predictions = evidence.read_records(path, "prospective_prediction", sport=sport, market_family=family)
            if not predictions:
                continue
            artifacts = {}
            for model in evidence.read_records(path, "prospective_model", sport=sport, market_family=family):
                artifact = json.loads(model["payload"]).get("model_artifact")
                if isinstance(artifact, dict) and models._digest(artifact) == model["artifact_hash"]:
                    artifacts[model["model_id"]] = artifact
            latest = {}
            for row in predictions:
                if row.get("model_id") not in artifacts:
                    continue
                key = row["event_id"]
                if key not in latest or row["prediction_timestamp"] > latest[key]["prediction_timestamp"]:
                    latest[key] = row
            for event_id, row in latest.items():
                event = evidence.load_record(path, "prospective_event", event_id)
                start = _time(event["scheduled_start"]) if event else None
                if start is None or start < now - LOOKBACK:
                    continue
                features = (json.loads(row["payload"]).get("feature_snapshot") or {}).get("features")
                fit = artifacts[row["model_id"]].get("model") or {}
                try:
                    mean = models._expected(fit, features)
                    sd = float(fit["residual_sd"])
                except (KeyError, TypeError, IndexError, ValueError):
                    continue
                if not (math.isfinite(mean) and math.isfinite(sd) and sd > 0):
                    continue
                out.append({"sport": sport, "family": family, "home_team": event["home_team"],
                            "away_team": event["away_team"], "start": start, "mean": mean, "sd": sd,
                            "predicted_at": row["prediction_timestamp"]})
    return out


def _side(row):
    """(family kind, canonical line, selection) for a board candidate, or None.

    Spreads: the score is the home margin and the canonical side is the home team at its own
    line, so an away pick at L is the opposite side of home at -L. Totals: canonical Over.
    """
    market = str(row.get("market_type") or "").lower()
    pick = str(row.get("best_pick") or "")
    if market.startswith("spread"):
        found = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", pick)
        if not found:
            return None
        line = float(found.group(1))
        if market == "spread_home":
            return "spread", line, "canonical"
        if market == "spread_away":
            return "spread", -line, "opposite"
        return None
    if market.startswith("total"):
        found = re.search(r"(\d+(?:\.\d+)?)\s*$", pick)
        if not found:
            return None
        return "total", float(found.group(1)), ("canonical" if market == "total_over" else "opposite")
    return None


def _decimal(odds):
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds >= 100:
        return 1 + odds / 100.0
    if odds <= -100:
        return 1 + 100.0 / abs(odds)
    return None


def attach(frame: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    """``frame`` with the research columns; rows without a unique matching prediction stay empty."""
    from app_core.prospective_research_models import score_probabilities
    out = frame.copy()
    for column in COLUMNS:
        out[column] = np.nan if column != "research_model_predicted_at" else None
    if out.empty or not predictions:
        return out
    for index, row in out.iterrows():
        sport = str(row.get("league") or row.get("sport") or "").upper()
        side = _side(row)
        if sport not in FAMILIES or side is None:
            continue
        kind, line, selection = side
        family = FAMILIES[sport][kind]
        start = _time(row.get("game_start_utc") or row.get("commence_time") or "")
        found = [p for p in predictions
                 if p["sport"] == sport and p["family"] == family
                 and _same_team(row.get("home_team"), p["home_team"])
                 and _same_team(row.get("away_team"), p["away_team"])
                 and (start is None or abs(p["start"] - start) <= START_TOLERANCE)]
        if len(found) != 1:
            continue
        p = found[0]
        try:
            split = score_probabilities(p["mean"], p["sd"], family, line, selection=selection)
        except ValueError:
            continue
        out.at[index, "research_model_probability"] = split["win"]
        out.at[index, "research_model_push_probability"] = split["push"]
        price = _decimal(row.get("odds_american"))
        if price is not None:
            out.at[index, "research_model_ev"] = split["win"] * (price - 1) - split["loss"]
        out.at[index, "research_model_predicted_at"] = p["predicted_at"]
    return out


def apply(frame: pd.DataFrame, *, setting=os.environ.get, path=None, now=None, refresh=None, read=None):
    """(frame with the research columns, diagnostic). Never raises; changes no other column."""
    if not enabled(setting):
        out = frame.copy()
        for column in COLUMNS:
            out[column] = np.nan if column != "research_model_predicted_at" else None
        return out, {"status": "OFF"}
    diagnostic = {}
    try:
        diagnostic["store"] = (refresh or start_store_refresh)(setting)
    except Exception as exc:  # never break Refresh
        diagnostic["store"] = {"status": "ERROR", "error": type(exc).__name__}
    try:
        predictions = (read or _cached_load)(path, now=now)
    except Exception as exc:  # an absent or unreadable store leaves the column empty
        predictions, diagnostic["error"] = [], type(exc).__name__
    out = attach(frame, predictions)
    filled = int(pd.to_numeric(out["research_model_probability"], errors="coerce").notna().sum())
    diagnostic.update(status="ATTACHED" if filled else "NO_MATCHING_PREDICTION",
                      predictions=len(predictions), candidates_with_probability=filled)
    return out, diagnostic


def _cached_load(path=None, *, now=None):
    """``load`` once per store version (file size and mtime) and hour."""
    from pathlib import Path
    from app_core import prospective_evidence as evidence
    target = Path(evidence._path(path))
    stat = target.stat()
    hour = (now or datetime.now(timezone.utc)).strftime("%Y%m%d%H")
    key = (str(target), stat.st_size, stat.st_mtime_ns, hour)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = load(path, now=now)
    return _CACHE[key]
