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

How the app gets the predictions (2026-11-07). The research cycle publishes a small board to
the evidence Drive folder after each run (``publish``, from ``scripts/run_research_scheduler.py``):
one write-once JSON object per UTC hour, ``parlaypicker/research-model-board-v1/<hour>.json``,
holding the latest prediction's mean score and SD per upcoming game. The app fetches the newest
one by name (``fetch_board``, at most the last 6 hours, in a background thread, at most every
10 minutes) and computes each candidate's own side and line from it.

(The first version restored the whole canonical store into the app. A full restore reads every
remote object into memory, about 89,000, which the app never finished: on 2026-11-07 every
Refresh found the restore restarted and nothing recorded.)

``PARLAYPICKER_RESEARCH_MODEL_COLUMN=false`` turns it all off.
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
           "research_model_predicted_at", "research_model_status")
FAMILIES = {"NBA": {"spread": "SPREAD", "total": "TOTAL"}, "NCAAB": {"spread": "SPREAD", "total": "TOTAL"},
            "NHL": {"spread": "PUCK_LINE", "total": "TOTAL"}}
START_TOLERANCE = timedelta(hours=3)
LOOKBACK = timedelta(hours=12)


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


BOARD_PREFIX = "parlaypicker/research-model-board-v1/"
BOARD_LOOKBACK_HOURS = 6
BOARD_REFETCH_SECONDS = 600
_BOARD = {"thread": None, "last": None, "predictions": None, "fetched_at": None}


def board_key(when: datetime) -> str:
    return BOARD_PREFIX + when.astimezone(timezone.utc).strftime("%Y-%m-%dT%HZ") + ".json"


def _encode_board(predictions, now) -> bytes:
    rows = [dict(p, start=p["start"].isoformat()) for p in predictions]
    return json.dumps({"schema": "research-model-board-v1", "published_at": now.isoformat(),
                       "research_only": True, "predictions": rows},
                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode_board(raw) -> list[dict]:
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema") != "research-model-board-v1":
        raise ValueError("not a research model board")
    out = []
    for p in data.get("predictions") or []:
        start = _time(p.get("start"))
        if start is None:
            continue
        out.append(dict(p, start=start, mean=float(p["mean"]), sd=float(p["sd"])))
    return out


def publish(path, client, *, now=None) -> dict:
    """Write this hour's board from the canonical store, once. Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        predictions = load(path, now=now)
        if not predictions:
            return {"status": "NO_PREDICTIONS"}
        key = board_key(now)
        try:
            client.put_object(Key=key, Body=_encode_board(predictions, now), IfNoneMatch="*")
        except Exception as exc:
            if type(exc).__name__ == "AlreadyExists":
                return {"status": "ALREADY_PUBLISHED", "predictions": len(predictions)}
            raise
        return {"status": "PUBLISHED", "predictions": len(predictions)}
    except Exception as exc:  # research only: never break the research cycle
        return {"status": "ERROR", "error": type(exc).__name__}


def fetch_board(client, *, now=None):
    """(predictions, key) from the newest board in the last ``BOARD_LOOKBACK_HOURS`` hours, or ([], None)."""
    now = now or datetime.now(timezone.utc)
    for hours in range(BOARD_LOOKBACK_HOURS):
        key = board_key(now - timedelta(hours=hours))
        try:
            response = client.get_object(Key=key)
        except Exception as exc:
            if "missing" in str(exc).lower() or type(exc).__name__ in ("NoSuchKey", "KeyError", "FileNotFoundError"):
                continue
            raise
        with response["Body"] as body:
            return _decode_board(body.read()), key
    return [], None


def start_board_fetch(setting=os.environ.get, *, client_factory=None, clock=None):
    """Fetch the newest board in a background thread (one per process, at most every 10 minutes)."""
    import threading
    import time
    clock = clock or time.time
    if not str(setting("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "") or "").strip():
        return {"status": "MISSING_CREDENTIALS"}
    thread = _BOARD["thread"]
    if thread is not None and thread.is_alive():
        return {"status": "IN_PROGRESS_BACKGROUND", "last": _BOARD["last"]}
    fetched = _BOARD["fetched_at"]
    if fetched is not None and 0 <= clock() - fetched < BOARD_REFETCH_SECONDS:
        return {"status": "RECENT", "last": _BOARD["last"]}

    def run():
        try:
            if client_factory is None:
                from app_core.evidence_drive import DriveStore
                from app_core.evidence_remote import settings
                from app_core.read_only_drive import ReadOnlyDrive
                client = ReadOnlyDrive(DriveStore(settings()[0]))
            else:
                client = client_factory()
            predictions, key = fetch_board(client)
            _BOARD["predictions"] = predictions
            _BOARD["last"] = ({"status": "FETCHED", "board": key.rsplit("/", 1)[-1], "predictions": len(predictions)}
                              if key else {"status": "NO_BOARD_PUBLISHED"})
        except BaseException as exc:  # recorded, so the next Refresh can say why
            _BOARD["last"] = {"status": "FETCH_FAILED", "error": type(exc).__name__}
        finally:
            _BOARD["fetched_at"] = clock()
    thread = threading.Thread(target=run, name="research-model-board-fetch", daemon=True)
    _BOARD["thread"] = thread
    thread.start()
    return {"status": "STARTED_IN_BACKGROUND", "last": _BOARD["last"]}


def board_predictions(path=None, *, now=None) -> list[dict]:
    """The fetched board's predictions for games after now - 12h; FileNotFoundError until a fetch finishes."""
    if _BOARD["predictions"] is None:
        raise FileNotFoundError("research model board not fetched yet")
    now = now or datetime.now(timezone.utc)
    return [p for p in _BOARD["predictions"] if p["start"] >= now - LOOKBACK]


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


def _empty(frame: pd.DataFrame, status: str) -> pd.DataFrame:
    out = frame.copy()
    for column in COLUMNS:
        out[column] = np.nan if column not in ("research_model_predicted_at", "research_model_status") else None
    out["research_model_status"] = status
    return out


def attach(frame: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    """``frame`` with the research columns; ``research_model_status`` says why a row is empty.

    Per row: ATTACHED, NO_RESEARCH_MODEL_FOR_SPORT (only NBA, NCAAB and NHL have one),
    UNSUPPORTED_MARKET, NO_PREDICTION_FOR_GAME, AMBIGUOUS_GAME_MATCH, UNSUPPORTED_LINE.
    """
    from app_core.prospective_research_models import score_probabilities
    out = _empty(frame, "NO_PREDICTION_FOR_GAME")
    if out.empty:
        return out
    for index, row in out.iterrows():
        sport = str(row.get("league") or row.get("sport") or "").upper()
        if sport not in FAMILIES:
            out.at[index, "research_model_status"] = "NO_RESEARCH_MODEL_FOR_SPORT"
            continue
        side = _side(row)
        if side is None:
            out.at[index, "research_model_status"] = "UNSUPPORTED_MARKET"
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
            out.at[index, "research_model_status"] = "AMBIGUOUS_GAME_MATCH" if found else "NO_PREDICTION_FOR_GAME"
            continue
        p = found[0]
        try:
            split = score_probabilities(p["mean"], p["sd"], family, line, selection=selection)
        except ValueError:
            out.at[index, "research_model_status"] = "UNSUPPORTED_LINE"
            continue
        out.at[index, "research_model_probability"] = split["win"]
        out.at[index, "research_model_push_probability"] = split["push"]
        price = _decimal(row.get("odds_american"))
        if price is not None:
            out.at[index, "research_model_ev"] = split["win"] * (price - 1) - split["loss"]
        out.at[index, "research_model_predicted_at"] = p["predicted_at"]
        out.at[index, "research_model_status"] = "ATTACHED"
    return out


def _store_note(store: dict) -> str:
    """The background restore's state, in a few words, for a status cell."""
    store = store if isinstance(store, dict) else {}
    status = str(store.get("status") or "UNKNOWN")
    last = store.get("last") if isinstance(store.get("last"), dict) else {}
    note = status
    if last.get("status"):
        note += f"; last restore {last['status']}" + (f" {last['error']}" if last.get("error") else "")
        if last.get("board"):
            note += f" {last['board']}"
    return note


def apply(frame: pd.DataFrame, *, setting=os.environ.get, path=None, now=None, refresh=None, read=None):
    """(frame with the research columns, diagnostic). Never raises; changes no other column.

    ``research_model_status`` on every row says why a row has no probability; a store-wide
    reason (no credentials, restore still running or failed, store unreadable, store holds
    no current predictions) is written on every row with the restore's state.
    """
    if not enabled(setting):
        return _empty(frame, "OFF"), {"status": "OFF"}
    diagnostic = {}
    try:
        diagnostic["store"] = (refresh or start_board_fetch)(setting)
    except Exception as exc:  # never break Refresh
        diagnostic["store"] = {"status": "ERROR", "error": type(exc).__name__}
    note = _store_note(diagnostic["store"])
    try:
        predictions = (read or board_predictions)(path, now=now)
    except FileNotFoundError:
        status = ("STORE_MISSING_CREDENTIALS" if diagnostic["store"].get("status") == "MISSING_CREDENTIALS"
                  else f"STORE_NOT_DOWNLOADED_YET ({note})")
        return _empty(frame, status), dict(diagnostic, status="NO_MATCHING_PREDICTION", error="FileNotFoundError",
                                           reason=status, predictions=0)
    except Exception as exc:  # an unreadable store leaves the column empty
        status = f"STORE_READ_ERROR {type(exc).__name__} ({note})"
        return _empty(frame, status), dict(diagnostic, status="NO_MATCHING_PREDICTION", error=type(exc).__name__,
                                           reason=status, predictions=0)
    if not predictions:
        status = f"NO_CURRENT_PREDICTIONS_IN_STORE ({note})"
        return _empty(frame, status), dict(diagnostic, status="NO_MATCHING_PREDICTION", reason=status, predictions=0)
    out = attach(frame, predictions)
    counts = out["research_model_status"].value_counts().to_dict()
    filled = int(counts.get("ATTACHED", 0))
    diagnostic.update(status="ATTACHED" if filled else "NO_MATCHING_PREDICTION", predictions=len(predictions),
                      candidates_with_probability=filled, row_status=counts)
    return out, diagnostic
