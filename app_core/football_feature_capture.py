"""Record live generated football inputs without claiming historical availability."""
from datetime import datetime, timezone
import json
import math
import pandas as pd
from core.exposure_ledger import digest
from core.wager_decisions import aware

FIELDS = ("football_feature_receipt", "football_feature_capture_status")


def capture(frame, now=None):
    out = frame.copy()
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        raise ValueError("aware capture time required")
    out["football_feature_receipt"] = None
    out["football_feature_capture_status"] = None
    for index, row in out.iterrows():
        sport = str(row.get("league", row.get("League", ""))).upper()
        if sport not in {"NFL", "NCAAF"}:
            continue
        # Expanded market candidates retain the provider kickoff under this raw
        # field before the evidence projection assigns game_start_utc.
        start = (aware(row.get("game_start_utc")) or aware(row.get("commence_time"))
                 or aware(row.get("commence_time_raw")))
        if start is None or clock >= start:
            out.at[index, "football_feature_capture_status"] = "NOT_VERIFIED_PREGAME"
            continue
        features = {}
        for name in frame.columns:
            if not name.startswith("feature_"):
                continue
            try:
                value = float(row[name])
            except (ValueError, TypeError):
                continue
            if math.isfinite(value):
                features[name] = value
        if not features:
            out.at[index, "football_feature_capture_status"] = "NO_FEATURE_VALUES"
            continue
        def text(key):
            v = row.get(key)
            return None if v is None or pd.isna(v) else str(v)
        payload = dict(schema="football-feature-observation-v1", sport=sport,
            observed_at=clock.isoformat(), game_start_utc=start.isoformat(),
            home_team_id=text("home_team_id"), away_team_id=text("away_team_id"),
            home_team=text("home_team"), away_team=text("away_team"),
            matchup_id=text("matchup_id"), stats_source=text("stats_source"),
            stats_resolution_status=text("stats_resolution_status"),
            stats_fallback_reason=text("stats_fallback_reason"), features=features,
            source_available_at=None, historical_availability_verified=False,
            training_authorized=False)
        out.at[index, "football_feature_receipt"] = json.dumps(
            {"payload":payload,"sha256":digest(payload)}, sort_keys=True, allow_nan=False)
        out.at[index, "football_feature_capture_status"] = "OBSERVED_RESEARCH_INPUTS"
    return out
