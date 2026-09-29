"""Bounded, resumable, read-only census of prospective research evidence.

This module deliberately operates on immutable remote object bytes.  It does
not restore a database and does not import any capture, reconciliation,
backup, publication, billing, activation, or wagering entry point.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import base64
import binascii
import hashlib
import json
from pathlib import Path
import time


SCHEMA = "parlaypicker-read-only-census-v1"
CHECKPOINT_SCHEMA = "parlaypicker-read-only-census-checkpoint-v1"
CANONICAL_PREFIX = "parlaypicker/canonical-prospective-v1/"

SPORT_MARKETS = {
    "NFL": ("SPREAD", "TOTAL"),
    "NCAAF": ("SPREAD", "TOTAL"),
    "NBA": ("SPREAD", "TOTAL"),
    "NCAAB": ("SPREAD", "TOTAL"),
    "MLB": ("RUN_LINE", "TOTAL"),
    "NHL": ("PUCK_LINE", "TOTAL"),
}
SCOPES = tuple(f"{sport}/{market}" for sport, markets in SPORT_MARKETS.items()
               for market in markets)
SOURCE_PREFIXES = {
    "NFL": "parlaypicker/nfl-market-v1/",
    "NCAAF": "parlaypicker/ncaaf-prospective-v1/",
    "NBA": "parlaypicker/nba-market-v1/",
    "NCAAB": "parlaypicker/ncaab-market-v1/",
    "MLB": "parlaypicker/mlb-prospective-v1/",
    "NHL": "parlaypicker/nhl-market-v1/",
}
PREFIX_SPORT = {prefix: sport for sport, prefix in SOURCE_PREFIXES.items()}
TARGET_PREFIXES = (CANONICAL_PREFIX, *SOURCE_PREFIXES.values())
MAX_CANONICAL_OBJECT_BYTES = 60_000_000
MAX_SOURCE_OBJECT_BYTES = 40_000_000
READ_BATCH_OBJECTS = 8
SOURCE_KINDS = {
    "NFL": {"capture", "scores"},
    "NCAAF": {"model", "capture", "scores", "closing"},
    "NBA": {"capture", "scores", "pregame_close_candidate"},
    "NCAAB": {"capture", "scores", "pregame_close_candidate"},
    "MLB": {"model", "capture", "scores", "closing"},
    "NHL": {"capture", "scores", "pregame_close_candidate"},
}

# The canonical key binds each row to its SQLite primary key.  Declaring the
# keys here avoids opening or mutating a local database merely to inspect
# immutable object bytes.
PRIMARY_KEYS = {
    "prospective_event": ("event_id",),
    "prospective_quote": ("quote_id",),
    "prospective_close": ("close_id",),
    "prospective_result": ("result_id",),
    "prospective_model": ("model_id",),
    "prospective_model_training_result": ("model_id", "result_id"),
    "prospective_calibration": ("calibration_id",),
    "prospective_calibration_result": ("calibration_id", "result_id"),
    "prospective_prediction": ("observation_id",),
    "prospective_validation_plan": ("validation_plan_id",),
    "prospective_validation_artifact": ("artifact_id",),
    "prospective_deployment_review": ("deployment_id",),
    "prospective_football_event": ("version_id",),
    "prospective_football_quote": ("quote_id",),
    "prospective_football_result": ("result_id",),
    "prospective_football_settlement": ("settlement_id",),
    "prospective_football_training_row": ("training_row_id",),
    "prospective_football_team_identity": ("identity_id",),
    "prospective_football_theover": ("research_row_id",),
    "prospective_football_coverage": ("coverage_id",),
    "prospective_football_cycle_coverage": ("coverage_id",),
}


class CensusIntegrityError(ValueError):
    pass


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _prefix_for(name):
    return next((prefix for prefix in TARGET_PREFIXES if name.startswith(prefix)), None)


def _metadata_token(items):
    return _sha(_json_bytes(sorted((item.get("id"), item.get("name"),
                                    item.get("sha256Checksum")) for item in items)))


def _membership_digest(groups):
    return _sha(_json_bytes(sorted((name, _metadata_token(items))
                                    for name, items in groups.items())))


def _checkpoint_digest(value):
    unsigned = {key: item for key, item in value.items()
                if key != "checkpoint_sha256"}
    return _sha(_json_bytes(unsigned))


def _read_checkpoint(path):
    if path is None or not Path(path).is_file():
        return None, None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None, "CHECKPOINT_JSON_INVALID"
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
        return None, "CHECKPOINT_SCHEMA_UNSUPPORTED"
    if value.get("checkpoint_sha256") != _checkpoint_digest(value):
        return None, "CHECKPOINT_DIGEST_MISMATCH"
    return value, None


def _atomic_json(path, value):
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(target)


def _save_checkpoint(path, *, scope_hash, operation_id, inventory_digests,
                     processed, source_revision):
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "storage_scope_hash": scope_hash,
        "operation_id": operation_id,
        "source_revision": source_revision,
        "requested_scopes": list(SCOPES),
        "inventory_membership": inventory_digests,
        "processed": processed,
        "updated_at": _utc_now(),
    }
    value["checkpoint_sha256"] = _checkpoint_digest(value)
    _atomic_json(path, value)


def _decode_blob(value):
    if not isinstance(value, dict):
        return value
    if set(value) != {"blob_base64"} or not isinstance(value["blob_base64"], str):
        raise CensusIntegrityError("CANONICAL_BLOB_INVALID")
    try:
        return base64.b64decode(value["blob_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CensusIntegrityError("CANONICAL_BLOB_INVALID") from exc


def _normalize_market(value, sport):
    value = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "SPREADS": "SPREAD",
        "SPREAD": "SPREAD",
        "RUNLINE": "RUN_LINE",
        "RUN_LINES": "RUN_LINE",
        "RUN_LINE": "RUN_LINE",
        "PUCKLINE": "PUCK_LINE",
        "PUCK_LINES": "PUCK_LINE",
        "PUCK_LINE": "PUCK_LINE",
        "TOTALS": "TOTAL",
        "TOTAL": "TOTAL",
    }
    market = aliases.get(value)
    if market == "SPREAD" and sport == "MLB":
        return "RUN_LINE"
    if market == "SPREAD" and sport == "NHL":
        return "PUCK_LINE"
    return market


def _canonical_fact(name, raw):
    if len(raw) > MAX_CANONICAL_OBJECT_BYTES:
        raise CensusIntegrityError("CANONICAL_OBJECT_TOO_LARGE")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise CensusIntegrityError("CANONICAL_JSON_INVALID") from exc
    if (not isinstance(payload, dict) or payload.get("schema") != 1
            or _json_bytes(payload) != raw):
        raise CensusIntegrityError("CANONICAL_ENCODING_INVALID")
    table = payload.get("table")
    columns, values = payload.get("columns"), payload.get("row")
    primary = PRIMARY_KEYS.get(table)
    if (primary is None or not isinstance(columns, list) or not isinstance(values, list)
            or len(columns) != len(values) or len(columns) != len(set(columns))
            or any(column not in columns for column in primary)):
        raise CensusIntegrityError("CANONICAL_SCHEMA_INVALID")
    decoded = [_decode_blob(value) for value in values]
    row = dict(zip(columns, decoded))
    identity = [row[column] for column in primary]
    expected = (f"{CANONICAL_PREFIX}{table}/"
                f"{_sha(_json_bytes([table, identity]))}.json")
    if name != expected:
        raise CensusIntegrityError("CANONICAL_KEY_MISMATCH")
    for field, digest_field in (("payload", "payload_hash"),
                                ("raw_source", "source_hash")):
        if field in row and digest_field in row:
            value = row[field]
            if not isinstance(value, (str, bytes)):
                raise CensusIntegrityError("CANONICAL_EVIDENCE_VALUE_INVALID")
            encoded = value.encode() if isinstance(value, str) else value
            if _sha(encoded) != row[digest_field]:
                raise CensusIntegrityError("CANONICAL_EVIDENCE_HASH_MISMATCH")
    sport = str(row.get("sport") or "").upper() or None
    market = _normalize_market(row.get("market_family"), sport)
    fact = {
        "record_type": table,
        "sport": sport,
        "market_family": market,
        "event_id": row.get("event_id") or row.get("game_id"),
    }
    keep = (
        "model_id", "model_version", "training_start", "training_cutoff",
        "training_observation_count", "independent_event_count", "feature_version",
        "training_code_commit", "created_at", "available_at", "artifact_hash",
        "calibration_id", "calibration_version", "fit_start", "fit_end", "method",
        "observation_id", "prediction_timestamp", "quote_id", "quote_verified",
        "close_id", "close_verified", "result_id", "settlement_id",
        "training_row_id", "training_row_status", "available_for_training_at",
        "validation_plan_id", "version", "validation_start", "validation_end",
        "holdout_start", "holdout_end", "minimum_independent_sample",
        "minimum_effective_sample", "frozen_at", "artifact_id", "status",
        "report_hash", "deployment_id", "deployment_state", "reviewed_at",
        "evidence_snapshot_id", "evidence_hash", "runtime_hash", "source_commit",
        "capture_run_id", "requested_slate_success", "observed_at",
    )
    fact.update({field: row[field] for field in keep if row.get(field) is not None})
    if isinstance(row.get("payload"), str):
        try:
            detail = json.loads(row["payload"])
        except ValueError:
            detail = None
        if isinstance(detail, dict):
            for key in ("metrics", "observed_metrics", "evaluation", "cohort",
                        "selection_count", "calibration_count", "evaluation_count"):
                if key in detail:
                    fact[key] = detail[key]
    return [fact]


def _event_identity(event):
    for key in ("game_id", "event_id", "cfbd_id", "id", "key"):
        if event.get(key) is not None:
            return str(event[key])
    return "sha256:" + _sha(_json_bytes(event))


def _source_facts(name, raw, prefix):
    if len(raw) > MAX_SOURCE_OBJECT_BYTES:
        raise CensusIntegrityError("SOURCE_OBJECT_TOO_LARGE")
    digest = _sha(raw)
    if name != prefix + digest + ".json":
        raise CensusIntegrityError("SOURCE_KEY_MISMATCH")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise CensusIntegrityError("SOURCE_JSON_INVALID") from exc
    if (not isinstance(record, dict) or record.get("schema") != 1
            or _json_bytes(record) != raw):
        raise CensusIntegrityError("SOURCE_ENCODING_INVALID")
    sport = PREFIX_SPORT[prefix]
    if record.get("kind") not in SOURCE_KINDS[sport]:
        raise CensusIntegrityError("SOURCE_KIND_INVALID")
    data = record.get("data")
    if not isinstance(data, dict):
        raise CensusIntegrityError("SOURCE_DATA_INVALID")
    declared = str(data.get("sport") or sport).upper()
    if declared != sport:
        raise CensusIntegrityError("SOURCE_SPORT_MISMATCH")
    facts = [{
        "record_type": "source_record",
        "sport": sport,
        "market_family": None,
        "source_kind": record.get("kind"),
        "created_at": record.get("created_at"),
    }]
    events = data.get("events")
    if not isinstance(events, list):
        events = [data] if any(data.get(key) is not None for key in
                               ("game_id", "event_id", "cfbd_id", "id")) else []
    for event in events:
        if isinstance(event, dict):
            facts.append({"record_type": "source_event", "sport": sport,
                          "market_family": None, "event_id": _event_identity(event),
                          "source_kind": record.get("kind")})
    if record.get("kind") == "model":
        facts.append({
            "record_type": "source_model",
            "sport": sport,
            "market_family": _normalize_market(data.get("market_family"), sport),
            "model_id": data.get("model_id") or data.get("id") or digest,
            "model_version": data.get("model_version") or data.get("version"),
            "artifact_hash": data.get("artifact_hash") or digest,
            "training_cutoff": data.get("training_cutoff") or data.get("trained_through"),
            "independent_event_count": data.get("independent_event_count"),
        })
    return facts


def _facts(name, raw, prefix):
    if prefix == CANONICAL_PREFIX:
        return _canonical_fact(name, raw)
    return _source_facts(name, raw, prefix)


def _scope_keys(fact):
    sport = fact.get("sport")
    if sport not in SPORT_MARKETS:
        return ()
    market = fact.get("market_family")
    if market in SPORT_MARKETS[sport]:
        return (f"{sport}/{market}",)
    return tuple(f"{sport}/{candidate}" for candidate in SPORT_MARKETS[sport])


def _public_fact(fact, fields):
    return {field: fact[field] for field in fields if fact.get(field) is not None}


def _scope_report(scope, facts, raw_names, namespace_states, snapshot):
    sport, market = scope.split("/", 1)
    relevant = [fact for fact in facts if scope in _scope_keys(fact)]
    event_ids = sorted({str(fact["event_id"]) for fact in relevant
                        if fact.get("event_id") is not None})
    training_ids = sorted({str(fact["event_id"]) for fact in relevant
                           if fact.get("record_type") == "prospective_football_training_row"
                           and fact.get("training_row_status") == "TRAINING_READY"
                           and fact.get("event_id") is not None})
    models = [_public_fact(fact, (
        "model_id", "model_version", "artifact_hash", "training_start",
        "training_cutoff", "training_observation_count", "independent_event_count",
        "feature_version", "training_code_commit", "available_at"))
        for fact in relevant if fact.get("record_type") in
        {"prospective_model", "source_model"}]
    calibrations = [_public_fact(fact, (
        "calibration_id", "calibration_version", "model_id", "artifact_hash",
        "fit_start", "fit_end", "method", "available_at"))
        for fact in relevant if fact.get("record_type") == "prospective_calibration"]
    plans = [_public_fact(fact, (
        "validation_plan_id", "version", "model_id", "calibration_id",
        "training_cutoff", "validation_start", "validation_end", "holdout_start",
        "holdout_end", "minimum_independent_sample", "minimum_effective_sample",
        "frozen_at", "artifact_hash"))
        for fact in relevant if fact.get("record_type") == "prospective_validation_plan"]
    artifacts = [_public_fact(fact, (
        "artifact_id", "validation_plan_id", "status", "report_hash", "created_at",
        "metrics", "observed_metrics", "evaluation"))
        for fact in relevant if fact.get("record_type") == "prospective_validation_artifact"]
    reviews = [_public_fact(fact, (
        "deployment_id", "artifact_id", "validation_plan_id", "deployment_state",
        "reviewed_at")) for fact in relevant
        if fact.get("record_type") == "prospective_deployment_review"]
    binding_counts = defaultdict(int)
    for fact in relevant:
        if fact.get("record_type") == "prospective_prediction":
            key = (fact.get("model_id"), fact.get("calibration_id"),
                   fact.get("source_commit"), fact.get("runtime_hash"),
                   fact.get("evidence_snapshot_id"))
            binding_counts[key] += 1
    bindings = [
        {"model_id": key[0], "calibration_id": key[1], "source_commit": key[2],
         "runtime_hash": key[3], "evidence_snapshot_id": key[4], "observations": count}
        for key, count in sorted(binding_counts.items(), key=lambda item: repr(item[0]))
    ]
    required = (CANONICAL_PREFIX, SOURCE_PREFIXES[sport])
    required_states = [namespace_states[prefix]["status"] for prefix in required]
    if any(state == "BLOCKED" for state in required_states):
        state = "BLOCKED"
    elif any(state != "COMPLETE" for state in required_states):
        state = "PARTIAL"
    else:
        state = "COMPLETE"
    if state != "COMPLETE":
        qualification = "UNKNOWN_INCOMPLETE_CENSUS"
        next_blocker = "COMPLETE_VERIFIED_CANONICAL_AND_SOURCE_READS"
    elif not models:
        qualification = "BLOCKED_NO_MODEL"
        next_blocker = "MODEL_OWNER_QUALIFICATION"
    elif not calibrations:
        qualification = "BLOCKED_NO_CALIBRATION"
        next_blocker = "MODEL_OWNER_CALIBRATION"
    elif not plans:
        qualification = "BLOCKED_NO_FROZEN_PLAN"
        next_blocker = "MODEL_OWNER_FREEZE_VALIDATION_PLAN"
    elif not any(item.get("status") == "VALIDATION_PASSED" for item in artifacts):
        qualification = "BLOCKED_NO_PASSED_VALIDATION"
        next_blocker = "MODEL_OWNER_UNTOUCHED_EVALUATION"
    else:
        qualification = "EVIDENCE_PRESENT_OWNER_REVIEW_REQUIRED"
        next_blocker = "OWNER_REVIEW_AND_SEPARATE_ACTIVATION_DECISION"
    direct_eligibility = bool(any(
        fact.get("record_type") == "prospective_football_training_row"
        for fact in relevant))
    return {
        "scope": scope,
        "census_state": state,
        "snapshot": snapshot,
        "required_namespaces": {prefix: namespace_states[prefix] for prefix in required},
        "counts": {
            "raw_objects": len(raw_names),
            # Every selected remote object is one immutable canonical/source
            # record; expanded source events are counted separately below.
            "records": len(raw_names),
            "unique_events": len(event_ids),
            "independent_eligible_games": len(training_ids) if direct_eligibility else None,
            "independent_eligible_reason": (None if direct_eligibility else
                "NO_DIRECT_CANONICAL_ELIGIBILITY_ROWS_FOR_SCOPE"),
            "quotes": sum(fact.get("record_type") in
                          {"prospective_quote", "prospective_football_quote"}
                          for fact in relevant),
            "verified_quotes": sum(fact.get("record_type") in
                          {"prospective_quote", "prospective_football_quote"}
                          and fact.get("quote_verified") == 1 for fact in relevant),
            "results": sum(fact.get("record_type") in
                           {"prospective_result", "prospective_football_result"}
                           for fact in relevant),
            "settlements": sum(fact.get("record_type") ==
                               "prospective_football_settlement" for fact in relevant),
        },
        "model_records": models or [{"status": "UNKNOWN", "reason":
            "NO_MODEL_RECORD_IN_VERIFIED_CENSUS" if state == "COMPLETE" else
            "CANONICAL_CENSUS_INCOMPLETE"}],
        "calibration_records": calibrations or [{"status": "UNKNOWN", "reason":
            "NO_CALIBRATION_RECORD_IN_VERIFIED_CENSUS" if state == "COMPLETE" else
            "CANONICAL_CENSUS_INCOMPLETE"}],
        "consumer_bindings": bindings or [{"status": "UNKNOWN", "reason":
            "NO_PREDICTION_BINDING_IN_VERIFIED_CENSUS" if state == "COMPLETE" else
            "CANONICAL_CENSUS_INCOMPLETE"}],
        "validation_plans": plans or [{"status": "UNKNOWN", "reason":
            "NO_FROZEN_PLAN_IN_VERIFIED_CENSUS" if state == "COMPLETE" else
            "CANONICAL_CENSUS_INCOMPLETE"}],
        "validation_artifacts": artifacts,
        "deployment_reviews": reviews,
        "qualification_status": qualification,
        "activation_status": "NOT_ACTIVATED_BY_READ_ONLY_CENSUS",
        "next_blocker": next_blocker,
        "next_blocker_owner": "model_owner" if "MODEL_OWNER" in next_blocker else "repository_owner",
    }


def _build_report(*, source_revision, inventory, namespace_groups, processed,
                  namespace_errors, metrics, checkpoint_note, started_at,
                  terminal_reason=None, budget_contract=None):
    namespace_states = {}
    for prefix in TARGET_PREFIXES:
        names = set(namespace_groups[prefix])
        done = {name for name in names if name in processed}
        errors = namespace_errors.get(prefix, [])
        if errors:
            status = "BLOCKED"
        elif done == names:
            status = "COMPLETE"
        else:
            status = "PARTIAL"
        namespace_states[prefix] = {
            "status": status,
            "inventory_objects": len(names),
            "verified_objects": len(done),
            "remaining_objects": len(names - done),
            "membership_sha256": _membership_digest(namespace_groups[prefix]),
            "errors": errors,
        }
    all_facts = []
    raw_by_sport = defaultdict(set)
    for name, entry in processed.items():
        for fact in entry.get("facts", []):
            all_facts.append(fact)
            for scope in _scope_keys(fact):
                raw_by_sport[scope].add(name)
    snapshot = {
        "source_revision": source_revision,
        "as_of": _utc_now(),
        "storage_scope_hash": inventory.scope_hash,
        "inventory_operation_id": inventory.operation_id,
        "inventory_listing_pages": inventory.listing_pages,
        "inventory_metadata_items_seen": inventory.metadata_items_seen,
    }
    scopes = [_scope_report(scope, all_facts, raw_by_sport[scope],
                            namespace_states, snapshot) for scope in SCOPES]
    statuses = {item["census_state"] for item in scopes}
    overall = ("BLOCKED" if "BLOCKED" in statuses else
               "PARTIAL" if "PARTIAL" in statuses else "COMPLETE")
    if terminal_reason and overall == "COMPLETE":
        overall = "PARTIAL"
    return {
        "schema": SCHEMA,
        "execution_kind": "AUTHENTICATED_READ_ONLY_REMOTE_CENSUS",
        "status": overall,
        "terminal_reason": terminal_reason,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "source_revision": source_revision,
        "requested_scopes": list(SCOPES),
        "read_only_contract": {
            "remote_operations": ["complete_inventory", "verified_object_read"],
            "restore_calls": 0,
            "reconciliation_calls": 0,
            "capture_calls": 0,
            "receipt_recovery_calls": 0,
            "backup_calls": 0,
            "activation_calls": 0,
            "billing_calls": 0,
            "wager_calls": 0,
        },
        "checkpoint": checkpoint_note,
        "budget_contract": budget_contract,
        "metrics": metrics,
        "namespaces": namespace_states,
        "scopes": scopes,
        "launch_authority": "NONE",
    }


def run_census(client, *, checkpoint_path=None, output_path=None,
               source_revision="UNKNOWN", max_objects=4000,
               max_bytes=500_000_000, deadline_seconds=2400,
               full_verify=False, clock=time.monotonic, progress=None):
    """Run one bounded read-only census slice and persist resumable evidence."""
    if max_objects <= 0 or max_bytes <= 0 or deadline_seconds <= 0:
        raise ValueError("census_limits_must_be_positive")
    started_at, started = _utc_now(), clock()
    inventory_started = clock()
    inventory = client.discover_complete_inventory(namespace=SCHEMA)
    inventory_seconds = clock() - inventory_started
    groups = {prefix: {} for prefix in TARGET_PREFIXES}
    for item in inventory.files:
        prefix = _prefix_for(item.get("name", ""))
        if prefix:
            groups[prefix].setdefault(item["name"], []).append(item)
    digests = {prefix: _membership_digest(group) for prefix, group in groups.items()}
    namespace_errors = defaultdict(list)
    conflicted_names = set()
    for prefix, names in groups.items():
        for name, items in names.items():
            checksums = {item.get("sha256Checksum") for item in items}
            if len(checksums - {None, ""}) > 1:
                conflicted_names.add(name)
                namespace_errors[prefix].append({
                    "reason": "CONFLICTING_DUPLICATE_PROVIDER_CHECKSUMS",
                    "object_name_sha256": _sha(name.encode()),
                })
    checkpoint, rejection = _read_checkpoint(checkpoint_path)
    processed = {}
    invalidated = defaultdict(int)
    if checkpoint is not None:
        if checkpoint.get("storage_scope_hash") != inventory.scope_hash:
            rejection = "CHECKPOINT_STORAGE_SCOPE_MISMATCH"
        elif checkpoint.get("requested_scopes") != list(SCOPES):
            rejection = "CHECKPOINT_REQUESTED_SCOPE_MISMATCH"
        elif checkpoint.get("source_revision") != source_revision:
            rejection = "CHECKPOINT_SOURCE_REVISION_MISMATCH"
        else:
            current_names = {name: items for group in groups.values()
                             for name, items in group.items()}
            for name, entry in checkpoint.get("processed", {}).items():
                items = current_names.get(name)
                if items is None:
                    invalidated["deleted"] += 1
                elif name in conflicted_names:
                    invalidated["conflicting_duplicate"] += 1
                elif not all(item.get("sha256Checksum") for item in items):
                    invalidated["missing_provider_checksum"] += 1
                elif entry.get("metadata_token") != _metadata_token(items):
                    invalidated["changed_metadata"] += 1
                elif entry.get("content_sha256") not in {
                        item.get("sha256Checksum") for item in items}:
                    invalidated["checksum_conflict"] += 1
                elif full_verify:
                    invalidated["full_verification_requested"] += 1
                else:
                    processed[name] = entry
    if rejection:
        processed = {}
    queues = {}
    for prefix, names in groups.items():
        queues[prefix] = deque(sorted(name for name in names if name not in processed
                                      and not any(error.get("object_name_sha256") ==
                                      _sha(name.encode()) for error in namespace_errors[prefix])))
    selected = []
    active = deque(prefix for prefix in TARGET_PREFIXES if queues[prefix])
    while active and len(selected) < max_objects:
        prefix = active.popleft()
        selected.append((prefix, queues[prefix].popleft()))
        if queues[prefix]:
            active.append(prefix)
    metrics = {
        "inventory_seconds": round(inventory_seconds, 6),
        "listing_traversals": 1,
        "listing_pages": inventory.listing_pages,
        "metadata_items_seen": inventory.metadata_items_seen,
        "download_verification_seconds": 0.0,
        "parse_import_seconds": 0.0,
        "object_read_batches": 0,
        "objects_downloaded": 0,
        "bytes_downloaded": 0,
        "verified_new_objects": 0,
        "verified_reused_objects": len(processed),
        "invalidated_checkpoint_objects": dict(invalidated),
    }
    terminal_reason = None
    budget_contract = {
        "max_objects": max_objects,
        "max_bytes": max_bytes,
        "deadline_seconds": deadline_seconds,
        "read_batch_objects": READ_BATCH_OBJECTS,
        "object_limit_enforcement": "HARD_SELECTION_LIMIT",
        "byte_limit_enforcement": "POST_BATCH_SOFT_LIMIT",
        "maximum_byte_limit_overrun": READ_BATCH_OBJECTS * MAX_CANONICAL_OBJECT_BYTES,
        "deadline_enforcement": "PRE_BATCH_WITH_TRANSPORT_TIMEOUTS",
        "workflow_shutdown_margin_seconds": 300,
        "hard_kill_artifact_retention": "NOT_GUARANTEED",
    }
    # One batch is two waves at DriveStore's existing four-worker limit. This
    # retains duplicate-byte conflict checks while bounding byte/deadline
    # overrun and leaving five minutes for the workflow artifact-upload step.
    for offset in range(0, len(selected), READ_BATCH_OBJECTS):
        if clock() - started >= deadline_seconds:
            terminal_reason = "DEADLINE_REACHED"
            break
        batch = selected[offset:offset + READ_BATCH_OBJECTS]
        exact_names = [name for _, name in batch]
        read_started = clock()
        try:
            values = client.read_verified_prefixes(
                Prefixes=exact_names, inventory=inventory, cache_dir=None,
                full_verify=full_verify)
            metrics["object_read_batches"] += 1
            report = getattr(client, "last_read_report", None)
            if report is not None:
                metrics["objects_downloaded"] += report.objects_downloaded
                metrics["bytes_downloaded"] += report.bytes_downloaded
        except Exception as exc:
            # Exception messages from authenticated transports can contain
            # request details.  Keep terminal diagnostics class-only here.
            reason = type(exc).__name__
            for prefix, _ in batch:
                namespace_errors[prefix].append({"reason": "VERIFIED_READ_FAILED",
                                                 "detail": reason})
            terminal_reason = "VERIFIED_READ_FAILED"
            break
        metrics["download_verification_seconds"] += clock() - read_started
        parse_started = clock()
        for prefix, name in batch:
            matches = values.get(name, [])
            if len(matches) != 1 or matches[0][0] != name:
                namespace_errors[prefix].append({"reason": "EXACT_OBJECT_READ_MISMATCH",
                                                 "object_name_sha256": _sha(name.encode())})
                continue
            raw = matches[0][1]
            content_sha = _sha(raw)
            provider = {item.get("sha256Checksum") for item in groups[prefix][name]
                        if item.get("sha256Checksum")}
            if provider and provider != {content_sha}:
                namespace_errors[prefix].append({"reason": "PROVIDER_CHECKSUM_MISMATCH",
                                                 "object_name_sha256": _sha(name.encode())})
                continue
            try:
                facts = _facts(name, raw, prefix)
            except (CensusIntegrityError, TypeError, ValueError) as exc:
                namespace_errors[prefix].append({"reason": str(exc)[:200],
                                                 "object_name_sha256": _sha(name.encode())})
                continue
            processed[name] = {
                "namespace": prefix,
                "metadata_token": _metadata_token(groups[prefix][name]),
                "content_sha256": content_sha,
                "facts": facts,
            }
            metrics["verified_new_objects"] += 1
        metrics["parse_import_seconds"] += clock() - parse_started
        _save_checkpoint(checkpoint_path, scope_hash=inventory.scope_hash,
                         operation_id=inventory.operation_id,
                         inventory_digests=digests, processed=processed,
                         source_revision=source_revision)
        if metrics["bytes_downloaded"] >= max_bytes:
            terminal_reason = "BYTE_LIMIT_REACHED"
            break
        checkpoint_note = {
            "path_recorded": checkpoint_path is not None,
            "accepted_prior_checkpoint": checkpoint is not None and rejection is None,
            "rejection_reason": rejection,
            "invalidations": dict(invalidated),
        }
        report_value = _build_report(
            source_revision=source_revision, inventory=inventory,
            namespace_groups=groups, processed=processed,
            namespace_errors=namespace_errors, metrics=metrics,
            checkpoint_note=checkpoint_note, started_at=started_at,
            terminal_reason=terminal_reason or "SLICE_IN_PROGRESS",
            budget_contract=budget_contract)
        _atomic_json(output_path, report_value)
        if progress:
            progress({
                "status": report_value["status"],
                "terminal_reason": report_value["terminal_reason"],
                "verified_new_objects": metrics["verified_new_objects"],
                "verified_reused_objects": metrics["verified_reused_objects"],
                "scopes": {item["scope"]: item["census_state"]
                           for item in report_value["scopes"]},
            })
    if terminal_reason is None and any(
            name not in processed for names in groups.values() for name in names):
        terminal_reason = "OBJECT_LIMIT_REACHED"
    _save_checkpoint(checkpoint_path, scope_hash=inventory.scope_hash,
                     operation_id=inventory.operation_id,
                     inventory_digests=digests, processed=processed,
                     source_revision=source_revision)
    checkpoint_note = {
        "path_recorded": checkpoint_path is not None,
        "accepted_prior_checkpoint": checkpoint is not None and rejection is None,
        "rejection_reason": rejection,
        "invalidations": dict(invalidated),
    }
    report_value = _build_report(
        source_revision=source_revision, inventory=inventory,
        namespace_groups=groups, processed=processed,
        namespace_errors=namespace_errors, metrics=metrics,
        checkpoint_note=checkpoint_note, started_at=started_at,
        terminal_reason=terminal_reason, budget_contract=budget_contract)
    _atomic_json(output_path, report_value)
    if progress:
        progress({
            "status": report_value["status"],
            "terminal_reason": report_value["terminal_reason"],
            "verified_new_objects": metrics["verified_new_objects"],
            "verified_reused_objects": metrics["verified_reused_objects"],
            "scopes": {item["scope"]: item["census_state"]
                       for item in report_value["scopes"]},
        })
    return report_value
