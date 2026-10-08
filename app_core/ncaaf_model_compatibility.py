"""Exact recovered NCAAF model reader. Static inspection only, never inference.

The old schema/runtime reader remains unchanged. This reader recognizes one
reviewed recovery, with historical and consumed identities kept separately.
It opens no store, imports no provider, and grants no source or wager authority.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform

from app_core import ncaaf_research as research, ncaaf_identity as identity
from app_core.ncaaf_history import timestamp

VERSION = "ncaaf-recovered-model-compatibility-v1"
BINDING_PATH = Path(__file__).resolve().parents[1] / "docs/paid-launch/ncaaf-compatibility-binding-v1.json"
MAX_RECORD_BYTES = 64 * 1024
CAPABILITIES = {"spread": "selected_side_full_game_half_point_cover",
                "total": "full_game_half_point_over_under"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def _binding():
    return json.loads(BINDING_PATH.read_bytes())


def _record(raw, expected):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_RECORD_BYTES,
            "NCAAF_COMPAT_RECORD_SIZE")
    require(hashlib.sha256(raw).hexdigest() == expected, "NCAAF_COMPAT_RECORD_HASH")
    record = json.loads(raw)
    require(set(record) == {"schema", "kind", "created_at", "data"}
            and record["schema"] == 1 and record["kind"] == "model"
            and encode(record) == raw, "NCAAF_COMPAT_RECORD_SCHEMA")
    return record


def current_runtime():
    """Actual installed bytes and environment, distinct from Git/history hashes."""
    root = Path(__file__).resolve().parents[1]
    binding = _binding()
    components = {}
    for name, approved in binding["current_components"].items():
        raw = (root / name).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(actual in approved["installed_sha256"], "NCAAF_COMPAT_RUNTIME_COMPONENT_CHANGED")
        components[name] = dict(sha256=actual, reviewed_git_blob_sha256=approved["git_blob_sha256"])
    # No dynamic fallback is used for event matching. Record its actual state;
    # unreviewed contents are rejected rather than presumed historically absent.
    dynamic = root / "data/dynamic_aliases.json"
    require(not dynamic.exists(), "NCAAF_COMPAT_DYNAMIC_ALIASES_UNREVIEWED")
    require(digest(identity.ALIASES) == binding["current_alias_table_sha256"],
            "NCAAF_COMPAT_ALIAS_TABLE_CHANGED")
    from core import team_mapper
    require(team_mapper.NHL_EXACT_MAP.get("new york rangers") == "New York Rangers"
            and team_mapper.NHL_EXACT_MAP.get("new york islanders") == "New York Islanders",
            "NCAAF_COMPAT_SHARED_CITY_CONFLICT")
    return dict(components=components, python=platform.python_version(),
        numpy=research.np.__version__, implementation=platform.python_implementation(),
        platform=platform.platform(), dynamic_aliases=dict(status="ABSENT", consumed=False),
        alias_table_sha256=digest(identity.ALIASES))


def envelope(*, family, model_name="ridge"):
    """Describe this software compatibility binding, never accept a source."""
    binding = _binding()
    require(family in CAPABILITIES and model_name == "ridge", "NCAAF_COMPAT_TARGET_UNSUPPORTED")
    return dict(version=VERSION, binding_sha256=digest(binding),
        original_record_sha256=binding["original_record_sha256"],
        predecessor_record_sha256=binding["predecessor_record_sha256"],
        artifact_sha256=binding["artifact_sha256"],
        historical_runtime=deepcopy(binding["historical_runtime"]),
        consumed_reader=current_runtime(), model_name=model_name, family=family,
        capability=CAPABILITIES[family], integer_push="UNVALIDATED_UNAVAILABLE",
        source_acceptance=False, probability_calibration=False,
        scientific_qualification=False, wagering_authority=False, live_stake=0)


def read_model(original_bytes, predecessor_bytes, compatibility):
    """Verify authentic records statically; preserve original bytes and declarations."""
    binding = _binding()
    original = _record(original_bytes, binding["original_record_sha256"])
    predecessor = _record(predecessor_bytes, binding["predecessor_record_sha256"])
    old_clock, recovered_clock = timestamp(predecessor["created_at"]), timestamp(original["created_at"])
    require(old_clock is not None and recovered_clock is not None and old_clock < recovered_clock,
            "NCAAF_COMPAT_LINEAGE_CLOCK_CONFLICT")
    require(all(digest(v["components"]) == v["sha256"]
                for k, v in binding["historical_runtime"].items() if k in {"source", "recovered"}),
            "NCAAF_COMPAT_HISTORICAL_COMPONENT_CONFLICT")
    data, previous = original["data"], predecessor["data"]
    require(set(data) == {"artifact", "artifact_hash", "runtime_hash", "policy", "recovery"}
            and set(previous) == {"artifact", "artifact_hash", "runtime_hash", "policy"},
            "NCAAF_COMPAT_RECOVERY_SCHEMA")
    require(data["recovery"] == binding["recovery"]
            and data["recovery"]["source_model_id"] == binding["predecessor_record_sha256"]
            and data["recovery"]["production_eligible"] is False,
            "NCAAF_COMPAT_RECOVERY_CONFLICT")
    require(data["runtime_hash"] == binding["historical_runtime"]["recovered"]["sha256"]
            and previous["runtime_hash"] == binding["historical_runtime"]["source"]["sha256"]
            and data["recovery"]["source_runtime"] == previous["runtime_hash"],
            "NCAAF_COMPAT_LINEAGE_CONFLICT")
    require(all(data[k] == previous[k] for k in ("artifact", "artifact_hash", "policy")),
            "NCAAF_COMPAT_ARTIFACT_LINEAGE_CONFLICT")
    artifact = data["artifact"]
    require(digest(artifact) == data["artifact_hash"] == binding["artifact_sha256"]
            and digest(artifact["protocol"]) == binding["protocol_sha256"]
            and digest(artifact["models"]) == binding["parameters_sha256"]
            and artifact["train_hash"] == binding["training_sha256"]
            and artifact["calibration_hash"] == binding["calibration_sha256"]
            and artifact["protocol"] == research.PROTOCOL
            and artifact["protocol"]["production_eligible"] is False,
            "NCAAF_COMPAT_ARTIFACT_CONFLICT")
    require(artifact["source_hash"] == binding["current_components"]["app_core/ncaaf_research.py"]["git_blob_sha256"],
            "NCAAF_COMPAT_MATH_CHANGED")
    require(isinstance(compatibility, dict) and compatibility == envelope(
        family=compatibility.get("family"), model_name=compatibility.get("model_name")),
        "NCAAF_COMPAT_ENVELOPE_CONFLICT")
    return dict(version=VERSION, status="MODEL_READER_COMPATIBLE",
        original_bytes=original_bytes, predecessor_bytes=predecessor_bytes,
        original_record=deepcopy(original), predecessor_record=deepcopy(predecessor),
        compatibility=deepcopy(compatibility), artifact=deepcopy(artifact))


def export_model(checked):
    """Private JSON preserves complete original bytes; no public projection."""
    original = checked["original_bytes"]
    predecessor = checked["predecessor_bytes"]
    read_model(original, predecessor, checked["compatibility"])
    return dict(version=VERSION, original_record_b64=base64.b64encode(original).decode("ascii"),
        predecessor_record_b64=base64.b64encode(predecessor).decode("ascii"),
        compatibility=deepcopy(checked["compatibility"]))


def load_model(packet):
    require(isinstance(packet, dict) and set(packet) == {
        "version", "original_record_b64", "predecessor_record_b64", "compatibility"}
        and packet["version"] == VERSION, "NCAAF_COMPAT_PACKET_SCHEMA")
    require(all(isinstance(packet[k], str) and len(packet[k]) <= MAX_RECORD_BYTES * 2
                for k in ("original_record_b64", "predecessor_record_b64")), "NCAAF_COMPAT_RECORD_SIZE")
    return read_model(base64.b64decode(packet["original_record_b64"], validate=True),
        base64.b64decode(packet["predecessor_record_b64"], validate=True), packet["compatibility"])
