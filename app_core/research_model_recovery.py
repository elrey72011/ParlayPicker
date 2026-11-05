"""Explicitly reviewed research-only runtime transition; no generic re-freeze."""
from copy import deepcopy
from app_core import ncaaf_prospective as ncaaf
from app_core import ncaaf_prospective_store as store

# git diff 61838bc2..0d36bd74 changes only NFL aliases in core/team_mapper.py.
# All four NCAAF model, feature and identity modules are byte-identical.
SOURCE_RUNTIME = "bd837b0b5f9d175c235002b43a9c862df47c205582432da42114b8ebcb220e57"
TARGET_RUNTIME = "d5acdd2cd5dc9264eaf9a322fe1b7498033b3c6273ad31396154fc644486eb23"
ARTIFACT = "9334f2a4904e9cdb88f0b50563c058b846647d876a72e46a0548035a27c2d25e"

# Second reviewed transition (roadmap D15). Since 2026-10-03 every NCAAF capture
# failed with stale_frozen_model because two later commits changed files inside
# ncaaf.runtime_hash, neither touching the model, features or history code:
#   93a7f236 (d5acdd2c -> 6f314e5f) adds identity aliases in ncaaf_identity.py only:
#            McNeese, LSU / Louisiana State and the "gardnerwebb" spelling.
#   1ccf0a28 (6f314e5f -> 8283b60a) changes NHL New York names in core/team_mapper.py only.
# ncaaf_prospective.py, ncaaf_research.py and ncaaf_history.py are byte-identical.
# The artifact stays research-only (production_eligible False) and keeps its hash.
ALIAS_RUNTIME = "8283b60a3e1e3310719526ef60320a45560de4b0a04c1faedeca3218102936c0"
REVIEWED_TRANSITIONS = {
    (SOURCE_RUNTIME, TARGET_RUNTIME): "reviewed_NFL_alias_only_change_0d36bd74",
    (TARGET_RUNTIME, ALIAS_RUNTIME): "reviewed_NCAAF_identity_alias_and_NHL_mapper_only_93a7f236_1ccf0a28",
    (SOURCE_RUNTIME, ALIAS_RUNTIME): "reviewed_NFL_alias_0d36bd74_then_NCAAF_identity_alias_and_NHL_mapper_93a7f236_1ccf0a28",
}


def recover_ncaaf(model, path):
    current = ncaaf.runtime_hash()
    data = model["data"]
    if data.get("runtime_hash") == current:
        return model
    # Unknown code/model changes still require an independently reviewed freeze.
    artifact = data.get("artifact", {})
    reason = REVIEWED_TRANSITIONS.get((data.get("runtime_hash"), current))
    if (reason is None
            or data.get("artifact_hash") != ARTIFACT or ncaaf.digest(artifact) != ARTIFACT
            or artifact.get("protocol", {}).get("production_eligible") is not False):
        raise ValueError("stale_frozen_model")
    updated = deepcopy(data)
    updated["runtime_hash"] = current
    updated["recovery"] = {"source_model_id": model["id"], "source_runtime": data["runtime_hash"],
                           "reason": reason,
                           "production_eligible": False}
    # Append with the real current timestamp. Historical captures keep their IDs.
    existing = next((r for r in store.records(path) if r["kind"] == "model" and r["data"] == updated), None)
    if existing:
        return existing
    key = store.save("model", updated, path)
    return next(r for r in store.records(path) if r["id"] == key)
