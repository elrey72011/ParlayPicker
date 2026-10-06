"""Read-only private NFL dataset preparation. No fitting, registration or authority.

Self hashes preserve integrity, not source authenticity. Independent review is
supplied by the caller as a separate immutable, hash-bound decision artifact.
Proposed assignments never overwrite a governing assignment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from app_core import nfl_inference_evidence as inputs, research_replay as retained
from app_core.producer_provenance import clock
from core.probability_calibration import inspect_calibration_artifact, CALIBRATION_SCHEMA_VERSION

VERSION = "nfl-calibration-evidence-v1"
ROLES = ("development", "calibration", "validation", "holdout")
STAGES = ("raw_model", "original_blend", "ui_refresh")


def partition_barriers(observations):
    """All earlier outcomes precede later quotes, even with skipped partitions."""
    from itertools import combinations
    for earlier, later in combinations(ROLES, 2):
        left = [r for r in observations if r.get("proposed_role") == earlier]
        right = [r for r in observations if r.get("proposed_role") == later]
        if left and right:
            ends = [clock((r.get("settlement") or {}).get("payload", {}).get("available_at")) for r in left]
            starts = [clock((r.get("contract") or {}).get("offer", {}).get("source_time")) for r in right]
            if any(t is None for t in ends+starts) or max(ends) >= min(starts):
                for r in left+right:
                    r["errors"].append("PARTITION_OUTCOME_QUOTE_BARRIER")


def binding(packet, stage, refresh=None):
    """A stage is part of predictor identity, not a probability column alias."""
    if stage not in STAGES:
        raise ValueError("UNKNOWN_CALIBRATION_STAGE")
    return dict(stage=stage, predictor_id=packet["predictor_id"], target=packet["target"],
                feature_order=packet["feature_order"], runtime=packet["runtime"],
                predictor_callables=packet["predictor_callables"],
                artifact_hashes={k:v["sha256"] for k,v in packet["artifacts"].items()},
                configuration=packet["configuration"],
                blend=({k:packet["blend"][k] for k in ("consumed", "pipeline_sha256", "probability_semantics")}
                       if stage != "raw_model" else None),
                refresh=({k:refresh[k] for k in ("consumed", "probability_semantics")}
                         | {"artifact_sha256":refresh["artifact"]["sha256"]}
                         if stage == "ui_refresh" and refresh else None))


def read_review(path, expected_sha256):
    """Pin the independently supplied review; never accept a review embedded in a row."""
    raw = Path(path).read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("REVIEW_ARTIFACT_INTEGRITY")
    return json.loads(raw)


def _assessment(row, stage, expected, review, settlement, assignment):
    errors, gaps = [], []
    diagnostic = inputs.diagnose(row)
    errors.extend(diagnostic["errors"])
    gaps.extend(diagnostic["unknown"])
    item = json.loads(row.get("ml_estimate_metadata") or "{}")
    p = item.get("nfl_inputs", {}).get("payload")
    if not p or p.get("capture_status") == "FAILED":
        return dict(errors=errors, gaps=gaps or ["ORIGINAL_INPUTS_UNAVAILABLE"], packet=p)
    refreshes = item.get("nfl_ui_reblends", [])
    refresh = refreshes[-1]["payload"] if refreshes else None
    identity = binding(p, stage, refresh)
    if identity != expected:
        errors.append("PREDICTOR_PIPELINE_CONFIGURATION_MISMATCH")
    if stage == "ui_refresh" and refresh is None:
        gaps.append("UI_REFRESH_NOT_RECORDED")
    if diagnostic["status"] == "COMPLETE":
        try:
            inputs.replay(row)
        except ValueError as exc:
            errors.append("NUMERIC_REPLAY:" + str(exc))
    contract = p.get("event_offer") or {}
    event, offer = contract.get("event", {}), contract.get("offer", {})
    source_hash = inputs.digest(contract)
    crosswalk = (review or {}).get("canonical_game", {})
    if crosswalk.get("event") != event or not crosswalk.get("game_id"):
        gaps.append("CANONICAL_GAME_MAPPING_NOT_REVIEWED")
    review_errors = []
    if not review or review.get("contract_sha256") != source_hash:
        gaps.append("INDEPENDENT_SOURCE_ADMISSIBILITY_REVIEW_MISSING")
    else:
        for name in ("exact_offer", "product_rules", "quote_clock", "feature_rights", "quote_rights", "settlement_rights"):
            if review.get(name) != "ACCEPTED":
                review_errors.append("SOURCE_ADMISSIBILITY:" + name)
        gaps.extend(review_errors)
        if not review.get("operator") or not review.get("product") or not review.get("review_id"):
            gaps.append("SOURCE_PRODUCT_REVIEW_ID_MISSING")
    if not settlement:
        gaps.append("VERIFIED_SETTLEMENT_MISSING")
    else:
        # This is an independently reviewed exact original settlement document,
        # not a final-score reconstruction or a candidate's asserted outcome.
        payload = settlement.get("payload", {})
        if inputs.digest(payload) != settlement.get("sha256"):
            errors.append("SETTLEMENT_INTEGRITY")
        if payload.get("contract_sha256") != source_hash:
            errors.append("SETTLEMENT_EXACT_OFFER_MISMATCH")
        if not review or review.get("settlement_sha256") != settlement.get("sha256"):
            gaps.append("SETTLEMENT_REVIEW_NOT_BOUND")
        original = payload.get("original")
        if not isinstance(original, dict) or not original:
            gaps.append("ORIGINAL_SETTLEMENT_DOCUMENT_MISSING")
        elif (original.get("event") != event or original.get("offer") != offer
              or original.get("outcome") != payload.get("outcome")
              or original.get("available_at") != payload.get("available_at")):
            errors.append("SETTLEMENT_ORIGINAL_DOCUMENT_CONFLICT")
        if payload.get("outcome") not in {"WIN", "LOSS"}:
            gaps.append("SETTLEMENT_NOT_DECIDED")
        at, start = clock(payload.get("available_at")), clock(event.get("start"))
        if at is None or start is None or at <= start:
            errors.append("SETTLEMENT_AVAILABILITY_INVALID")
    # Original source/model availability and an exclusion of this game from model
    # development must be demonstrated even for historical calibration data.
    lineage = (review or {}).get("out_of_sample", {})
    if lineage.get("packet_sha256") != inputs.digest(p) or lineage.get("game_excluded_from_development") is not True:
        gaps.append("OUT_OF_SAMPLE_LINEAGE_NOT_DEMONSTRATED")
    if clock(lineage.get("predictor_available_at")) is None or clock(p.get("inference_time")) is None:
        gaps.append("PREDICTOR_AVAILABILITY_NOT_DEMONSTRATED")
    elif clock(lineage["predictor_available_at"]) > clock(p["inference_time"]):
        errors.append("FUTURE_PREDICTOR")
    return dict(errors=errors, gaps=gaps, packet=p, refresh=refresh,
                canonical_game=crosswalk.get("game_id") if crosswalk.get("event") == event else None,
                contract=contract, identity=identity, assignment=assignment)


def build_dataset(export_ids, *, path, stage, expected_binding, reviews=None,
                  settlements=None, proposed_assignments=None, approved_assignments=None,
                  calibration_path=None):
    """Verify retained capture/export sources and propose, never grant, scientific roles.

    Reviews and settlements are keyed by digest of the exact original contract.
    Assignments are keyed by named provider game, with immutable input packets
    additionally bound in the independent review. Missing reviews stay gaps.
    """
    if stage not in STAGES or expected_binding.get("stage") != stage:
        raise ValueError("EXPLICIT_CALIBRATION_BINDING_REQUIRED")
    reviews, settlements = reviews or {}, settlements or {}
    proposed_assignments = proposed_assignments or {}
    approved_assignments = approved_assignments or {}
    observations, seen_sources = [], set()
    for export_id in sorted(set(export_ids)):
        export, sources = retained.read_export(export_id, path=path)
        if export["source_boundary"] != "RETAINED":
            raise ValueError("CAPTURE_SOURCE_NOT_RETAINED")
        for sid, source in sources.items():
            # Repeated exports never create repeated observations.
            if sid in seen_sources:
                continue
            seen_sources.add(sid)
            frame = retained.frame_from_payload(source["original"]["producer"])
            for row in frame.to_dict("records") if frame is not None else []:
                if str(row.get("league", row.get("League", ""))).upper() != "NFL":
                    continue
                try:
                    item = json.loads(row.get("ml_estimate_metadata") or "{}")
                    p = item.get("nfl_inputs", {}).get("payload", {})
                    contract = p.get("event_offer") or {}
                    event = contract.get("event", {})
                    key = event.get("provider_namespace", "UNKNOWN") + ":" + event.get("provider_event_id", "UNKNOWN")
                    h = inputs.digest(contract)
                    assignment = proposed_assignments.get(key, {})
                    assessed = _assessment(row, stage, expected_binding, reviews.get(h), settlements.get(h), assignment)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    observations.append(dict(snapshot_id=sid, export_id=export_id, game="UNKNOWN",
                        errors=["INPUT_SCHEMA:"+type(exc).__name__], gaps=[], proposed_role=None, approved_assignment=None))
                    continue
                errors, gaps = assessed["errors"], assessed["gaps"]
                role = assignment.get("role")
                if role not in ROLES:
                    gaps.append("PROPOSED_ROLE_MISSING")
                existing = approved_assignments.get(key)
                if existing and existing.get("role") != role:
                    errors.append("APPROVED_ROLE_CONFLICT")
                if (role in {"validation", "holdout"}
                    and (assignment.get("outcomes_uninspected") is not True
                         or (reviews.get(h) or {}).get("outcomes_previously_inspected") is not False
                         or assignment.get("custodian_seal") is None
                         or clock(assignment.get("sealed_at")) is None
                         or clock(event.get("start")) is None
                         or clock(assignment.get("sealed_at")) >= clock(event.get("start")))):
                    errors.append("EVALUATION_ROLE_CONTAMINATION")
                if assignment.get("selected_contract_sha256") != h:
                    gaps.append("OFFER_NOT_IN_PROPOSED_SAMPLE")
                offer = contract.get("offer", {})
                fresh = assessed.get("refresh")
                raw = p.get("raw_probability")
                blended = (p.get("blend") or {}).get("probability")
                ui = fresh.get("probability") if fresh else None
                probability = {"raw_model":raw, "original_blend":blended, "ui_refresh":ui}[stage]
                group = assignment.get("groups", {})
                for name in ("season", "week"):
                    if group.get(name) is None:
                        gaps.append("DEPENDENCE_GROUP_MISSING:"+name)
                observations.append(dict(snapshot_id=sid, export_id=export_id, game=key,
                    canonical_game=assessed.get("canonical_game"),
                    original_source_sha256=inputs.digest(source), contract=contract,
                    packet=p, ui_refresh_receipts=item.get("nfl_ui_reblends", []),
                    raw_model_probability=raw, original_blended_probability=blended,
                    validated_ui_refresh_probability=ui if not errors else None,
                    calibration_input=probability, calibration_binding=assessed.get("identity"),
                    source_admissibility_review=reviews.get(h), settlement=settlements.get(h),
                    proposed_role=role, proposed_assignment=assignment, approved_assignment=existing,
                    groups=dict(group, home=event.get("home"), away=event.get("away"),
                        selected_side=offer.get("side"), signed_line=offer.get("line"),
                        line_sign="positive" if (offer.get("line") or 0)>0 else "negative",
                        kickoff=event.get("start")), errors=errors, gaps=gaps))
    games = defaultdict(list)
    for row in observations:
        games[row.get("canonical_game") or row["game"]].append(row)
    duplicates = 0
    for game, members in games.items():
        roles = {row.get("proposed_role") for row in members}
        if len(roles) > 1:
            for row in members:
                row["errors"].append("GAME_CROSSES_ROLES")
        members.sort(key=lambda row: ("OFFER_NOT_IN_PROPOSED_SAMPLE" in row["gaps"],
            (row.get("contract") or {}).get("inference_time", ""), row["snapshot_id"]))
        for row in members[1:]:
            duplicates += 1
            row["gaps"].append("DUPLICATE_GAME_OFFER")
    # Chronological boundaries use outcome availability, not merely kickoff order.
    partition_barriers(observations)
    for row in observations:
        row["errors"] = sorted(set(row["errors"]))
        row["gaps"] = sorted(set(row["gaps"]))
        row["admissibility"] = "REJECTED" if row["errors"] else "INCOMPLETE" if row["gaps"] else "SOFTWARE_ADMISSIBLE_PROPOSAL"
    accepted = [r for r in observations if r["admissibility"] == "SOFTWARE_ADMISSIBLE_PROPOSAL"]
    report = dict(version=VERSION, observations=len(observations),
        provider_games=len({r["game"] for r in observations}),
        independent_games=len({r["canonical_game"] for r in observations if r.get("canonical_game")}),
        duplicate_offers=duplicates, admissibility=dict(Counter(r["admissibility"] for r in observations)),
        proposed_role_counts={role:len({r["canonical_game"] for r in accepted if r["proposed_role"] == role}) for role in ROLES},
        approved_assignments=approved_assignments, proposed_assignments=proposed_assignments,
        exclusion_counts=dict(Counter(code for r in observations for code in r["errors"]+r["gaps"])),
        calibration_input_binding=expected_binding, calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_artifact=inspect_calibration_artifact(calibration_path),
        fitted=False, scientific_acceptance=False, wagering_authority=False,
        status="PROPOSAL_ONLY")
    return dict(version=VERSION, report=report, observations=observations)


def export_private(dataset):
    """Owner-only bytes and digest; callers must not put this in public packages."""
    raw = retained.encode(dataset).encode()
    import hashlib
    return raw, hashlib.sha256(raw).hexdigest()
