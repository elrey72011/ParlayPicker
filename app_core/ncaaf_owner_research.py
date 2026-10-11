"""Explicit Robert-owned private research. Never independent acceptance.

Receipt processing checks an already recorded owner's review; it never performs
that human review, registers sources, fetches data or creates wagering authority.
Shared technical validators retain their strict, independent defaults.
"""
from copy import deepcopy
import hashlib
import json

from app_core import ncaaf_model_compatibility as model
from app_core import ncaaf_response_custody as custody
from app_core import ncaaf_compatible_pipeline as native
from app_core import ncaaf_prospective_chronology as chronology
from app_core import ncaaf_history as history

VERSION = "ncaaf-owner-reviewed-private-inputs-v1"
RESULT_VERSION = "ncaaf-owner-reviewed-private-result-v1"
SUBJECT_VERSION = "ncaaf-owner-reviewed-private-subject-v1"
REVIEW_VERSION = "ncaaf-owner-verification-v1"
OWNER = "Robert Velarde"
STATUS = "OWNER_REVIEWED"
SCOPE = "PRIVATE_RESEARCH"
ATTESTATION = "I personally verified the exact referenced evidence and recorded my conclusions; this is owner review, not independent acceptance."
REASONS = frozenset("""NCAAF_OWNER_REVIEW_SCHEMA NCAAF_OWNER_REVIEW_SUBJECT_CONFLICT
NCAAF_OWNER_REVIEW_INCOMPLETE NCAAF_OWNER_REVIEW_CLOCK_CONFLICT
NCAAF_PRIVATE_RESEARCH_NOT_SELECTED NCAAF_PRIVATE_RESEARCH_ONLY
NCAAF_OWNER_MODE_CONFLICT NCAAF_OWNER_RESULT_CONFLICT""".split())
require = model.require


def subject_hash(packet):
    p = packet["payload"]
    return model.digest(dict(version=SUBJECT_VERSION,
        evidence_label=p["evidence_label"], custody_packet=p["custody_packet"],
        dependency_review=p["dependency_review"]))


def evidence_subjects(packet):
    """Exact evidence references, not assertions that those facts are correct."""
    c = packet["payload"]["custody_packet"]["payload"]
    n = c["native_packet"]["payload"]
    o = n["observation"]["payload"]
    return dict(
        event_mapping=[model.digest(o["mapping_review"])],
        offer_and_settlement=[model.digest(o["quote"]), model.digest(o["source_review"]["terms_review"])],
        source_permissions=[model.digest(packet["payload"]["dependency_review"]["permissions_review"])],
        original_response_custody=[v["payload"]["body_sha256"] for v in c["response_objects"]],
        feature_provenance=[v["sha256"] for v in n["dependency_objects"]],
        model_compatibility=[model.digest(o["model"])])


class ReviewPolicy:
    """A private contract dispatched by exact version, never an acceptance map."""
    VERSION = "ncaaf-owner-custody-inputs-v1"
    RESULT_VERSION = "ncaaf-owner-custody-result-v1"
    NATIVE_VERSION = "ncaaf-owner-native-inputs-v1"
    NATIVE_RESULT_VERSION = "ncaaf-owner-native-result-v1"
    OBSERVATION_VERSION = "ncaaf-owner-prospective-observation-v1"
    OFFER_VERSION = "ncaaf-owner-offer-review-v1"
    DEPENDENCY_VERSION = "ncaaf-owner-dependency-review-v1"
    CUSTODY_VERSION = "ncaaf-owner-custody-review-v1"
    OWNER = OWNER

    def __init__(self, packet):
        p = packet["payload"]
        self.packet = packet
        self.c = p["custody_packet"]["payload"]
        self.o = self.c["native_packet"]["payload"]["observation"]["payload"]
        self.dependency_review = p["dependency_review"]
        self.approval = dict(dependency_source_review=deepcopy(self.dependency_review))

    def _owner(self, value, clock="reviewed_at"):
        return (isinstance(value, dict) and value.get("reviewer") == OWNER
                and history.timestamp(value.get(clock)) is not None)

    def check_mapping(self, *args, **kwargs):
        from app_core.ncaaf_owner_mapping import check_mapping
        return check_mapping(*args, **kwargs, review_policy=self)

    def mapping(self, value):
        return self._owner(value) and value == self.o["mapping_review"]

    def terms(self, value):
        return self._owner(value) and value == self.o["source_review"]["terms_review"]

    def permission(self, value):
        return self._owner(value) and value == self.dependency_review["permissions_review"]

    def offer(self, value):
        return self._owner(value) and value == self.o["source_review"]["owner_verification"]

    def dependency(self, value):
        return self._owner(value) and value == self.dependency_review["owner_verification"]

    def custody(self, value):
        return self._owner(value) and value == self.c["custody_admission"]["owner_verification"]


def load(packet):
    require(isinstance(packet, dict) and set(packet) == {"payload", "sha256"}
        and isinstance(packet["payload"], dict) and model.digest(packet["payload"]) == packet["sha256"],
        "NCAAF_OWNER_REVIEW_SCHEMA")
    p = packet["payload"]
    require(set(p) == {"version", "evidence_label", "custody_packet", "dependency_review", "owner_review"}
        and p["version"] == VERSION and p["evidence_label"] in {"RETAINED", "SYNTHETIC"},
        "NCAAF_OWNER_MODE_CONFLICT")
    policy = ReviewPolicy(packet)
    custody.load(p["custody_packet"], review_policy=policy)
    require(p["custody_packet"]["payload"]["evidence_label"] == p["evidence_label"], "NCAAF_OWNER_MODE_CONFLICT")
    review = p["owner_review"]
    require(isinstance(review, dict) and set(review) == {
        "version", "review_id", "reviewer", "reviewed_at", "review_status", "use_scope",
        "subject_version", "subject_sha256", "attestation", "conclusions"}
        and review["version"] == REVIEW_VERSION and review["reviewer"] == OWNER
        and review["review_status"] == STATUS and review["use_scope"] == SCOPE
        and isinstance(review["review_id"], str) and bool(review["review_id"].strip())
        and review["attestation"] == ATTESTATION, "NCAAF_OWNER_REVIEW_SCHEMA")
    require(review["subject_version"] == SUBJECT_VERSION and review["subject_sha256"] == subject_hash(packet),
        "NCAAF_OWNER_REVIEW_SUBJECT_CONFLICT")
    subjects = evidence_subjects(packet)
    conclusions = review["conclusions"]
    require(isinstance(conclusions, dict) and set(conclusions) == set(subjects), "NCAAF_OWNER_REVIEW_INCOMPLETE")
    for facet, hashes in subjects.items():
        value = conclusions[facet]
        require(isinstance(value, dict) and set(value) == {"conclusion", "evidence_sha256", "finding"}
            and value["conclusion"] == "VERIFIED" and value["evidence_sha256"] == hashes
            and isinstance(value["finding"], str) and len(value["finding"].strip()) >= 16,
            "NCAAF_OWNER_REVIEW_INCOMPLETE")
    reviewed, checkpoint = [history.timestamp(v) for v in (review["reviewed_at"], policy.o["as_of"])]
    receipts = [policy.o["mapping_review"], policy.o["source_review"]["terms_review"],
        policy.o["source_review"]["owner_verification"], policy.dependency_review["permissions_review"],
        policy.dependency_review["owner_verification"], policy.c["custody_admission"]["owner_verification"]]
    clocks = [history.timestamp(v.get("reviewed_at")) for v in receipts]
    require(reviewed is not None and checkpoint is not None and all(clocks)
        and all(t <= reviewed for t in clocks) and reviewed <= checkpoint
        and all(v.get("reviewer") == OWNER for v in receipts), "NCAAF_OWNER_REVIEW_CLOCK_CONFLICT")
    return packet


def view(packet):
    return custody.view(packet["payload"]["custody_packet"])


def result_version(packet):
    return RESULT_VERSION


def accepted_dependencies(packet, approval, at):
    load(packet)
    policy = ReviewPolicy(packet)
    reviewed = history.timestamp(packet["payload"]["owner_review"]["reviewed_at"])
    require(history.timestamp(at) is not None and reviewed < history.timestamp(at), "NCAAF_OWNER_REVIEW_CLOCK_CONFLICT")
    return custody.verify(packet["payload"]["custody_packet"], policy.approval, at, review_policy=policy)


def infer(packet, at):
    from app_core import ncaaf_pipeline_evidence as adapter
    require(adapter.private_research_selected(), "NCAAF_PRIVATE_RESEARCH_NOT_SELECTED")
    load(packet)
    policy = ReviewPolicy(packet)
    accepted_dependencies(packet, None, at)
    checked, center, win, saved = custody.infer(packet["payload"]["custody_packet"], at, review_policy=policy)
    result = dict(version=RESULT_VERSION, input_sha256=packet["sha256"],
        review_status=STATUS, use_scope=SCOPE, owner_review=deepcopy(packet["payload"]["owner_review"]),
        custody_computation=saved, raw_probability=win, inference_time=at,
        source_acceptance=False, scientific_acceptance=False, probability_calibration=False,
        wagering_authority=False, wager_action="PASS", live_stake=0)
    return checked, center, win, dict(payload=result, sha256=model.digest(result))


def inspect_result(packet, saved, at):
    load(packet)
    policy = ReviewPolicy(packet)
    accepted_dependencies(packet, None, at)
    require(isinstance(saved, dict) and set(saved) == {"payload", "sha256"}
        and model.digest(saved["payload"]) == saved["sha256"], "NCAAF_OWNER_RESULT_CONFLICT")
    r = saved["payload"]
    require(set(r) == set("version input_sha256 review_status use_scope owner_review custody_computation raw_probability inference_time source_acceptance scientific_acceptance probability_calibration wagering_authority wager_action live_stake".split())
        and r["version"] == RESULT_VERSION and r["input_sha256"] == packet["sha256"]
        and r["review_status"] == STATUS and r["use_scope"] == SCOPE
        and r["owner_review"] == packet["payload"]["owner_review"] and r["inference_time"] == at
        and all(r[k] is False for k in ("source_acceptance", "scientific_acceptance", "probability_calibration", "wagering_authority"))
        and r["wager_action"] == "PASS" and r["live_stake"] == 0, "NCAAF_OWNER_RESULT_CONFLICT")
    checked, original = custody.inspect_result(packet["payload"]["custody_packet"], r["custody_computation"], at, review_policy=policy)
    require(r["raw_probability"] == original["raw_probability"], "NCAAF_OWNER_RESULT_CONFLICT")
    return checked, deepcopy(r)


def is_private_source(source):
    """Scope detection survives captured metadata; labels do not grant trust."""
    try:
        meta = json.loads(source.get("ml_estimate_metadata", ""))
        return meta["ncaaf_inputs"]["payload"].get("version") == RESULT_VERSION
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
