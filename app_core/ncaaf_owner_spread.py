"""Explicit Robert-owned private research. Never independent acceptance.

Receipt processing checks an already recorded owner's review; it never performs
that human review, registers sources, fetches data or creates wagering authority.
Shared technical validators retain their strict, independent defaults.
"""
from copy import deepcopy
from functools import wraps
from app_core import ncaaf_spread_terms_projection as projection
import hashlib
import json

from app_core import ncaaf_model_compatibility as model
from app_core import ncaaf_response_custody as custody
from app_core import ncaaf_compatible_pipeline as native
from app_core import ncaaf_prospective_chronology as chronology
from app_core import ncaaf_history as history
from app_core import ncaaf_staged_evidence as staged, ncaaf_future_target as aliases

VERSION = "ncaaf-owner-integer-inputs-v1"
HALF_VERSION = "ncaaf-owner-named-half-point-inputs-v1"
INPUT_VERSIONS = {VERSION, HALF_VERSION}
RESULT_VERSIONS = {"ncaaf-owner-integer-result-v1", "ncaaf-owner-named-half-point-result-v1"}
RESULT_VERSION = "ncaaf-owner-integer-result-v1"
SUBJECT_VERSION = "ncaaf-owner-spread-subject-v4"
REVIEW_VERSION = "ncaaf-owner-spread-verification-v4"
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
        dependency_review=p["dependency_review"], alias_review=p["alias_review"], staged_binding=p["staged_binding"]))


def evidence_subjects(packet):
    """Exact evidence references, not assertions that those facts are correct."""
    p = packet["payload"]
    c = p["custody_packet"]["payload"]
    n = c["native_packet"]["payload"]
    o = n["observation"]["payload"]
    return dict(
        staged_evidence=[p["staged_binding"]["sha256"]],
        reviewed_aliases=[model.digest(p["alias_review"])],
        event_mapping=[model.digest(o["mapping_review"])],
        offer_and_settlement=[model.digest(o["quote"]), model.digest(o["source_review"]["terms_review"])],
        source_permissions=[model.digest(packet["payload"]["dependency_review"]["permissions_review"])],
        original_response_custody=[v["payload"]["body_sha256"] for v in c["response_objects"]],
        feature_provenance=[v["sha256"] for v in n["dependency_objects"]],
        model_compatibility=[model.digest(o["model"])])


class ReviewPolicy:
    """A private contract dispatched by exact version, never an acceptance map."""
    VERSION = "ncaaf-owner-spread-custody-inputs-v4"
    RESULT_VERSION = "ncaaf-owner-spread-custody-result-v4"
    NATIVE_VERSION = "ncaaf-owner-spread-native-inputs-v4"
    NATIVE_RESULT_VERSION = "ncaaf-owner-spread-native-result-v4"
    OBSERVATION_VERSION = "ncaaf-owner-spread-prospective-observation-v4"
    OFFER_VERSION = "ncaaf-owner-spread-offer-review-v4"
    DEPENDENCY_VERSION = "ncaaf-owner-spread-dependency-review-v4"
    CUSTODY_VERSION = "ncaaf-owner-spread-custody-review-v4"
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
        p=self.packet['payload']
        if p['staged_binding']['payload'].get('version')=='ncaaf-reviewed-staged-evidence-v3':
            from app_core.ncaaf_reviewed_target import check_mapping
            return check_mapping(*args, **kwargs, review_policy=self)
        from app_core.ncaaf_future_target import check_mapping
        return check_mapping(*args, **kwargs, review_policy=self)

    def target_contract(self,kind,line):
        from app_core.ncaaf_spread_mass import target_contract
        return target_contract(kind,line,integer=self.packet['payload']['version']==VERSION)

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
    require(set(p) == {"version", "evidence_label", "custody_packet", "dependency_review", "owner_review", "alias_review", "staged_binding"}
        and p["version"] in INPUT_VERSIONS and p["evidence_label"] in {"RETAINED", "SYNTHETIC"},
        "NCAAF_OWNER_MODE_CONFLICT")
    policy = ReviewPolicy(packet)
    custody.load(p["custody_packet"], review_policy=policy)
    require(p["custody_packet"]["payload"]["evidence_label"] == p["evidence_label"], "NCAAF_OWNER_MODE_CONFLICT")
    staged.validate_binding(p["staged_binding"], packet, at=policy.o["as_of"])
    policy.check_mapping(policy.o['event'],policy.o['schedule'],policy.o['crosswalk'],policy.o['mapping_review'],as_of=history.timestamp(policy.o['as_of']))
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
    return RESULT_VERSION if packet["payload"]["version"]==VERSION else "ncaaf-owner-named-half-point-result-v1"


def accepted_dependencies(packet, approval, at):
    load(packet)
    policy = ReviewPolicy(packet)
    reviewed = history.timestamp(packet["payload"]["owner_review"]["reviewed_at"])
    require(history.timestamp(at) is not None and reviewed < history.timestamp(at), "NCAAF_OWNER_REVIEW_CLOCK_CONFLICT")
    staged.validate_binding(packet["payload"]["staged_binding"], packet, at=at)
    return custody.verify(packet["payload"]["custody_packet"], policy.approval, at, review_policy=policy)


def infer(packet, at):
    from app_core import ncaaf_pipeline_evidence as adapter
    require(adapter.private_research_selected(), "NCAAF_PRIVATE_RESEARCH_NOT_SELECTED")
    load(packet)
    policy = ReviewPolicy(packet)
    accepted_dependencies(packet, None, at)
    from app_core import ncaaf_spread_execution as execution, ncaaf_spread_mass as mass
    execution.begin(packet,at)
    checked, center, win, saved = custody.infer(packet["payload"]["custody_packet"], at, review_policy=policy)
    from app_core.ncaaf_pipeline_evidence import generated_time
    finished=generated_time()
    try:
        accepted_dependencies(packet,None,finished)
        native_receipt=saved['payload']['native_computation']['payload']
        masses=mass.inspect_receipt(native_receipt,policy.o['quote']['market_type'])
        proof=execution.complete(packet,saved,at,finished)
    except (ValueError,TypeError,KeyError):
        execution.unavailable_computation(packet,saved,at,finished)
        raise
    result = dict(version=result_version(packet), input_sha256=packet["sha256"],
        review_status=STATUS, use_scope=SCOPE, owner_review=deepcopy(packet["payload"]["owner_review"]),
        custody_computation=saved, raw_probability=win, inference_time=at,
        outcome_probabilities=masses,conditional_no_push=mass.conditional_no_push(masses),
        probability_semantics='unconditional_cover_push_loss',execution_receipt=proof,
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
    require(set(r) == set("version input_sha256 review_status use_scope owner_review custody_computation raw_probability inference_time outcome_probabilities conditional_no_push probability_semantics execution_receipt source_acceptance scientific_acceptance probability_calibration wagering_authority wager_action live_stake".split())
        and r["version"] == result_version(packet) and r["input_sha256"] == packet["sha256"]
        and r["review_status"] == STATUS and r["use_scope"] == SCOPE
        and r["owner_review"] == packet["payload"]["owner_review"] and r["inference_time"] == at
        and all(r[k] is False for k in ("source_acceptance", "scientific_acceptance", "probability_calibration", "wagering_authority"))
        and r["wager_action"] == "PASS" and r["live_stake"] == 0, "NCAAF_OWNER_RESULT_CONFLICT")
    checked, original = custody.inspect_result(packet["payload"]["custody_packet"], r["custody_computation"], at, review_policy=policy)
    require(r["raw_probability"] == original["raw_probability"], "NCAAF_OWNER_RESULT_CONFLICT")
    from app_core import ncaaf_spread_mass as mass, ncaaf_spread_execution as execution
    original_mass=mass.inspect_receipt(original['native_computation']['payload'],policy.o['quote']['market_type'])
    require(r['outcome_probabilities']==original_mass and r['conditional_no_push']==mass.conditional_no_push(original_mass)
        and r['probability_semantics']=='unconditional_cover_push_loss','NCAAF_SPREAD_MASS_CONFLICT')
    execution.inspect(packet,r['custody_computation'],r['execution_receipt'],at)
    accepted_dependencies(packet,None,r['execution_receipt']['completed_at'])
    return checked, deepcopy(r)


def is_private_source(source):
    """Scope detection survives captured metadata; labels do not grant trust."""
    try:
        meta = json.loads(source.get("ml_estimate_metadata", ""))
        return meta["ncaaf_inputs"]["payload"].get("version") in RESULT_VERSIONS
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def admitted_analysis(packet, source, *, inventory=None, authorization=None, custody_root=None):
    """Explicit already-reviewed source through the actual market caller.

    Like the original pilot's selected-market entry, this does not ask the live
    multi-book slate normalizer to choose a market or invent an orientation.
    Original raw quotes and every reader gate still bind the selected source.
    """
    import pandas as pd
    from app_core import ncaaf_pipeline_evidence as adapter
    from app_core.market_probability_model import predict_market_probabilities
    load(packet)
    require(adapter._offer_matches(packet, source), 'NCAAF_EVENT_OFFER_CONFLICT')
    frame = pd.DataFrame([deepcopy(source)])
    start = history.timestamp(source.get('game_start_utc'))
    require(start is not None, 'NCAAF_EVENT_OFFER_CONFLICT')
    from app_core.slate_coverage import ET, build_coverage, native_ncaaf
    day = start.astimezone(ET).date().isoformat()
    frame['date'] = pd.Timestamp(start); frame['game_date'] = day
    frame.attrs['ncaaf_schedule'] = deepcopy(inventory)
    from contextlib import nullcontext
    from app_core import ncaaf_spread_execution as execution
    active=execution.ACTIVE.get()
    scope=nullcontext() if active is not None else execution.execution([packet],authorization,root=custody_root,
        subject=packet['sha256'],reviewed_at=packet['payload']['owner_review']['reviewed_at'],at=adapter.generated_time())
    with scope, adapter.selected([packet], private_research=True):
        predictions = predict_market_probabilities(frame)
    for field in predictions: frame[field] = predictions[field]
    frame = adapter.finish(frame)
    at = json.loads(frame.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']['inference_time']
    frame.attrs['ncaaf_schedule'] = deepcopy(inventory)
    frame.attrs['slate_coverage'] = build_coverage([native_ncaaf(inventory, day)] if inventory else [], selected_date=day,
        as_of=at, run_id='ncaaf-owner-staged-'+model.digest(dict(event=source.get('matchup_id'), at=at)), candidates=frame, leagues=['NCAAF'])
    return frame

# Only this explicit successor selects reviewed scope interpretation. This
# creates no acceptance and never changes the original response or v1 context.
def _with_scope(fn):
    @wraps(fn)
    def checked(packet,*args,**kwargs):
        load(packet)
        with projection.selected(packet):
            return fn(packet,*args,**kwargs)
    return checked

accepted_dependencies=_with_scope(accepted_dependencies)
infer=_with_scope(infer)
inspect_result=_with_scope(inspect_result)
