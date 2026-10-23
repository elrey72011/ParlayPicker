"""Explicit, fresh native NCAAF research inputs for the normal analysis caller.

No store connector, provider call, fitting, restore, registration or authority
consumer exists here. Historical packets are never numerically replayed. The
original capture and its missing inference clock remain separate from a new,
explicitly requested inference on still-fresh prospective inputs.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform

from app_core import ncaaf_research_contract as contract, ncaaf_research as research
from app_core import producer_provenance as producer
from app_core import ncaaf_compatible_pipeline as compatible
from app_core import ncaaf_response_custody as custody
from app_core import ncaaf_owner_research as owner
from app_core import ncaaf_owner_terms_research as owner_terms
from app_core import ncaaf_owner_staged_research as owner_staged
from app_core import ncaaf_owner_spread as owner_spread
from app_core.research_estimate_trace import encode, fact, origin_metadata, generated_time

VERSION = "ncaaf-normal-pipeline-inputs-v1"
MAX_BYTES = 8 * 1024 * 1024
MAX_PACKETS = 4
MAX_SELECTED_BYTES = 16 * 1024 * 1024
_PRIVATE = ContextVar("ncaaf_owner_private_selected", default=False)
_SELECTED = ContextVar("ncaaf_explicit_research_packets", default=())
# Independently accepted exact packet/review and PUBLIC derived-output permission.
# Owner upload/selection never writes this catalog. It is empty in production.
ACCEPTED_PACKETS = {}
INPUT_VERSIONS = owner_spread.INPUT_VERSIONS | compatible.INPUT_VERSIONS | {custody.VERSION, owner.VERSION, owner_terms.VERSION, owner_staged.VERSION}
RESULT_VERSIONS = owner_spread.RESULT_VERSIONS | compatible.RESULT_VERSIONS | {custody.RESULT_VERSION, owner.RESULT_VERSION, owner_terms.RESULT_VERSION, owner_staged.RESULT_VERSION}


def _reader(packet):
    if packet['payload']['version'] in owner_spread.INPUT_VERSIONS: return owner_spread
    if packet['payload']['version'] == owner_staged.VERSION: return owner_staged
    if packet['payload']['version'] in {owner_terms.VERSION,owner_staged.VERSION}: return owner_terms
    if packet['payload']['version'] == owner.VERSION: return owner
    return custody if packet['payload']['version'] == custody.VERSION else compatible
PUBLIC_REASONS = frozenset("""NCAAF_ORIGINAL_PACKET_NOT_SELECTED NCAAF_EXACT_OFFER_NOT_SELECTED
NCAAF_PACKET_AMBIGUOUS NCAAF_PACKET_INTEGRITY NCAAF_PACKET_SCHEMA NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED
NCAAF_SOURCE_REVIEW_NOT_ACCEPTED NCAAF_PUBLIC_DERIVED_RIGHTS_UNAVAILABLE NCAAF_INFERENCE_FAILED
NCAAF_QUOTE_INFERENCE_START_CLOCK_CONFLICT NCAAF_EVENT_OFFER_CONFLICT NCAAF_ORIGINAL_FEATURES_UNAVAILABLE
NCAAF_FROZEN_RUNTIME_MISMATCH NCAAF_ARTIFACT_READER_MISMATCH NCAAF_ARTIFACT_TARGET_CONFLICT
NCAAF_ARTIFACT_INTEGRITY NCAAF_MODEL_RECORD_MISSING NCAAF_FUTURE_OR_STALE_DEPENDENCY
NCAAF_PERIOD_OR_SETTLEMENT_MISSING NCAAF_SOURCE_PAYOFF_OR_RIGHTS_UNSUPPORTED
NCAAF_EXACT_SOURCE_REVIEW_MISSING_OR_CONFLICT NCAAF_SOURCE_REVIEW_CLOCK_CONFLICT
NCAAF_CANONICAL_EVENT_CONFLICT NCAAF_EVENT_AMBIGUOUS NCAAF_EXACT_OFFER_AMBIGUOUS_OR_MISSING
NCAAF_ALTERED_ORDERED_FEATURES NCAAF_EVALUATED_HOLDOUT_CONTAMINATION NCAAF_RUNTIME_BINDING_CONFLICT
NCAAF_PROBABILITY_BINDING_CONFLICT NCAAF_AUTHORITY_FORBIDDEN NCAAF_SCHEDULE_RECEIPT_MISSING
NCAAF_SCHEDULE_RECEIPT_CONFLICT""".split()) | compatible.REASONS | custody.REASONS | owner.REASONS


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(raw, *, owner_upload=False):
    """Bounded original JSON/hash inspection only; no numerical reader or storage."""
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_BYTES, "NCAAF_PACKET_SCHEMA")
    packet = json.loads(raw)
    require(isinstance(packet, dict) and set(packet) == {"payload", "sha256"}, "NCAAF_PACKET_SCHEMA")
    p = packet["payload"]
    require(isinstance(p, dict) and p.get("version") in {contract.VERSION, *INPUT_VERSIONS}, "NCAAF_PACKET_SCHEMA")
    require(digest(p) == packet["sha256"], "NCAAF_PACKET_INTEGRITY")
    require(p.get("evidence_label") in {"RETAINED", "SYNTHETIC"}, "NCAAF_PACKET_SCHEMA")
    if owner_upload:
        require(p["evidence_label"] == "RETAINED", "NCAAF_PACKET_SCHEMA")
    if p["version"] in INPUT_VERSIONS:
        _reader(packet).load(packet)
    return packet


@contextmanager
def selected(packets=(), *, private_research=False):
    require(isinstance(packets, (list, tuple)) and len(packets) <= MAX_PACKETS, "NCAAF_PACKET_SCHEMA")
    packets = tuple(load(encode(p).encode()) for p in packets)
    require(len(packets) <= MAX_PACKETS and sum(len(encode(p).encode()) for p in packets) <= MAX_SELECTED_BYTES,
            "NCAAF_PACKET_SCHEMA")
    require(type(private_research) is bool, "NCAAF_OWNER_MODE_CONFLICT")
    require(private_research or not any(p["payload"]["version"] in owner_spread.INPUT_VERSIONS | {owner.VERSION,owner_terms.VERSION,owner_staged.VERSION} for p in packets), "NCAAF_PRIVATE_RESEARCH_NOT_SELECTED")
    private_token = _PRIVATE.set(private_research)
    token = _SELECTED.set(deepcopy(packets))
    try:
        yield
    finally:
        _SELECTED.reset(token)
        _PRIVATE.reset(private_token)


def private_research_selected():
    return _PRIVATE.get()


def selection_requested():
    return bool(_SELECTED.get())


def reader_binding():
    root = Path(__file__).resolve().parents[1]
    from app_core.ncaaf_reviewed_target import source_hash
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
        platform=platform.platform(), numpy=research.np.__version__,
        artifacts={name: source_hash(name,historical=True)
                   for name in ("app_core/ncaaf_pipeline_evidence.py", "app_core/market_probability_model.py",
                       "app_core/ncaaf_research_contract.py", "app_core/producer_provenance.py",
                       "app_core/research_estimate_trace.py", "core/streamlit_pipeline.py")})


def accepted(packet):
    if packet['payload']['version'] in owner_spread.INPUT_VERSIONS:
        owner_spread.load(packet)
        return deepcopy(owner_spread.ReviewPolicy(packet).approval)
    if packet["payload"]["version"] == owner_staged.VERSION:
        owner_staged.load(packet)
        return deepcopy(owner_staged.ReviewPolicy(packet).approval)
    if packet["payload"]["version"] == owner_terms.VERSION:
        owner_terms.load(packet)
        return deepcopy(owner_terms.ReviewPolicy(packet).approval)
    if packet["payload"]["version"] == owner.VERSION:
        owner.load(packet)
        return deepcopy(owner.ReviewPolicy(packet).approval)
    approval = ACCEPTED_PACKETS.get(packet["sha256"])
    require(isinstance(approval, dict) and approval.get("source_review_sha256") ==
            digest(_transport(packet)["source_review"]), "NCAAF_SOURCE_REVIEW_NOT_ACCEPTED")
    require(approval.get("public_derived_output") == "permitted", "NCAAF_PUBLIC_DERIVED_RIGHTS_UNAVAILABLE")
    return deepcopy(approval)


def _projection(packet):
    return owner_spread.projection if packet['payload']['version'] in owner_spread.INPUT_VERSIONS else owner_terms.projection


def _transport(packet):
    return _reader(packet).view(packet) if packet["payload"]["version"] in INPUT_VERSIONS else packet["payload"]


def _offer_matches(packet, source):
    if producer.text(source.get("league") or source.get("League")).upper() != "NCAAF":
        return False
    p = _transport(packet)
    q = p["original_quote"]
    chosen = p["selection"]
    kind = producer.text(source.get("market_type"))
    line = producer.number(source.get("total_line" if kind.startswith("total") else "spread_line"))
    if (kind != chosen["kind"]
            or line != chosen["line"] or producer.number(source.get("odds_american")) != chosen["price"]):
        return False
    matches = producer._matches(source)
    if packet['payload']['version'] in owner_spread.INPUT_VERSIONS:
        matches = owner_spread.projection.matches(packet,matches)
    elif packet['payload']['version'] in {owner_terms.VERSION,owner_staged.VERSION}:
        matches = owner_terms.projection.matches(packet,matches)
    # The normal caller transports the original quote before it promotes its
    # provider aliases. Use that exact receipt; reject conflicting aliases.
    if any(producer.text(source.get(k)) and producer.text(source.get(k)) != q.get(k)
           for k in ("provider_event_id", "provider_namespace")):
        return False
    return len(matches) == 1 and matches[0].get("provider_event_id") == chosen["event_id"] and all(matches[0].get(k) == q.get(k) for k in
        ("book", "market_type", "point", "price", "recorded_at", "provider_event_id",
         "provider_namespace", "event_home_team", "event_away_team", "event_start_utc",
         "period", "period_source", "rules", "rules_source"))


def _fresh(packet, at):
    p = packet["payload"]
    q = p["original_quote"]
    quote, inference, start = [contract.history.timestamp(v) for v in
        (q["recorded_at"], at, q["event_start_utc"])]
    require(all((quote, inference, start)) and 0 <= (inference-quote).total_seconds() <= 900
            and inference < start, "NCAAF_QUOTE_INFERENCE_START_CLOCK_CONFLICT")
    finish = contract.history.timestamp(p["original_capture_finished_at"])
    require(finish is not None and finish <= inference, "NCAAF_QUOTE_INFERENCE_START_CLOCK_CONFLICT")
    require(p["original_inference_time"] is None and p["historical_publication_time_verified"] is False,
            "NCAAF_PACKET_SCHEMA")
    return inference


def _checked(packet, source, at):
    """Only called by a new, explicitly selected prospective analysis inference."""
    load(encode(packet).encode())
    require(_offer_matches(packet, source), "NCAAF_EVENT_OFFER_CONFLICT")
    p = packet["payload"]
    require(abs(p["selection"]["line"] % 1) == .5, "NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED")
    now = _fresh(packet, at)  # Reject historical/stale packets BEFORE any numerical reader.
    accepted(packet)
    rebuilt = contract.export_observation(p["records"], source_review=p["source_review"],
        evidence_label=p["evidence_label"], **p["selection"])
    require(rebuilt == packet, "NCAAF_PACKET_INTEGRITY")
    c, m, e, q, f, target = contract._observation(contract._records(p["records"]), **p["selection"])
    contract._state(c["data"]["inputs"], now)
    # Native v1's finish clock is preserved. This is a NEW inference clock.
    require(contract.history.timestamp(c["created_at"]) <= now, "NCAAF_QUOTE_INFERENCE_START_CLOCK_CONFLICT")
    return m, f, target


def board_schedule(source, packet, inventory, at):
    """Retain the already fetched selected ESPN event; no requests or store read."""
    if not producer.text(source.get("matchup_id")).startswith("espn:college-football:"):
        return None
    require(isinstance(inventory, dict), "NCAAF_SCHEDULE_RECEIPT_MISSING")
    from app_core.ncaaf_schedule import match_event
    q = _transport(packet)["original_quote"]
    event, status = match_event(dict(home_team=q["event_home_team"], away_team=q["event_away_team"],
        commence_time=q["event_start_utc"]), inventory)
    require(status == "MATCHED" and event["schedule_event_id"] == source.get("matchup_id"), "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
    observed, inference = [contract.history.timestamp(t) for t in (inventory.get("observed_at"), at)]
    require(observed is not None and inference is not None and observed <= inference, "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
    sid = event["schedule_provider_id"]
    original = [r for r in inventory.get("identity_events", []) if str(r.get("id")) == sid]
    require(bool(original), "NCAAF_SCHEDULE_RECEIPT_MISSING")
    body = dict(source="espn_schedule", schema_version=1, observed_at=inventory["observed_at"],
        start_date=inventory["start_date"], end_date=inventory["end_date"],
        complete=inventory["complete"], status=inventory["status"], reasons=inventory["reasons"],
        events=[deepcopy(event)], identity_events=deepcopy(original))
    require(len(encode(body).encode()) <= 128*1024, "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
    return dict(scope="selected_event_only", inventory_sha256=digest(inventory), payload=body, sha256=digest(body))


def predict(source, *, inventory=None):
    at = generated_time()
    result = dict(ml_probability=float("nan"), ml_probability_source="", ml_target="",
        ml_projection=float("nan"), ml_residual_scale=float("nan"), ml_feature_quality="unavailable",
        ml_inference_status="unavailable", ml_unavailable_reason="NCAAF_EXACT_OFFER_NOT_SELECTED")
    attempted = None
    try:
        packets = [p for p in _SELECTED.get() if _offer_matches(p, source)]
        require(len(packets) == 1, "NCAAF_PACKET_AMBIGUOUS" if packets else "NCAAF_EXACT_OFFER_NOT_SELECTED")
        attempted = packets[0]
        schedule = board_schedule(source, attempted, inventory, at)
        is_compatible = attempted["payload"]["version"] in INPUT_VERSIONS
        if is_compatible:
            approval = accepted(attempted)
            dependency_review = _reader(attempted).accepted_dependencies(attempted, approval, at)
            checked, center, win, computation = _reader(attempted).infer(attempted, at)
            m, target = checked["model"]["original_record"], checked["target"]
            fit = checked["fit"]
        else:
            m, features, target = _checked(attempted, source, at)
        family, kind, line = target["family"], source["market_type"], target["signed_line"]
        fitted_target = "margin" if family == "spread" else "total"
        if not is_compatible:
            fit = m["data"]["artifact"]["models"][attempted["payload"]["selection"]["model_name"]][fitted_target]
            center = float(research.centers(fit, [features], fitted_target)[0])
            threshold = (-line if kind == "spread_home" else line) if family == "spread" else line
            mass = research.probabilities(center, fit["sigma"], threshold, total=family == "total")
            win = mass["over" if kind in {"spread_home", "total_over"} else "under"]
            require(mass["push"] == 0 and math.isclose(win, attempted["payload"]["probabilities"]["win"], abs_tol=1e-12),
                    "NCAAF_PROBABILITY_BINDING_CONFLICT")
        if is_compatible and attempted['payload']['version'] in owner_spread.INPUT_VERSIONS:
            masses=computation['payload']['outcome_probabilities']
            result.update(push_probability=masses['push'],loss_probability=masses['loss'],
                probability_semantics='unconditional_win_push_loss')
        result.update(ml_probability=win, ml_probability_source="ncaaf-research-v1:" +
            m["data"]["artifact_hash"] + ":" + _transport(attempted)["selection"]["model_name"],
            ml_target="spread_cover" if family == "spread" else "total",
            ml_projection=center if kind != "spread_away" else -center,
            ml_residual_scale=fit["sigma"], ml_feature_quality="native_ordered_seven_day_lag_features",
            ml_inference_status="success", ml_unavailable_reason="")
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, RuntimeError, AttributeError) as exc:
        reason = str(exc)
        result["ml_unavailable_reason"] = reason if reason in PUBLIC_REASONS else "NCAAF_INFERENCE_FAILED"
    line = producer._line(source)
    origin_source = _projection(attempted).presentation(attempted,source) if attempted is not None and attempted['payload']['version'] in owner_spread.INPUT_VERSIONS | {owner_terms.VERSION,owner_staged.VERSION} and result['ml_inference_status']=='success' else source
    metadata = origin_metadata(origin_source, result, line, generated_at=at)
    metadata, fields = producer.record(origin_source, result, metadata, at,
        ncaaf_schedule=producer.text(source.get("matchup_id")).startswith("espn:college-football:"))
    item = json.loads(metadata)
    p = dict(version=VERSION, status=result["ml_inference_status"], reason=result["ml_unavailable_reason"],
        inference_time=at, scientific_acceptance=False, wagering_authority=False, live_stake=0)
    if attempted is None and any(v["payload"]["version"] in INPUT_VERSIONS for v in _SELECTED.get()):
        p.update(version=compatible.RESULT_VERSION, selected_packet_hashes=[v["sha256"] for v in _SELECTED.get()])
    if result["ml_inference_status"] == "success":
        p.update(original_packet=deepcopy(attempted), consumed_reader=reader_binding(),
            board_schedule=schedule, raw_probability=fact(result["ml_probability"]), original_blend=None, ui_refresh=None)
        if attempted["payload"]["version"] in INPUT_VERSIONS:
            p.update(version=_reader(attempted).result_version(attempted), computation=computation, consumed_reader=compatible_reader_binding(attempted),
                consumed_dependency_review=dependency_review)
            if attempted['payload']['version'] in owner_spread.INPUT_VERSIONS:
                p['outcome_probabilities']=deepcopy(computation['payload']['outcome_probabilities'])
    elif attempted is not None:
        p["attempted_packet_sha256"] = attempted["sha256"]
        if attempted["payload"]["version"] in INPUT_VERSIONS:
            p.update(version=_reader(attempted).result_version(attempted), original_packet=deepcopy(attempted))
    item["ncaaf_inputs"] = dict(payload=p, sha256=digest(p))
    result.update(fields, ml_estimate_metadata=encode(item))
    return result


def diagnose(source, item=None, *, _comparison=None):
    """Static provenance checks. Never replay or recalculate historical features."""
    try:
        item = item if item is not None else json.loads(source.get("ml_estimate_metadata", ""))
        saved = item["ncaaf_inputs"]
        p = saved["payload"]
        require(set(saved) == {"payload", "sha256"} and digest(p) == saved["sha256"], "NCAAF_PACKET_INTEGRITY")
        if p.get("version") in RESULT_VERSIONS:
            return _diagnose_compatible(source, item, p, _comparison=_comparison)
        require(p["version"] == VERSION, "NCAAF_PACKET_SCHEMA")
        require(p["scientific_acceptance"] is False and p["wagering_authority"] is False and p["live_stake"] == 0,
                "NCAAF_AUTHORITY_FORBIDDEN")
        if p["status"] != "success":
            return dict(status="INCOMPLETE", reason=p["reason"] if p["reason"] in PUBLIC_REASONS else "NCAAF_INFERENCE_FAILED")
        packet = load(encode(p["original_packet"]).encode())
        require(_offer_matches(packet, source), "NCAAF_EVENT_OFFER_CONFLICT")
        schedule = p["board_schedule"]
        if producer.text(source.get("matchup_id")).startswith("espn:college-football:"):
            require(isinstance(schedule,dict) and schedule.get("scope") == "selected_event_only"
                and digest(schedule["payload"]) == schedule["sha256"], "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
            verified = board_schedule(source,packet,schedule["payload"],p["inference_time"])
            require(verified["payload"] == schedule["payload"], "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
        else:
            require(schedule is None, "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
        accepted(packet)
        _fresh(packet, p["inference_time"])
        require(p["inference_time"] == item["generated_at"] and p["consumed_reader"] == reader_binding(), "NCAAF_RUNTIME_BINDING_CONFLICT")
        require(p["raw_probability"] == item["probability"] == fact(source.get("ml_probability"))
            and math.isclose(p["raw_probability"]["value"], packet["payload"]["probabilities"]["win"], abs_tol=1e-12),
            "NCAAF_PROBABILITY_BINDING_CONFLICT")
        require(abs(packet["payload"]["selection"]["line"] % 1) == .5, "NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED")
        model = next(r for r in packet["payload"]["records"] if r["kind"] == "model")
        expected = "ncaaf-research-v1:" + model["data"]["artifact_hash"] + ":" + packet["payload"]["selection"]["model_name"]
        require(source.get("ml_probability_source") == expected and item["predictor_id"] == fact(expected), "NCAAF_ARTIFACT_TARGET_CONFLICT")
        original = dict(item)
        original.pop("ncaaf_inputs")
        from app_core.research_estimate_trace import origin_rejection
        require(origin_rejection(dict(source, ml_estimate_metadata=encode(original))) is None, "NCAAF_EVENT_OFFER_CONFLICT")
        return dict(status="COMPLETE", reason="AVAILABLE")
    except (ValueError, TypeError, KeyError, StopIteration, IndexError, AttributeError) as exc:
        return dict(status="REJECTED", reason=str(exc) if str(exc) in PUBLIC_REASONS else "NCAAF_PACKET_SCHEMA")


def compatible_reader_binding(packet=None):
    binding = dict(pipeline=reader_binding(), compatible_caller_sha256=hashlib.sha256(
        Path(compatible.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    if packet is not None and packet["payload"]["version"] == compatible.SUCCESSOR_VERSION:
        binding["chronology_reader"] = dict(version=compatible.chronology.VERSION, sha256=hashlib.sha256(
            Path(compatible.chronology.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    if packet is not None and packet['payload']['version'] in {custody.VERSION, owner.ReviewPolicy.VERSION, owner_terms.ReviewPolicy.VERSION,owner_staged.ReviewPolicy.VERSION}:
        binding['custody_reader'] = dict(version=packet['payload']['version'], components=custody.implementation(),
            native_reader=compatible_reader_binding(packet['payload']['native_packet']))
    if packet is not None and packet['payload']['version'] in {owner.ReviewPolicy.NATIVE_VERSION,owner_terms.ReviewPolicy.NATIVE_VERSION,owner_staged.ReviewPolicy.NATIVE_VERSION}:
        from app_core import ncaaf_owner_mapping
        binding['private_components'] = {module.__name__: hashlib.sha256(
            Path(module.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for module in (compatible.chronology, ncaaf_owner_mapping)}
    if packet is not None and packet["payload"]["version"] == owner.VERSION:
        binding["owner_reader"] = dict(version=owner.VERSION, sha256=hashlib.sha256(
            Path(owner.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            custody_reader=compatible_reader_binding(packet["payload"]["custody_packet"]))
    if packet is not None and packet['payload']['version'] in {owner_terms.VERSION,owner_staged.VERSION}:
        binding['owner_terms_reader'] = dict(version=owner_terms.VERSION,components={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for module in (owner_terms,owner_terms.projection)},custody_reader=compatible_reader_binding(packet['payload']['custody_packet']))
    if packet is not None and packet['payload']['version'] == owner_staged.VERSION:
        from app_core import ncaaf_staged_evidence, ncaaf_staged_acquisition, ncaaf_owner_event_crosswalk
        binding['owner_staged_reader'] = dict(version=owner_staged.VERSION, components={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for m in (owner_staged,ncaaf_staged_evidence,ncaaf_staged_acquisition,ncaaf_owner_event_crosswalk)}, custody_reader=compatible_reader_binding(packet['payload']['custody_packet']))
    if packet is not None and packet['payload']['version'] in owner_spread.INPUT_VERSIONS:
        from app_core import ncaaf_future_target,ncaaf_future_staged,ncaaf_spread_mass,ncaaf_spread_execution,ncaaf_spread_pair
        binding['owner_spread_reader']=dict(version=packet['payload']['version'],components={Path(m.__file__).name:
            hashlib.sha256(Path(m.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for m in (owner_spread,owner_spread.projection,compatible.chronology,custody,ncaaf_future_target,ncaaf_future_staged,
                owner_staged.staged.stages,ncaaf_spread_mass,ncaaf_spread_execution,ncaaf_spread_pair)})
        if packet['payload']['staged_binding']['payload'].get('version')=='ncaaf-reviewed-staged-evidence-v3':
            from app_core import ncaaf_reviewed_target,ncaaf_reviewed_staged
            binding['owner_spread_reader']['reviewed_target_components']={Path(m.__file__).name:
                hashlib.sha256(Path(m.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                for m in (ncaaf_reviewed_target,ncaaf_reviewed_staged)}
            binding['pipeline']['artifacts']['app_core/ncaaf_pipeline_evidence.py']=ncaaf_reviewed_target.source_hash('app_core/ncaaf_pipeline_evidence.py')
    historical=packet is None or packet['payload']['version'] not in owner_spread.INPUT_VERSIONS or packet['payload']['staged_binding']['payload'].get('version')!='ncaaf-reviewed-staged-evidence-v3'
    if historical:
        from app_core.ncaaf_reviewed_target import HISTORICAL_COMPONENTS,source_hash
        def preserve(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    path='app_core/'+key
                    if path in HISTORICAL_COMPONENTS and isinstance(item,str):value[key]=source_hash(path,historical=True)
                    else:preserve(item)
        preserve(binding)
    return binding


def _diagnose_compatible(source, item, p, *, _comparison=None):
    comparison=None
    require(p["scientific_acceptance"] is False and p["wagering_authority"] is False and p["live_stake"] == 0,
            "NCAAF_AUTHORITY_FORBIDDEN")
    if p["status"] != "success":
        return dict(status="INCOMPLETE", reason=p["reason"] if p["reason"] in PUBLIC_REASONS else "NCAAF_INFERENCE_FAILED")
    packet = load(encode(p["original_packet"]).encode())
    require(p["version"] == _reader(packet).result_version(packet), "NCAAF_COMPAT_COMPUTATION_RECEIPT_CONFLICT")
    require(_offer_matches(packet, source), "NCAAF_EVENT_OFFER_CONFLICT")
    schedule = p["board_schedule"]
    if producer.text(source.get("matchup_id")).startswith("espn:college-football:"):
        require(isinstance(schedule, dict) and schedule.get("scope") == "selected_event_only"
            and digest(schedule["payload"]) == schedule["sha256"], "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
        require(board_schedule(source, packet, schedule["payload"], p["inference_time"])["payload"] == schedule["payload"],
                "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
    else:
        require(schedule is None, "NCAAF_SCHEDULE_RECEIPT_CONFLICT")
    approval = accepted(packet)
    require(p["consumed_dependency_review"] == _reader(packet).accepted_dependencies(packet, approval, p["inference_time"]),
            "NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT")
    checked, computation = _reader(packet).inspect_result(packet, p["computation"], p["inference_time"])
    require(p["inference_time"] == item["generated_at"] and p["consumed_reader"] == compatible_reader_binding(packet),
            "NCAAF_RUNTIME_BINDING_CONFLICT")
    require(p["raw_probability"] == item["probability"] == fact(source.get("ml_probability")) == fact(computation["raw_probability"]),
            "NCAAF_PROBABILITY_BINDING_CONFLICT")

    if packet['payload']['version'] in owner_spread.INPUT_VERSIONS:
        masses=computation['outcome_probabilities']
        require(p.get('outcome_probabilities')==masses,'NCAAF_PROBABILITY_BINDING_CONFLICT')
        for field,key in (('push_probability','push'),('loss_probability','loss')):
            actual=producer.number(source.get(field))
            require(actual is None or actual==masses[key],'NCAAF_PROBABILITY_BINDING_CONFLICT')
        if computation['execution_receipt']['authorization']['payload']['max_computations']==2:
            from app_core.ncaaf_spread_pair import inspect_row
            comparison=inspect_row(packet,p)
    expected = "ncaaf-research-v1:" + checked["model"]["original_record"]["data"]["artifact_hash"] + ":ridge"
    require(source.get("ml_probability_source") == expected and item["predictor_id"] == fact(expected), "NCAAF_ARTIFACT_TARGET_CONFLICT")
    original = dict(item)
    original.pop("ncaaf_inputs")
    from app_core.research_estimate_trace import origin_rejection
    origin_source = _projection(packet).presentation(packet,source) if packet['payload']['version'] in owner_spread.INPUT_VERSIONS | {owner_terms.VERSION,owner_staged.VERSION} else source
    require(origin_rejection(dict(origin_source, ml_estimate_metadata=encode(original))) is None, "NCAAF_EVENT_OFFER_CONFLICT")
    # A caller-owned return channel for this one successful validation only.
    # No verdict or mutable input is retained between public invocations.
    if _comparison is not None and comparison is not None:_comparison.append(comparison)
    return dict(status="COMPLETE", reason="AVAILABLE")


def finish(frame):
    for index, row in frame.iterrows():
        try:
            item = json.loads(row.get("ml_estimate_metadata", ""))
            saved = item["ncaaf_inputs"]
            if saved["payload"]["status"] != "success":
                continue
            saved["payload"]["original_blend"] = dict(probability=fact(row.get("calibrated_probability")),
                semantics="unvalidated_market_context_blend_not_native_model_probability")
            saved["sha256"] = digest(saved["payload"])
            frame.at[index, "ml_estimate_metadata"] = encode(item)
        except (ValueError, TypeError, KeyError):
            continue
    return frame
