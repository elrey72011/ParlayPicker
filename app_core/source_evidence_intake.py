"""Private exact-offer intake using replay retention, never listing registration.

The upload and its review are untrusted data. Only the existing independently
accepted listing catalog can bind a review to these exact bytes. It remains empty
in production. No provider calls, document execution or authority reader exists.
"""
from __future__ import annotations
import base64
from contextlib import closing
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

VERSION = "exact-offer-source-evidence-v1"
REVIEW_VERSION = "exact-offer-source-admissibility-v1"
PREFIX = "parlaypicker:source-evidence:v1:"
MAX_BYTES = 4 * 1024 * 1024
ACTIVE = ContextVar("private_source_evidence", default=())
IDENTITY_FIELDS = {"provider_namespace","provider_event_id","sport","home","away","start","bookmaker","market","selection","line","price","provider_quote_id","source_time"}
PURPOSES = {"bridge", "listing", "period", "settlement", "effective_window", "clock", "rights"}


def digest(value):
    from app_core.source_contract import digest as canonical_digest
    return canonical_digest(value)


def ref(packet):
    return PREFIX + digest(packet)



def absent(value):
    return value is None or value == "" or value == {} or value == [] or (isinstance(value,str) and value.strip().upper() in {"UNKNOWN","UNVERIFIED","UNAVAILABLE","NOT_RECORDED"})


def assess(packet, offer, *, inference_time=None, quote_clock_field=None):
    from app_core.source_contract import ACCEPTED_LISTINGS, RULES
    from app_core.producer_provenance import clock
    result = dict(version=VERSION, reference="", status="UNKNOWN", diagnostics=[],
                  receipt=None, quote_clock_field=quote_clock_field, scientific_acceptance=False, wagering_authority=False)
    missing, errors = [], []
    def need(value, field):
        if absent(value):
            missing.append("SOURCE_MISSING_" + field.upper())
            return False
        return True
    def need_text(value,field):
        if not need(value,field): return False
        if not isinstance(value,str) or not value.strip():
            errors.append("SOURCE_INVALID_"+field.upper())
            return False
        return True
    def keys(value, fields, label, optional=()):
        if not isinstance(value, dict) or set(value) - set(fields):
            errors.append("SOURCE_SCHEMA_" + label.upper())
            return False
        for key in set(fields)-set(value)-set(optional):
            missing.append("SOURCE_MISSING_" + label.upper()+"_"+key.upper())
        return True
    try:
        if len(json.dumps(packet,allow_nan=False).encode())>MAX_BYTES: raise ValueError("SOURCE_PACKET_SIZE_INVALID")
        result["reference"] = ref(packet)
        result["receipt"] = deepcopy(packet)
        if not keys(packet, {"version", "evidence", "review"}, "packet"):
            raise ValueError("SOURCE_PACKET_SCHEMA_UNSUPPORTED")
        if packet.get("version") != VERSION: errors.append("SOURCE_PACKET_VERSION_UNSUPPORTED")
        evidence = packet.get("evidence") or {}
        if not keys(evidence, {"identity", "inference_time", "product", "listing", "terms", "clock", "rights", "references", "documents"}, "evidence", optional={"inference_time"}):
            raise ValueError("SOURCE_PACKET_SCHEMA_UNSUPPORTED")
        actual = evidence.get("identity") or {}
        if set(offer) != IDENTITY_FIELDS: errors.append("SOURCE_CONSUMED_IDENTITY_SCHEMA_UNSUPPORTED")
        if keys(actual, IDENTITY_FIELDS, "identity"):
            for name, value in offer.items():
                if name == "provider_quote_id" and value is None and name in actual and actual[name] is None: continue
                if not (need(actual.get(name), "identity_"+name) if name in {"line","price"} else need_text(actual.get(name), "identity_"+name)): continue
                if actual[name] != value: errors.append("SOURCE_CONFLICT_"+name.upper())
                if absent(value): missing.append("SOURCE_CONSUMED_"+name.upper()+"_UNKNOWN")
        for k,v in {"provider_namespace":"the_odds_api","sport":"americanfootball_nfl","market":"spreads","bookmaker":"novig"}.items():
            if not absent(offer.get(k)) and offer[k]!=v: errors.append("SOURCE_SCOPE_UNSUPPORTED")
        line = offer.get("line")
        if not absent(line) and (isinstance(line, bool) or not isinstance(line,(int,float)) or abs(line*2-round(line*2))>1e-9 or abs(line-round(line))<=1e-9):
            errors.append("SOURCE_HALF_POINT_REQUIRED")
        if not absent(offer.get("home")) and not absent(offer.get("away")) and (offer["home"]==offer["away"] or (not absent(offer.get("selection")) and offer["selection"] not in {offer["home"],offer["away"]})):
            errors.append("SOURCE_NAMED_ORIENTATION_CONFLICT")
        product=evidence.get("product") or {}
        if keys(product, {"operator", "product", "jurisdiction", "bookmaker"}, "product"):
            for k in product: need_text(product[k],"product_"+k)
            if not absent(product.get("bookmaker")) and not absent(offer.get("bookmaker")) and product["bookmaker"] != offer["bookmaker"]: errors.append("SOURCE_CONFLICT_PRODUCT_BOOKMAKER")
        listing=evidence.get("listing") or {}
        if keys(listing, {"id", "reference_team", "count", "comparison", "position", "price_units"}, "listing"):
            for k in listing:
                (need if k=="count" else need_text)(listing[k],"listing_"+k)
            expected_listing={"reference_team":offer.get("selection"),"comparison":"above","position":"YES","count":(-line if isinstance(line,(int,float)) else None),"price_units":"american_odds_from_bound_yes_offer"}
            for k,v in expected_listing.items():
                if not absent(v) and not absent(listing.get(k)) and listing[k]!=v: errors.append("SOURCE_SELECTED_SIDE_CONFLICT_"+k.upper())
        terms=evidence.get("terms") or {}
        term_fields={"period", "overtime", "rule_version", "effective_from", "effective_until", "scheduling", "cancellation", "void", "payoff"}
        if keys(terms,term_fields,"terms"):
            for k in terms:
                (need if k=="overtime" else need_text)(terms[k],"terms_"+k)
            if not absent(terms.get("period")) and terms["period"] != "full_game": errors.append("SOURCE_PERIOD_CONFLICT")
            if not absent(terms.get("overtime")) and terms["overtime"] is not True: errors.append("SOURCE_OVERTIME_CONFLICT")
            if not absent(terms.get("rule_version")) and terms["rule_version"] != RULES: errors.append("SOURCE_RULE_VERSION_UNSUPPORTED")
            if not absent(terms.get("payoff")) and terms["payoff"] != "FVS": errors.append("SOURCE_FVS_PAYOFF_CONFLICT")
        rights=evidence.get("rights") or {}
        if keys(rights, {"owner", "product", "storage", "research", "derived_output", "effective_from", "effective_until", "limitations"}, "rights"):
            for k in rights:
                (need if k=="product" else need_text)(rights[k],"rights_"+k)
            if not absent(rights.get("product")) and product and (not isinstance(rights["product"],dict) or any(rights["product"].get(k)!=v for k,v in product.items() if not absent(v))): errors.append("SOURCE_RIGHTS_PRODUCT_CONFLICT")
            for k in ("storage","research","derived_output"):
                if not absent(rights.get(k)) and rights[k] != "permitted": errors.append("SOURCE_RIGHTS_"+k.upper()+"_NOT_PERMITTED")
        source_clock=evidence.get("clock") or {}
        if keys(source_clock,{"field", "meaning", "timezone"},"clock"):
            for k in source_clock: need_text(source_clock[k],"clock_"+k)
            if not absent(source_clock.get("field")) and source_clock["field"] not in ("market.last_update","bookmaker.last_update"): errors.append("SOURCE_QUOTE_CLOCK_FIELD_UNSUPPORTED")
            if quote_clock_field is not None and not absent(source_clock.get("field")) and source_clock["field"] != quote_clock_field: errors.append("SOURCE_CONFLICT_QUOTE_CLOCK_FIELD")
            if not absent(source_clock.get("timezone")) and source_clock["timezone"] != "UTC": errors.append("SOURCE_QUOTE_CLOCK_TIMEZONE_CONFLICT")
        documents=evidence.get("documents") or []
        ids=set()
        if not isinstance(documents,list) or len(documents)>32: errors.append("SOURCE_DOCUMENT_SCHEMA_UNSUPPORTED");documents=[]
        if not documents: missing.append("SOURCE_MISSING_DOCUMENTS")
        for d in documents:
            if not keys(d,{"id","sha256","bytes_base64","media_type","source"},"document"): continue
            if not need_text(d.get("id"),"document_id"): continue
            if d["id"] in ids: errors.append("SOURCE_DOCUMENT_ID_CONFLICT");continue
            ids.add(d["id"])
            if not all(need_text(d.get(k),"document_"+k) for k in ("sha256","bytes_base64","media_type","source")): continue
            raw=base64.b64decode(d["bytes_base64"],validate=True)
            if not raw or hashlib.sha256(raw).hexdigest()!=d["sha256"]: errors.append("SOURCE_DOCUMENT_INTEGRITY_FAILURE")
        references=evidence.get("references") or {}
        if keys(references,PURPOSES,"references"):
            for purpose in PURPOSES:
                r=references.get(purpose) or {}
                if not keys(r,{"document_id","locator"},"reference_"+purpose): continue
                if need_text(r.get("document_id"),"reference_"+purpose) and r["document_id"] not in ids:
                    (errors if documents else missing).append("SOURCE_REFERENCE_DOCUMENT_MISSING_"+purpose.upper())
                need_text(r.get("locator"),"reference_locator_"+purpose)
        review=packet.get("review") or {}
        review_fields={"version","id","reviewer","decision","evidence_sha256","identity_sha256","product_sha256","rights_sha256","rule_version","reviewed_at","expires_at"}
        if keys(review,review_fields,"review"):
            for k in review: need_text(review[k],"review_"+k)
            for k,value in [("version",REVIEW_VERSION),("decision","ACCEPTED_FOR_PRIVATE_RESEARCH"),("evidence_sha256",digest(evidence)),("identity_sha256",digest(actual)),("product_sha256",digest(product)),("rights_sha256",digest(rights)),("rule_version",terms.get("rule_version"))]:
                if not absent(value) and not absent(review.get(k)) and review[k]!=value: errors.append("SOURCE_REVIEW_CONFLICT_"+k.upper())
        accepted=ACCEPTED_LISTINGS.get(result["reference"])
        if not accepted: missing.append("SOURCE_ADMISSIBILITY_REVIEW_NOT_ACCEPTED")
        elif not isinstance(accepted,dict) or accepted != {"source_evidence_sha256":digest(packet),"source_review_sha256":digest(review)}:
            errors.append("SOURCE_ACCEPTANCE_BINDING_CONFLICT")
        # Source documents cannot invent a future inference clock. The producer
        # supplies it on replay; any separately supplied original clock must match.
        values=[actual.get("source_time"),actual.get("start"),terms.get("effective_from"),terms.get("effective_until"),rights.get("effective_from"),rights.get("effective_until"),review.get("reviewed_at"),review.get("expires_at")]
        times=[]
        for i,v in enumerate(values):
            if not need(v,"clock_"+str(i)): times.append(None)
            else:
                t=clock(v);times.append(t)
                if t is None: errors.append("SOURCE_INVALID_CLOCK_"+str(i))
        declared=evidence.get("inference_time")
        if not absent(declared) and clock(declared) is None: errors.append("SOURCE_INVALID_DECLARED_INFERENCE_CLOCK")
        inferred=clock(inference_time) if inference_time is not None else clock(declared)
        if inference_time is not None and inferred is None: errors.append("SOURCE_INVALID_CONSUMED_INFERENCE_CLOCK")
        if not absent(declared) and inference_time is not None and clock(declared)!=inferred: errors.append("SOURCE_CONFLICT_INFERENCE_TIME")
        if all(times):
            q,start,lo,hi,rlo,rhi,reviewed,expiry=times
            valid=(lo<=q<hi and rlo<=q<rhi and q<start and reviewed<expiry)
            if inferred is not None: valid=valid and q<=inferred<start and lo<=inferred<hi and rlo<=inferred<rhi and reviewed<=inferred<expiry
            if not valid: errors.append("SOURCE_CONTRACT_STALE_OR_NOT_YET_EFFECTIVE")
    except (ValueError,TypeError,KeyError,AttributeError,OverflowError) as exc:
        errors.append(str(exc) if str(exc).startswith("SOURCE_") else "SOURCE_PACKET_SCHEMA_UNSUPPORTED")
    result["diagnostics"]=sorted(set(errors+missing))
    result["status"]="REJECTED" if errors else "UNKNOWN" if missing else "VERIFIED"
    return result


def intake(raw, *, path=None):
    """Explicit owner upload. Retain original bytes, never register/execute review."""
    from app_core import prediction_evidence as evidence, research_replay as replay
    if not isinstance(raw,bytes) or not raw or len(raw)>MAX_BYTES: raise ValueError("SOURCE_PACKET_SIZE_INVALID")
    def unique(pairs):
        value={}
        for k,v in pairs:
            if k in value: raise ValueError("SOURCE_DUPLICATE_JSON_KEY")
            value[k]=v
        return value
    packet=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x: (_ for _ in ()).throw(ValueError("SOURCE_NONFINITE_JSON")))
    if not isinstance(packet,dict): raise ValueError("SOURCE_PACKET_SCHEMA_UNSUPPORTED")
    target=Path(path or evidence.database_path()).resolve()
    root=evidence.ROOT.resolve()
    if target==root or root in target.parents or any((parent/".git").exists() for parent in target.parents):
        raise ValueError("SOURCE_PRIVATE_STORAGE_OUTSIDE_REPOSITORY_REQUIRED")
    supplied_evidence=packet.get("evidence") or {}
    if not isinstance(supplied_evidence,dict): raise ValueError("SOURCE_PACKET_SCHEMA_UNSUPPORTED")
    supplied=supplied_evidence.get("identity") or {}
    if not isinstance(supplied,dict): raise ValueError("SOURCE_IDENTITY_SCHEMA_UNSUPPORTED")
    assessment=assess(packet,{k:None if absent(supplied.get(k)) else supplied[k] for k in IDENTITY_FIELDS})
    if assessment["status"]=="REJECTED": raise ValueError(";".join(assessment["diagnostics"]))
    reference=ref(packet)
    retained=dict(packet=packet,original_bytes_base64=base64.b64encode(raw).decode(),original_sha256=hashlib.sha256(raw).hexdigest())
    payload=replay.encode(retained)
    with closing(evidence.connect(target)) as db, db:
        db.execute("INSERT OR IGNORE INTO research_source_intakes VALUES (?,?,?)",(reference,payload,replay.digest(payload)))
        existing=db.execute("SELECT payload FROM research_source_intakes WHERE reference=?",(reference,)).fetchone()
        if existing != (payload,): raise ValueError("SOURCE_INTAKE_ORIGINAL_BYTES_CONFLICT")
    return {k:assessment[k] for k in ("version","reference","status","diagnostics","scientific_acceptance","wagering_authority")}


def _retained(reference, *, path=None):
    from app_core import prediction_evidence as evidence, research_replay as replay
    target=Path(path or evidence.database_path()).resolve()
    with closing(sqlite3.connect(target.as_uri()+"?mode=ro",uri=True)) as db:
        row=db.execute("SELECT payload,payload_hash FROM research_source_intakes WHERE reference=?",(reference,)).fetchone()
    if row is None: raise ValueError("SOURCE_INTAKE_NOT_RETAINED")
    payload,sha=row
    if replay.digest(payload)!=sha: raise ValueError("SOURCE_INTAKE_INTEGRITY_FAILURE")
    retained=json.loads(payload);raw=base64.b64decode(retained["original_bytes_base64"],validate=True)
    if hashlib.sha256(raw).hexdigest()!=retained["original_sha256"] or json.loads(raw)!=retained["packet"] or ref(retained["packet"])!=reference: raise ValueError("SOURCE_INTAKE_INTEGRITY_FAILURE")
    return retained,raw


def read(reference, *, path=None):
    return _retained(reference,path=path)[0]["packet"]


def download(reference, *, path=None):
    return _retained(reference,path=path)[1]


@contextmanager
def selected(references=(), *, path=None):
    """Per-run owner-selected local packets; no process/global acceptance cache."""
    if not isinstance(references,(tuple,list)) or len(references)>16 or any(not isinstance(r,str) for r in references):
        raise ValueError("SOURCE_SELECTED_INTAKE_SCHEMA_UNSUPPORTED")
    packets=[]
    for reference in references:
        try: packets.append(read(reference,path=path))
        except (OSError,sqlite3.Error,ValueError,TypeError,KeyError) as exc:
            raise ValueError("SOURCE_SELECTED_INTAKE_UNAVAILABLE") from exc
    token=ACTIVE.set(tuple(packets))
    try: yield
    finally: ACTIVE.reset(token)


def for_offer(offer, *, reference=None, quote_clock_field=None):
    packets=list(ACTIVE.get())
    if reference is not None:
        if not isinstance(reference,str) or not reference.startswith(PREFIX):
            return dict(version=VERSION,reference="",status="REJECTED",diagnostics=["SOURCE_INTAKE_REFERENCE_INVALID"],receipt=None,quote_clock_field=quote_clock_field,scientific_acceptance=False,wagering_authority=False)
        packets=[p for p in packets if ref(p)==reference]
        if not packets:
            return dict(version=VERSION,reference=reference,status="UNKNOWN",diagnostics=["SOURCE_INTAKE_NOT_OWNER_SELECTED"],receipt=None,quote_clock_field=quote_clock_field,scientific_acceptance=False,wagering_authority=False)
    else:
        packets=[p for p in packets if (p.get("evidence") or {}).get("identity")==offer]
        if not packets and ACTIVE.get() and offer.get("sport")=="americanfootball_nfl" and offer.get("bookmaker")=="novig" and offer.get("market")=="spreads":
            # A unique original event/selection anchor can diagnose a changed
            # line/price/clock. It cannot accept or transfer that old offer.
            anchor=("provider_namespace","provider_event_id","bookmaker","market","selection")
            packets=[p for p in ACTIVE.get() if all((p.get("evidence") or {}).get("identity",{}).get(k)==offer.get(k) for k in anchor)]
            if not packets:
                return dict(version=VERSION,reference="",status="UNKNOWN",diagnostics=["SOURCE_INTAKE_NO_EXACT_OFFER"],receipt=None,quote_clock_field=quote_clock_field,scientific_acceptance=False,wagering_authority=False)
    if len(packets)>1:
        return dict(version=VERSION,reference="",status="REJECTED",diagnostics=["SOURCE_INTAKE_AMBIGUOUS"],receipt=None,quote_clock_field=quote_clock_field,scientific_acceptance=False,wagering_authority=False)
    return assess(packets[0],offer,quote_clock_field=quote_clock_field) if packets else None


def replay(contract,inference_time):
    if absent(inference_time): return ["SOURCE_CONSUMED_INFERENCE_CLOCK_UNKNOWN"]
    packet=contract.get("receipt")
    if packet is None:
        return contract.get("diagnostics") or ["SOURCE_INTAKE_NOT_RETAINED"]
    expected=assess(packet,contract.get("identity") or {},inference_time=inference_time,quote_clock_field=contract.get("quote_clock_field"))
    if {k:v for k,v in contract.items() if k!="identity"}!=expected:
        return sorted(set(["SOURCE_CAPTURE_BINDING_CONFLICT"]+expected["diagnostics"]))
    return expected["diagnostics"]
