"""Explicit retrieval-bound successor. Delegates every v5 admission/math gate.

Acquisition, extraction, interpretation, v5 substantive review and this linkage
review remain distinct receipts. No original reader or historical packet changes.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from app_core import nfl_pdf_research as previous, nfl_document_acquisition as acquisition

VERSION='nfl-owner-public-document-private-inputs-v7'
RESULT='nfl-owner-public-document-private-result-v7'
REASONS=previous.REASONS | {'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT','NFL_PRIVATE_DOCUMENT_REVIEW_MISSING','NFL_PRIVATE_DOCUMENT_CLOCK_CONFLICT'}
SELECTED=ContextVar('nfl_document_packets',default=())
IN_NATIVE=previous.IN_NATIVE
require=previous.require
digest=previous.digest
encode=previous.encode


def subject(p):return digest({k:v for k,v in p.items() if k!='owner_review'})


def reader_binding():
    root=Path(__file__).resolve().parents[1]
    return dict(previous=previous.reader_binding(),sources={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for p in ('app_core/nfl_document_research.py','app_core/nfl_document_acquisition.py')})


def load(raw,*,owner_upload=False):
    require(isinstance(raw,bytes) and 0<len(raw)<=previous.MAX_PACKET,'NFL_PRIVATE_PACKET_SCHEMA')
    packet=json.loads(raw,object_pairs_hook=acquisition.custody.unique);p=acquisition.envelope(packet)
    require(set(p)==set('version evidence_label event quote packet document_captures owner_review'.split()) and p['version']==VERSION,'NFL_PRIVATE_PACKET_SCHEMA')
    inner=previous.load(encode(p['packet']).encode(),owner_upload=owner_upload)['payload']
    require(p['event']==inner['event'] and p['quote']==inner['quote'] and p['evidence_label']==inner['evidence_label'],'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')
    require(isinstance(p['document_captures'],dict) and 0<len(p['document_captures'])<=4,'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')
    return packet


def checked(packet,at,*,source=None):
    p=load(encode(packet).encode())['payload'];inner=p['packet']['payload'];latest=previous._clock(inner['owner_review']['reviewed_at'],'NFL_PRIVATE_DOCUMENT_CLOCK_CONFLICT')
    try:
        for name,binding in p['document_captures'].items():
            require(name in {'schedule','listing','clock_document','rules'} and set(binding)=={'bundle','document_id'},'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')
            bundle=binding['bundle'];require(bundle['payload']['status']=='CAPTURED_UNREVIEWED','NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')
            obj=inner['objects'][name];prepared=obj.get('extraction',{}).get('prepared_at')
            require(acquisition.project(bundle,binding['document_id'],prepared_at=prepared)==obj,'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')
            receipt=obj['receipt'];latest=max(latest,previous._clock(receipt['observed_at'],'NFL_PRIVATE_DOCUMENT_CLOCK_CONFLICT'))
    except (ValueError,KeyError,TypeError) as exc:
        raise ValueError('NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT') from None
    review=p['owner_review']
    require(set(review)=={'version','reviewer','reviewed_at','subject_sha256','conclusion','basis'} and review['version']=='nfl-owner-document-linkage-review-v1'
        and review['reviewer']=='Robert Velarde' and review['subject_sha256']==subject(p) and review['conclusion']=='VERIFIED_ORIGINAL_RETRIEVAL_LINKAGE'
        and isinstance(review['basis'],str) and len(review['basis'])>=32,'NFL_PRIVATE_DOCUMENT_REVIEW_MISSING')
    require(latest<=previous._clock(review['reviewed_at'],'NFL_PRIVATE_DOCUMENT_CLOCK_CONFLICT')<previous._clock(at,'NFL_PRIVATE_DOCUMENT_CLOCK_CONFLICT'),'NFL_PRIVATE_DOCUMENT_CLOCK_CONFLICT')
    return previous.checked(p['packet'],at,source=source)


@contextmanager
def selected(packets=()):
    packets=tuple(load(encode(p).encode()) for p in packets)
    require(len(packets)<=previous.MAX_PACKETS and sum(len(encode(p).encode()) for p in packets)<=previous.MAX_SELECTED
        and len({p['sha256'] for p in packets})==len(packets),'NFL_PRIVATE_PACKET_AMBIGUOUS')
    token=SELECTED.set(deepcopy(packets))
    try:yield
    finally:SELECTED.reset(token)


def predict(source):
    at=__import__('app_core.research_estimate_trace',fromlist=['generated_time']).generated_time()
    observed=previous.producer._matches(source)
    matches=[p for p in SELECTED.get() if p['payload']['quote']['market_type']==source.get('market_type')
        and any(m.get('provider_event_id')==p['payload']['event']['provider_event_id']
            and m.get('provider_namespace')==p['payload']['event']['provider_namespace'] for m in observed)]
    packet=None
    try:
        require(len(matches)==1,'NFL_PRIVATE_PACKET_AMBIGUOUS' if matches else 'NFL_PRIVATE_EXACT_OFFER_NOT_SELECTED')
        packet=matches[0];checked(packet,at,source=source)
    except (ValueError,KeyError,TypeError) as exc:
        # Delegate the existing unavailable-result constructor without allowing
        # numerical execution. Preserve the exact versioned reason separately.
        with previous.selected([]):result=previous.predict(source)
        reason=str(exc) if str(exc) in REASONS else 'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT'
        result['ml_unavailable_reason']=reason
    else:
        with previous.selected([packet['payload']['packet']]):result=previous.predict(source)
    item=json.loads(result['ml_estimate_metadata']);p=item['nfl_private_inputs']['payload']
    p.update(version=RESULT,document_packet=deepcopy(packet),consumed_document_reader=reader_binding(),reason=result['ml_unavailable_reason'])
    item['nfl_private_inputs']['sha256']=digest(p);result['ml_estimate_metadata']=encode(item)
    return result


def underlying(source):
    value=dict(source);item=json.loads(value['ml_estimate_metadata']);p=item['nfl_private_inputs']['payload']
    require(p['version']==RESULT and digest(p)==item['nfl_private_inputs']['sha256'],'NFL_PRIVATE_PACKET_INTEGRITY')
    if p['status']=='success':
        checked(p['document_packet'],p['inference_time'],source=source)
        require(p['document_packet']['payload']['packet']==p['original_packet'] and p['consumed_document_reader']==reader_binding(),'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')
    p.pop('document_packet');p.pop('consumed_document_reader');p['version']=previous.RESULT
    item['nfl_private_inputs']['sha256']=digest(p);value['ml_estimate_metadata']=encode(item);return value


def diagnose(source):
    try:
        item=json.loads(source['ml_estimate_metadata']);p=item['nfl_private_inputs']['payload']
        require(p['version']==RESULT and digest(p)==item['nfl_private_inputs']['sha256'],'NFL_PRIVATE_PACKET_INTEGRITY')
        require(p['wager_action']=='PASS' and p['live_stake']==0 and all(p[k] is False for k in ('source_acceptance','calibration','scientific_qualification','wagering_authority')),'NFL_PRIVATE_AUTHORITY_FORBIDDEN')
        if p['status']!='success':return dict(status='INCOMPLETE',reason=p['reason'] if p['reason'] in REASONS else 'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')
        return previous.diagnose(underlying(source))
    except (ValueError,KeyError,TypeError) as exc:return dict(status='REJECTED',reason=str(exc) if str(exc) in REASONS else 'NFL_PRIVATE_DOCUMENT_BINDING_CONFLICT')


def private_display(source):
    assessed=diagnose(source)
    if assessed['status']=='COMPLETE':return dict(previous.private_display(underlying(source)),document_retrieval='ORIGINAL_WIRE_AND_DECODED_BYTES_RETAINED')
    return dict(status=assessed['status'],reason=assessed['reason'],raw_probability=None,ev=None,edge=None,stake=0,wager_action='PASS',review_status='UNVERIFIED',purpose='PRIVATE_RESEARCH',calibrated=False)

# Only this explicitly selected successor may consume the original public-guide
# object. V6 and all older readers retain their strict default rejection.
from functools import wraps as _wraps
from app_core import nfl_public_guide_custody as _guide
def _public_guide_scope(method):
    @_wraps(method)
    def scoped(*args,**kwargs):
        with _guide.selected():return method(*args,**kwargs)
    return scoped
checked=_public_guide_scope(checked)
predict=_public_guide_scope(predict)
underlying=_public_guide_scope(underlying)
diagnose=_public_guide_scope(diagnose)
private_display=_public_guide_scope(private_display)
load=_public_guide_scope(load)
_original_reader_binding=reader_binding
def reader_binding():
    value=_original_reader_binding()
    value['public_guide_reader']=dict(version=_guide.VERSION,sha256=__import__('hashlib').sha256(__import__('pathlib').Path(_guide.__file__).read_bytes()).hexdigest())
    return value
