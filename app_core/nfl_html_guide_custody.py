"""Explicit successor guide custody; old readers cannot borrow its semantics."""
import base64, hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from app_core import nfl_observed_custody as previous
from app_core.producer_provenance import clock

VERSION='nfl-public-guide-original-html-v2'
ACTIVE=ContextVar('nfl_html_guide_v8',default=False)

@contextmanager
def selected():
    token=ACTIVE.set(True)
    try:yield
    finally:ACTIVE.reset(token)

def read(obj):
    from app_core.nfl_document_acquisition import GUIDE_URL,body_ok
    previous.require(isinstance(obj,dict) and set(obj)=={'version','body_sha256','bytes_base64','receipt'} and obj['version']==VERSION,'NFL_PRIVATE_CUSTODY_MISSING')
    try:raw=base64.b64decode(obj['bytes_base64'],validate=True)
    except (ValueError,TypeError):raise ValueError('NFL_PRIVATE_CUSTODY_CORRUPT') from None
    previous.require(0<len(raw)<=previous.MAX_BODY and hashlib.sha256(raw).hexdigest()==obj['body_sha256'],'NFL_PRIVATE_CUSTODY_CORRUPT')
    r=obj['receipt']
    previous.require(isinstance(r,dict) and set(r)==set('source_id endpoint format evidence_label request_started_at received_at observed_at observation_meaning provider_clock_meaning'.split()),'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    previous.require(r['source_id']=='clock' and r['endpoint']==GUIDE_URL and r['format']=='text' and r['evidence_label'] in {'RETAINED','SYNTHETIC'} and r['observation_meaning']=='actual_local_first_observation_of_complete_decoded_body' and r['provider_clock_meaning']=='attributed_document_no_publication_clock','NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    times=[clock(r[k]) for k in ('request_started_at','received_at','observed_at')]
    previous.require(all(times) and times[0]<=times[1]<=times[2],'NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT')
    body_ok(raw,'html',public_guide_examples=True,html_syntax=True)
    return raw,raw.decode('utf-8')

def retained(raw,*,receipt):
    previous.require(isinstance(raw,bytes),'NFL_PRIVATE_CUSTODY_CORRUPT')
    value=dict(version=VERSION,body_sha256=hashlib.sha256(raw).hexdigest(),bytes_base64=base64.b64encode(raw).decode(),receipt=receipt)
    read(value);return value
