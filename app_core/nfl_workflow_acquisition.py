"""Hosted custody adapter around the unchanged released bounded transports.

The Windows CLI's EFS/configuration contract is not pretended to pass on Linux.
Instead, authenticated durable store claims and actual UI authority are checked
before work. No lease recovery, implicit refresh, fallback or transport retry.
"""
from copy import deepcopy
from datetime import timedelta
import hashlib
import http.client
from pathlib import Path
import re
import socket
import ssl
import time

from app_core import nfl_scoring_acquisition as scores, nfl_odds_acquisition as odds
from app_core import nfl_scoring_profile as profile, nfl_scoring_custody, nfl_observed_custody
from app_core import nfl_acquisition_diagnostics as diagnostics
from app_core.nfl_workflow_inputs import instant
from app_core.nfl_workflow_store import require, digest, utc

_TRANSPORT = (socket.create_connection,socket.getaddrinfo,socket.socket.connect,
              http.client.HTTPSConnection.connect,ssl.SSLContext.wrap_socket)
SCORING_LIMITS = dict(max_attempts=2,max_objects=1,max_wire_bytes=profile.MAX_WIRE,
    max_body_bytes=profile.MAX_BODY,max_aggregate_bytes=profile.MAX_TOTAL,
    max_packet_bytes=profile.MAX_PACKET,max_header_bytes=profile.MAX_HEADER_BYTES,
    connect_seconds=10,read_seconds=10,total_seconds=60)


def method_binding():
    from app_core.nfl_inference_evidence import runtime
    root=Path(__file__).resolve().parents[1]
    paths=('app_core/nfl_scoring_acquisition.py','app_core/nfl_odds_acquisition.py',
        'app_core/nfl_scoring_custody.py','app_core/nfl_observed_custody.py',
        'app_core/nfl_private_workflow.py','app_core/nfl_workflow_acquisition.py',
        'app_core/nfl_workflow_inputs.py','app_core/nfl_workflow_store.py')
    return dict(version='private-nfl-streamlit-acquisition-method-v1',runtime=runtime(),
        sources={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths})


def runtime_integrity():
    require(socket.create_connection.__module__=='socket' and socket.getaddrinfo.__module__=='socket'
        and getattr(socket.socket.connect,'__code__',None) is None
        and _TRANSPORT==(socket.create_connection,socket.getaddrinfo,socket.socket.connect,
                        http.client.HTTPSConnection.connect,ssl.SSLContext.wrap_socket), 'NFL_WORKFLOW_TRANSPORT_UNSUPPORTED')
    context=ssl.create_default_context()
    require(context.verify_mode==ssl.CERT_REQUIRED and context.check_hostname,'NFL_WORKFLOW_TLS_REQUIRED')


def request_for(prepared):
    start=instant(prepared['selected']['event']['start'])
    def z(value):return value.strftime('%Y-%m-%dT%H:%M:%SZ')
    request=dict(odds.SOURCE,params=dict(odds.PARAMS,
        commenceTimeFrom=z(start-timedelta(seconds=1)),commenceTimeTo=z(start+timedelta(seconds=1)),
        includeLinks='true',includeSids='true'))
    odds.check_request(request,version=odds.LISTING_VERSION)
    return request


def safe_code(error):
    value=str(error) if isinstance(error,ValueError) else ''
    return value if re.fullmatch(r'(NFL_|PRIVATE_)[A-Z_]+',value) else 'NFL_WORKFLOW_CAPTURE_FAILED'


def check_scope(operation, at):
    p=operation['payload']
    require(p['method']==method_binding(),'NFL_WORKFLOW_METHOD_CHANGED')
    require(instant(p['window']['not_before'])<=instant(at)<instant(p['window']['not_after']), 'NFL_WORKFLOW_OPERATION_EXPIRED')
    for event in p.get('events',[]):
        require(instant(at)<instant(event['start']),'NFL_WORKFLOW_KICKOFF_REACHED')


def check_launch(operation, at):
    check_scope(operation,at)
    require((instant(operation['payload']['window']['not_after'])-instant(at)).total_seconds()>60,
        'NFL_WORKFLOW_LATEST_LAUNCH_PASSED')


def capture_scores(store, operation, approval, permission, *, transport=None, wall=utc, monotonic=time.monotonic):
    require(store.synthetic or transport is None,'NFL_WORKFLOW_INJECTED_TRANSPORT_FORBIDDEN')
    if not store.synthetic:runtime_integrity()
    transport=transport or scores.HttpsTransport()
    check_launch(operation,wall())
    require(approval['scope_sha256']==digest(operation['payload']) and approval['maximum_http_requests']==2
        and approval['maximum_odds_credits']==0 and approval['maximum_extra_spending']==0,'NFL_WORKFLOW_REQUEST_AUTHORITY_CONFLICT')
    started=monotonic();deadline=started+SCORING_LIMITS['total_seconds']
    store.claim('operation:'+operation['id'],dict(approval=approval,operation=operation['id']))
    chain=[];phase='AUTHORIZATION';status=diagnostics.UNKNOWN
    try:
        request=safe=deepcopy(scores.PRIMARY)
        for hop in (1,2):
            check_scope(operation,wall())
            require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
            requested=wall()
            store.claim('http:'+operation['id']+':'+str(hop),dict(at=requested,request=safe,maximum_requests=2))
            stream=None;phase='TRANSPORT'
            try:
                stream=transport(deepcopy(request),deepcopy(SCORING_LIMITS),deadline,monotonic)
                status=diagnostics.status(stream.status);phase='RESPONSE_HEADERS';headers=scores.headers(stream)
                if stream.status==302 and hop==1:
                    location=headers.get('location');next_request,next_safe=scores.redirect_request(location)
                    body=None;transfer=None
                else:
                    location=None;phase='BODY_VALIDATION';body,transfer=scores.read(stream,SCORING_LIMITS,deadline,monotonic)
                received=wall()
            finally:
                if stream is not None:stream.close()
            check_scope(operation,received)
            require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
            step=dict(attempt=hop,request=safe,started_at=requested,received_at=received,status=status,
                      location_sha256=hashlib.sha256(location.encode()).hexdigest() if location else None)
            chain.append(step)
            store.put('capture-journal',step,identity='journal:'+operation['id']+':'+str(hop))
            if location:
                request,safe=next_request,next_safe
                continue
            transfer['requests']=chain
            phase='CUSTODY'
            original=nfl_scoring_custody.retained_score(body,receipt=dict(source_id=profile.SOURCE_ID,
                endpoint=profile.ENDPOINT,format='csv',evidence_label='SYNTHETIC' if store.synthetic else 'RETAINED',
                request_started_at=chain[0]['started_at'],received_at=received,observed_at=received,
                observation_meaning='actual_local_first_observation_of_complete_decoded_body',provider_clock_meaning='publisher_availability_unknown'),transfer=transfer)
            result=store.put('source',dict(object=original,permission=permission,operation_id=operation['id']),identity='source:'+operation['id'])
            store.put('capture-outcome',dict(status='COMPLETE',source_id=result['id'],requests_attempted=hop,
                maximum_requests=2,actual_odds_credits=0,completed_at=wall()),identity='outcome:'+operation['id'])
            return result
        raise ValueError('NFL_ACQUIRE_BUDGET_EXHAUSTED')
    except Exception as error:
        facts=error.diagnostic if isinstance(error,diagnostics.Failure) else diagnostics.facts(error,phase=phase,http_status=status,started=started,ended=monotonic())
        store.put('capture-outcome',dict(status='FAILED_NO_RETRY',reason=safe_code(error),
            diagnostics=diagnostics.checked(facts),requests_reserved=len(chain) if len(chain)==2 else 'SEE_DURABLE_HTTP_CLAIMS',
            maximum_requests=2,actual_odds_credits=0,at=wall()),identity='outcome:'+operation['id'])
        raise


def capture_market(store, operation, approval, prepared, *, credential=None, transport=None, wall=utc, monotonic=time.monotonic):
    require(store.synthetic or transport is None,'NFL_WORKFLOW_INJECTED_TRANSPORT_FORBIDDEN')
    if not store.synthetic:runtime_integrity()
    request=request_for(prepared)
    require(operation['payload']['request']==request,'NFL_WORKFLOW_REQUEST_CHANGED')
    require(approval['scope_sha256']==digest(operation['payload']) and approval['maximum_http_requests']==1
        and approval['maximum_odds_credits']==1 and approval['maximum_extra_spending']==0,'NFL_WORKFLOW_REQUEST_AUTHORITY_CONFLICT')
    check_launch(operation,wall())
    started=monotonic();deadline=started+odds.LIMITS['total_seconds']
    store.claim('operation:'+operation['id'],dict(approval=approval,operation=operation['id']))
    transport=transport or odds.HttpsTransport(credential,version=odds.LISTING_VERSION)
    phase='AUTHORIZATION';status=diagnostics.UNKNOWN
    try:
        requested=wall()
        store.claim('http:'+operation['id']+':1',dict(at=requested,request=request,maximum_requests=1,maximum_included_credit=1))
        stream=None;phase='TRANSPORT'
        try:
            stream=transport(deepcopy(request),deepcopy(odds.LIMITS),deadline,monotonic)
            status=diagnostics.status(stream.status);phase='BODY_VALIDATION'
            body,transfer=odds.read_body(stream,deadline,monotonic)
            transport.validate_body(body)
            received=wall()
            usage={name:int(value) if isinstance(value,str) and value.isdigit() else diagnostics.UNKNOWN
                for name in ('x-requests-last','x-requests-remaining','x-requests-used')
                for value in [stream.headers.get(name)]}
        finally:
            if stream is not None:stream.close()
        check_scope(operation,received);require(monotonic()<deadline,'NFL_ACQUIRE_DEADLINE')
        phase='CUSTODY'
        original=nfl_observed_custody.retained(body,receipt=dict(source_id='the-odds-api:v4:americanfootball_nfl',
            endpoint='https://'+odds.SOURCE['host']+odds.SOURCE['endpoint'],format='json',
            evidence_label='SYNTHETIC' if store.synthetic else 'RETAINED',request_started_at=requested,
            received_at=received,observed_at=received,observation_meaning='actual_local_first_observation_of_complete_decoded_body',provider_clock_meaning='provider_market_last_update'))
        # Preserve originals even if event mapping/half-point/freshness later fails.
        result=store.put('market',dict(object=original,transfer=transfer,usage=usage,
            operation_id=operation['id'],event_id=prepared['selected']['event']['canonical_event_id']),identity='market:'+operation['id'])
        store.put('capture-journal',dict(request=request,requested_at=requested,received_at=received,http_status=status,
            body_sha256=original['body_sha256'],body_bytes=len(body),usage=usage),identity='journal:'+operation['id']+':1')
        store.put('capture-outcome',dict(status='COMPLETE',market_id=result['id'],requests_attempted=1,
            included_credits_maximum=1,actual_credits=usage['x-requests-last'],completed_at=wall()),identity='outcome:'+operation['id'])
        return result
    except Exception as error:
        facts=error.diagnostic if isinstance(error,diagnostics.Failure) else diagnostics.facts(error,phase=phase,http_status=status,started=started,ended=monotonic())
        store.put('capture-outcome',dict(status='FAILED_NO_RETRY',reason=safe_code(error),
            diagnostics=diagnostics.checked(facts),requests_reserved=1,included_credits_reserved=1,at=wall()),identity='outcome:'+operation['id'])
        raise
