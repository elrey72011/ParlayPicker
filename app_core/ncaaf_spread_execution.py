"""Separately authorized one-shot private computations, never source acceptance."""
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
from pathlib import Path
from app_core import ncaaf_model_compatibility as model, ncaaf_history as history
from app_core import ncaaf_staged_acquisition as stage
from app_core.ncaaf_pilot_setup import inspect_security

VERSION='ncaaf-owner-spread-inference-authorization-v1'
AUTHORIZED_INFERENCES={}  # Exact operator configuration; uploads never populate it.
ACTIVE=ContextVar('ncaaf_spread_authorized_execution',default=None)
require=model.require


def validate(authorization,*,subject,inputs,at,reviewed_at):
    a=stage._envelope(authorization)
    require(set(a)==set('version reference subject_sha256 input_hashes authorized_at expires_at owner max_computations'.split())
        and a['version']==VERSION and isinstance(a['reference'],str) and a['reference']
        and a['subject_sha256']==subject and a['input_hashes']==inputs and a['owner']=='Robert Velarde'
        and type(a['max_computations']) is int and a['max_computations']==len(inputs) and len(inputs) in (1,2)
        and len(set(inputs))==len(inputs),'NCAAF_SPREAD_INFERENCE_AUTHORIZATION')
    now,authorized,expires,reviewed=[history.timestamp(t) for t in (at,a['authorized_at'],a['expires_at'],reviewed_at)]
    require(all((now,authorized,expires,reviewed)) and reviewed<=authorized<=now<expires,'NCAAF_SPREAD_INFERENCE_AUTHORIZATION_CLOCK')
    return a


@contextmanager
def execution(packets,authorization,*,root,subject,reviewed_at,at):
    require(ACTIVE.get() is None,'NCAAF_SPREAD_EXECUTION_ALREADY_ACTIVE')
    inputs=[p['sha256'] for p in packets]
    a=validate(authorization,subject=subject,inputs=inputs,at=at,reviewed_at=reviewed_at)
    require(AUTHORIZED_INFERENCES.get(a['reference'])==authorization['sha256'],'NCAAF_SPREAD_INFERENCE_NOT_AUTHORIZED')
    root=Path(root)
    require(root.is_dir() and inspect_security(root)['status']=='VERIFIED','NCAAF_SPREAD_PRIVATE_CUSTODY')
    for packet in packets:
        h=packet['payload']['staged_binding']['payload']['history']['payload']
        cfg=json.loads(base64.b64decode(h['control_receipt']['file_b64'],validate=True))['payload']
        require(Path(cfg['custody']['root']).resolve()==root.resolve(),'NCAAF_SPREAD_PRIVATE_CUSTODY')
    try:f=(root/(authorization['sha256']+'.inference-attempts.jsonl')).open('xb')
    except FileExistsError:raise ValueError('NCAAF_SPREAD_INFERENCE_ALREADY_SPENT') from None
    state=dict(authorization=deepcopy(authorization),subject=subject,inputs=inputs,reviewed_at=reviewed_at,
        used=set(),completed=set(),events=[],stream=f,root=root,closed=False)
    def append(event):
        stage.transport._append(f,event);state['events'].append(deepcopy(event))
    state['append']=append
    token=ACTIVE.set(state)
    try:
        with f:
            append(dict(kind='authorization',authorization=deepcopy(authorization),subject_sha256=subject,input_hashes=inputs,opened_at=at))
            try:
                yield
            finally:
                append(dict(kind='finish',status='COMPLETE' if state['completed']==set(inputs) else 'INCOMPLETE',
                    attempted=len(state['used']),completed=len(state['completed']),accepted=False,wager_action='PASS',live_stake=0))
                state['closed']=True
    finally:ACTIVE.reset(token)


def begin(packet,at):
    state=ACTIVE.get();require(state is not None and not state['closed'],'NCAAF_SPREAD_INFERENCE_NOT_AUTHORIZED')
    validate(state['authorization'],subject=state['subject'],inputs=state['inputs'],at=at,reviewed_at=state['reviewed_at'])
    sha=packet['sha256']
    require(sha in state['inputs'] and sha not in state['used'],'NCAAF_SPREAD_INFERENCE_ALREADY_SPENT')
    state['used'].add(sha);state['append'](dict(kind='attempt',input_sha256=sha,inference_time=at))


def complete(packet,computation,at,finished):
    state=ACTIVE.get();require(state is not None and packet['sha256'] in state['used']
        and packet['sha256'] not in state['completed'],'NCAAF_SPREAD_INFERENCE_ALREADY_SPENT')
    validate(state['authorization'],subject=state['subject'],inputs=state['inputs'],at=finished,reviewed_at=state['reviewed_at'])
    require(history.timestamp(finished)>=history.timestamp(at),'NCAAF_SPREAD_INFERENCE_CLOCK')
    raw_computation=model.encode(computation)
    require(len(raw_computation)<=stage.MAX_BUNDLE,'NCAAF_STAGED_BUNDLE_LIMIT')
    path=state['root']/(state['authorization']['sha256']+'.'+packet['sha256']+'.computation.json')
    stage._exclusive(path,raw_computation)
    require(path.read_bytes()==raw_computation,'NCAAF_SPREAD_INFERENCE_RECEIPT')
    state['append'](dict(kind='complete',input_sha256=packet['sha256'],computation_sha256=computation['sha256'],
        inference_time=at,completed_at=finished))
    state['completed'].add(packet['sha256'])
    raw=b''.join(model.encode(e)+b'\n' for e in state['events'])
    return dict(authorization=deepcopy(state['authorization']),journal_prefix_b64=base64.b64encode(raw).decode(),
        journal_prefix_sha256=__import__('hashlib').sha256(raw).hexdigest(),completed_at=finished)


def unavailable_computation(packet,computation,at,finished):
    """Retain a completed calculation that failed the final time/security gates.

    This is not a successful input/result receipt and cannot be replayed as one.
    Only our fixed reason is recorded; no exception or diagnostic text is kept.
    """
    state=ACTIVE.get()
    require(state is not None and packet['sha256'] in state['used'],'NCAAF_SPREAD_INFERENCE_NOT_AUTHORIZED')
    value=stage.seal(dict(version='ncaaf-spread-unavailable-computation-v1',status='UNAVAILABLE',
        input_sha256=packet['sha256'],computation=computation,inference_time=at,observed_finished_at=finished,
        reason='FINAL_INPUT_OR_AUTHORIZATION_GATE',accepted=False,wager_action='PASS',live_stake=0))
    stage._exclusive(state['root']/(state['authorization']['sha256']+'.'+packet['sha256']+'.unavailable.json'),model.encode(value))
    state['append'](dict(kind='unavailable',input_sha256=packet['sha256'],receipt_sha256=value['sha256']))


def inspect(packet,computation,proof,at):
    require(isinstance(proof,dict) and set(proof)=={'authorization','journal_prefix_b64','journal_prefix_sha256','completed_at'},
        'NCAAF_SPREAD_INFERENCE_RECEIPT')
    raw=base64.b64decode(proof['journal_prefix_b64'],validate=True)
    require(len(raw)<=128*1024 and __import__('hashlib').sha256(raw).hexdigest()==proof['journal_prefix_sha256'],'NCAAF_SPREAD_INFERENCE_RECEIPT')
    events=[json.loads(line) for line in raw.splitlines()];first=events[0];a=stage._envelope(proof['authorization'])
    require(set(first)=={'kind','authorization','subject_sha256','input_hashes','opened_at'} and first.get('kind')=='authorization' and first['authorization']==proof['authorization']
        and first['subject_sha256']==a['subject_sha256'] and first['input_hashes']==a['input_hashes']
        and events[-1]==dict(kind='complete',input_sha256=packet['sha256'],computation_sha256=computation['sha256'],
            inference_time=at,completed_at=proof['completed_at']),'NCAAF_SPREAD_INFERENCE_RECEIPT')
    validate(proof['authorization'],subject=a['subject_sha256'],inputs=a['input_hashes'],at=at,
        reviewed_at=packet['payload']['owner_review']['reviewed_at'])
    validate(proof['authorization'],subject=a['subject_sha256'],inputs=a['input_hashes'],at=proof['completed_at'],
        reviewed_at=packet['payload']['owner_review']['reviewed_at'])
    require(packet['sha256'] in a['input_hashes'] and len(events) in (3,5),'NCAAF_SPREAD_INFERENCE_RECEIPT')
    validate(proof['authorization'],subject=a['subject_sha256'],inputs=a['input_hashes'],at=first['opened_at'],
        reviewed_at=packet['payload']['owner_review']['reviewed_at'])
    used=[];previous=history.timestamp(first['opened_at'])
    for i in range(1,len(events),2):
        attempt,done=events[i:i+2]
        require(set(attempt)=={'kind','input_sha256','inference_time'} and attempt['kind']=='attempt'
            and set(done)=={'kind','input_sha256','computation_sha256','inference_time','completed_at'}
            and isinstance(done['computation_sha256'],str) and len(done['computation_sha256'])==64
            and done['kind']=='complete' and attempt['input_sha256']==done['input_sha256']
            and attempt['inference_time']==done['inference_time'] and attempt['input_sha256'] in a['input_hashes']
            and previous<=history.timestamp(done['inference_time'])<=history.timestamp(done['completed_at']), 'NCAAF_SPREAD_INFERENCE_RECEIPT')
        previous=history.timestamp(done['completed_at'])
        used.append(attempt['input_sha256'])
    require(len(used)==len(set(used))<=a['max_computations'],'NCAAF_SPREAD_INFERENCE_RECEIPT')
    return deepcopy(proof)
