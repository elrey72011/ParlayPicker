"""Synthetic-only actual HTTP/socket/CLI/custody diagnostics. No external calls."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
import time

import pytest

from app_core import ncaaf_pilot as pilot, ncaaf_staged_acquisition as stage
from app_core import ncaaf_transport_diagnostics as diagnostic
from scripts.benchmark_drive_history_loading import blocked_network
import ncaaf_future_pair_fixtures as f

SECRET = 'SYNTHETIC_NOT_A_REAL_CFBD_KEY_847562'


@contextmanager
def local_wire(monkeypatch, bodies, *, closing=True, mode='success', framing='length'):
    pairs=[socket.socketpair() for _ in bodies]
    stop=threading.Event(); connections=[]; sent=[]; workers=[]
    original_request=http.client.HTTPSConnection.request
    def serve(peer,raw):
        try:
            peer.settimeout(2);request=b''
            while b'\r\n\r\n' not in request:
                data=peer.recv(4096)
                if not data:return
                request+=data
                assert len(request)<=16384
            sent.append(True)  # Never retain headers, even synthetic ones.
            if mode=='headers':stop.wait(2);return
            header=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
            header+=b'X-Synthetic-Sensitive: '+SECRET.encode()+b'\r\n'
            header+=b'Connection: close\r\n' if closing else b'Connection: keep-alive\r\n'
            if framing=='chunked':
                header+=b'Transfer-Encoding: chunked\r\n'
                encoded=b''.join(hex(len(raw[i:i+4093]))[2:].encode()+b'\r\n'+raw[i:i+4093]+b'\r\n' for i in range(0,len(raw),4093))+b'0\r\n\r\n'
            else:
                declared=len(raw)+8 if mode=='truncated' else 524289 if mode=='oversized' else len(raw)
                header+=b'Content-Length: '+str(declared).encode()+b'\r\n';encoded=raw
            peer.sendall(header+b'\r\n')
            if mode=='body':stop.wait(2);return
            peer.sendall(encoded)
            if closing or mode=='truncated':peer.shutdown(socket.SHUT_WR)
            else:stop.wait(2)
        except OSError:
            pass  # Synthetic server cleanup; production errors are unchanged.
    for (_,peer),body in zip(pairs,bodies):
        worker=threading.Thread(target=serve,args=(peer,body),daemon=True);worker.start();workers.append(worker)
    def connect(conn):
        assert len(connections)<len(pairs),'Unexpected retry'
        conn.sock=pairs[len(connections)][0];connections.append(conn)
        if mode=='connect':stop.wait(2)
    def request(conn,*args,**kwargs):
        if mode=='write':stop.wait(2)
        return original_request(conn,*args,**kwargs)
    try:
        with blocked_network(),monkeypatch.context() as patch:
            patch.setattr(http.client.HTTPSConnection,'connect',connect)
            patch.setattr(http.client.HTTPSConnection,'request',request)
            yield dict(connections=connections,sent=sent,pairs=pairs)
    finally:
        stop.set()
        for pair in pairs:
            for sock in pair:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                sock.close()
        for worker in workers:
            worker.join(2);assert not worker.is_alive()


def cli_module():
    path=Path(__file__).resolve().parents[1]/'scripts/ncaaf_staged_acquisition.py'
    spec=importlib.util.spec_from_file_location('SYNTHETIC_ncaaf_diagnostic_cli',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def setup_cli(tmp_path,monkeypatch):
    p=f.plan();a,raw=f.stage_configuration(p,monkeypatch,tmp_path)
    paths=[]
    for name,body in [('plan',stage.model.encode(p)),('authorization',stage.model.encode(a)),('configuration',raw)]:
        path=tmp_path/(name+'.json');path.write_bytes(body);paths.append(path)
    c=cli_module()
    with pytest.raises(ValueError,match='RUNTIME_UNSUPPORTED'):c.verify_runtime()
    # Test-only transport block/loopback replacement intentionally fails the
    # real guard. Exercise the unchanged main below; never relax production.
    monkeypatch.setattr(c,'verify_runtime',lambda:None)
    monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(c.getpass,'getpass',lambda prompt:SECRET)
    monkeypatch.setattr(pilot,'utc',lambda:'2026-10-17T18:00:01Z')
    argv=['execute',str(paths[0]),'--authorization',str(paths[1]),'--controlled-bindings',str(paths[2]),
        '--bindings-sha256',hashlib.sha256(raw).hexdigest(),'--custody-root',str(tmp_path),'--transport-diagnostics']
    return c,p,argv


def read_result(tmp_path,p,stdout):
    summary=json.loads(stdout)
    b=json.loads((tmp_path/(p['sha256']+'.stage.json')).read_bytes())
    raw=(tmp_path/(p['sha256']+'.transport-diagnostics.json')).read_bytes()
    assert summary['transport_diagnostics_readback']=='PASS'
    d=diagnostic.read(tmp_path,b,raw,summary['transport_diagnostics_file_sha256'])
    assert SECRET.encode() not in raw+stdout.encode()
    assert b'X-Synthetic-Sensitive' not in raw
    assert d['provider_receipt']==d['credit_consumption']=='UNKNOWN'
    assert not d['acceptance'] and not d['inference']
    assert stage.verify_journal(tmp_path,b)
    return b,d,raw


@pytest.mark.parametrize('closing',[False,True])
@pytest.mark.parametrize('framing',['length','chunked'])
def test_real_cli_socket_original_custody_diagnostics_and_exact_readback(tmp_path,monkeypatch,capsys,closing,framing):
    c,p,args=setup_cli(tmp_path,monkeypatch)
    bodies=f.bodies()
    with local_wire(monkeypatch,bodies,closing=closing,framing=framing) as wire:
        assert c.main(args)==0
        b,d,raw=read_result(tmp_path,p,capsys.readouterr().out)
        assert b['payload']['status']=='CAPTURED_UNREVIEWED' and len(d['attempts'])==4
        for item,body in zip(b['payload']['objects'],bodies):
            assert item['body_sha256']==hashlib.sha256(body).hexdigest()
        assert all(a['http_status']==200 and a['request_write_completed'] and a['complete_original']
            and a['failure_phase']=='NONE' and a['reason'] is None for a in d['attempts'])
        journal=tmp_path/(p['sha256']+'.attempts.jsonl');before=journal.read_bytes()
        assert c.main(args)==1 and json.loads(capsys.readouterr().out)['status']=='REJECTED'
        assert len(wire['connections'])==len(wire['sent'])==4
        assert journal.read_bytes()==before
        assert (tmp_path/(p['sha256']+'.transport-diagnostics.json')).read_bytes()==raw


@pytest.mark.parametrize('mode,phase,status,written',[
    ('connect','connect_dns_tcp_tls','UNKNOWN',False),('write','request_write','UNKNOWN',False),
    ('headers','response_headers','UNKNOWN',True),('body','response_body_read',200,True)])
def test_phase_specific_local_delays_through_actual_cli_and_failure_reader(tmp_path,monkeypatch,capsys,mode,phase,status,written):
    c,p,args=setup_cli(tmp_path,monkeypatch)
    original=pilot._timed
    current_phase=[None]
    enter=diagnostic.Collector.enter
    def observed_enter(self,value):
        enter(self,value);current_phase[0]=value
    monkeypatch.setattr(diagnostic.Collector,'enter',observed_enter)
    native_init=pilot.HttpsTransport.__init__
    def slower_lookup(self,credentials,**kwargs):
        def supplied(provider):
            time.sleep(.12)  # A non-target phase must retain its real budget.
            return credentials(provider)
        native_init(self,supplied,**kwargs)
    monkeypatch.setattr(pilot.HttpsTransport,'__init__',slower_lookup)
    def synthetic_short_wait(operation,deadline,mono,cancelled,close):
        # Only this labelled fixture shortens the real waiter; the transport
        # still receives production10/10/60bounds, which remain unchanged.
        selected=min(deadline,mono()+.08) if current_phase[0]==phase else deadline
        return original(operation,selected,mono,cancelled,close)
    monkeypatch.setattr(pilot,'_timed',synthetic_short_wait)
    assert stage.LIMITS['connect_seconds']==stage.LIMITS['read_seconds']==10
    assert stage.LIMITS['shared_active_seconds']==60
    with local_wire(monkeypatch,[b'[]'],mode=mode) as wire:
        assert c.main(args)==1
        b,d,_=read_result(tmp_path,p,capsys.readouterr().out)
        assert b['payload']['reason']=='NCAAF_PILOT_DEADLINE' and b['payload']['attempts']==1
        assert b['payload']['objects']==[] and len(d['attempts'])==1
        item=d['attempts'][0]
        assert item['failure_phase']==phase and item['http_status']==status
        assert item['request_write_completed'] is written and not item['complete_original']
        assert item['reason']=='NCAAF_PILOT_DEADLINE'
        assert len(wire['connections'])==1
    before=len(wire['sent']);time.sleep(.02)
    assert len(wire['sent'])==before  # No late-worker retry or reopened GET.


@pytest.mark.parametrize('mode,phase,reason',[('truncated','body_validation','NCAAF_PILOT_INCOMPLETE_BODY'),
    ('oversized','body_validation','NCAAF_PILOT_BODY_LIMIT')])
def test_size_and_truncation_remain_fail_closed_with_observed_status(tmp_path,monkeypatch,capsys,mode,phase,reason):
    c,p,args=setup_cli(tmp_path,monkeypatch)
    with local_wire(monkeypatch,[b'[]'],mode=mode):
        assert c.main(args)==1
        b,d,_=read_result(tmp_path,p,capsys.readouterr().out)
        assert b['payload']['objects']==[] and b['payload']['attempts']==1
        assert b['payload']['reason']==reason
        assert d['attempts'][0]['http_status']==200 and d['attempts'][0]['failure_phase']==phase


def test_private_key_prompt_can_exceed_active_budget_without_spending_it(tmp_path,monkeypatch,capsys):
    c,p,args=setup_cli(tmp_path,monkeypatch);clock=f.Clock()
    def synthetic_private_entry(prompt):
        assert 'cfbd key' in prompt;clock.elapsed+=120;return SECRET
    monkeypatch.setattr(c.getpass,'getpass',synthetic_private_entry)
    acquire=stage.acquire;seen=[]
    def actual_acquisition(*args,**kwargs):
        seen.append(clock.elapsed)
        return acquire(*args,**kwargs,monotonic=clock.mono)
    monkeypatch.setattr(stage,'acquire',actual_acquisition)
    with local_wire(monkeypatch,f.bodies()):
        assert c.main(args)==0
        b,d,_=read_result(tmp_path,p,capsys.readouterr().out)
    assert seen==[120] and b['payload']['active_seconds']==d['active_seconds']==0


@pytest.mark.parametrize('attack',['phase','extra-secret','status','time','bool-time','wrong-bundle','wrong-journal','wrong-sources','wrong-hash','unsupported-observation','too-long','provider-claim'])
def test_rehashed_diagnostic_substitution_and_secret_fields_reject(tmp_path,monkeypatch,capsys,attack):
    c,p,args=setup_cli(tmp_path,monkeypatch)
    with local_wire(monkeypatch,f.bodies()):
        assert c.main(args)==0
        b,d,_=read_result(tmp_path,p,capsys.readouterr().out)
    value=deepcopy(d)
    if attack=='phase':value['attempts'][0]['phases'][0]['phase']=SECRET
    elif attack=='extra-secret':value['attempts'][0]['header']=SECRET
    elif attack=='status':value['attempts'][0]['http_status']=SECRET
    elif attack=='time':value['attempts'][0]['phases'][0]['seconds']=float('nan')
    elif attack=='bool-time':value['attempts'][0]['phases'][0]['seconds']=True
    elif attack=='wrong-bundle':value['bundle_sha256']='e'*64
    elif attack=='wrong-journal':value['journal_sha256']='e'*64
    elif attack=='wrong-sources':value['source_sha256']['app_core/ncaaf_pilot.py']='e'*64
    elif attack=='unsupported-observation':value['attempts'][0]['phases']=[v for v in value['attempts'][0]['phases'] if v['phase']!='response_headers']
    elif attack=='too-long':value['attempts'][0]['phases'][0]['seconds']=value['active_seconds']+1
    elif attack=='provider-claim':value['provider_receipt']='RECEIVED'
    envelope = dict(payload=value,sha256='e'*64) if attack=='time' else stage.seal(value)
    raw=json.dumps(envelope).encode();h=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):diagnostic.read(tmp_path,b,raw,'e'*64 if attack=='wrong-hash' else h)


def test_secret_in_body_still_rejects_original_without_leaking_diagnostics(tmp_path,monkeypatch,capsys):
    c,p,args=setup_cli(tmp_path,monkeypatch)
    with local_wire(monkeypatch,[json.dumps([dict(SYNTHETIC=SECRET)]).encode()]):
        assert c.main(args)==1
        b,d,raw=read_result(tmp_path,p,capsys.readouterr().out)
        assert b['payload']['reason']=='NCAAF_PILOT_CREDENTIAL_IN_BODY' and b['payload']['objects']==[]
        assert d['attempts'][0]['failure_phase']=='body_validation'
        assert SECRET.encode() not in stage.model.encode(b)+raw


def test_shared_budget_exhaustion_stays_spent_and_diagnostic_has_no_extra_request(tmp_path,monkeypatch,capsys):
    c,p,args=setup_cli(tmp_path,monkeypatch);clock=f.Clock();acquire=stage.acquire
    def actual_acquisition(*args,**kwargs):return acquire(*args,**kwargs,monotonic=clock.mono)
    monkeypatch.setattr(stage,'acquire',actual_acquisition)
    original=pilot.HttpsTransport.__call__
    def exhaust(self,request,limits,deadline,mono):
        stream=original(self,request,limits,deadline,mono);clock.elapsed=60;return stream
    monkeypatch.setattr(pilot.HttpsTransport,'__call__',exhaust)
    with local_wire(monkeypatch,[b'[]']) as wire:
        assert c.main(args)==1
        b,d,_=read_result(tmp_path,p,capsys.readouterr().out)
        assert b['payload']['reason']=='NCAAF_STAGED_DEADLINE' and b['payload']['active_seconds']==60
        assert b['payload']['attempts']==1 and len(wire['connections'])==1


def test_cancelled_stream_preserves_failure_phase_and_never_reopens(tmp_path,monkeypatch,capsys):
    c,p,args=setup_cli(tmp_path,monkeypatch);original=pilot.HttpsTransport.__call__
    def cancelled(self,*args,**kwargs):
        stream=original(self,*args,**kwargs);stream.close();return stream
    monkeypatch.setattr(pilot.HttpsTransport,'__call__',cancelled)
    with local_wire(monkeypatch,[b'[]']) as wire:
        assert c.main(args)==1
        b,d,_=read_result(tmp_path,p,capsys.readouterr().out)
        assert b['payload']['reason']=='NCAAF_PILOT_DEADLINE' and b['payload']['attempts']==1
        assert d['attempts'][0]['failure_phase']=='response_body_read'
        assert len(wire['connections'])==1


def test_standalone_network_free_plan_and_help_use_the_actual_cli(tmp_path):
    import subprocess
    script=Path(__file__).resolve().parents[1]/'scripts/ncaaf_staged_acquisition.py'
    plan=tmp_path/'SYNTHETIC-plan.json';plan.write_bytes(stage.model.encode(f.plan()))
    help_result=subprocess.run([sys.executable,'-I','-B',str(script),'--help'],text=True,capture_output=True,timeout=30)
    assert help_result.returncode==0 and '--transport-diagnostics' in help_result.stdout
    # Explicit existing test libraries are needed only for optional numerical
    # imports. The child performs plan/validation, never execute or transport.
    libs=str(Path(pytest.__file__).resolve().parent.parent)
    code='import sys,runpy;sys.path.insert(0,'+repr(libs)+');sys.argv='+repr([str(script),'plan',str(plan)])+';runpy.run_path('+repr(str(script))+',run_name="__main__")'
    result=subprocess.run([sys.executable,'-I','-B','-c',code],text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    output=json.loads(result.stdout)
    assert output['requests']==f.plan()['payload']['requests']
    assert output['status']=='AWAITING_EXACT_AUTHORIZATION'
    assert output['network_requests']==output['credential_loads']==output['storage_operations']==0
    assert output['accepted'] is False and output['inference'] is False


def test_unselected_cli_keeps_legacy_call_contract_and_creates_no_sidecar(tmp_path,monkeypatch,capsys):
    c,p,args=setup_cli(tmp_path,monkeypatch);args.remove('--transport-diagnostics')
    native_transport=pilot.HttpsTransport;native_acquire=stage.acquire
    def historical_transport(credentials):return native_transport(credentials)
    def historical_acquire(plan,authorization,*,root,transport,history_bundle,prequote,wall):
        return native_acquire(plan,authorization,root=root,transport=transport,history_bundle=history_bundle,prequote=prequote,wall=wall)
    monkeypatch.setattr(pilot,'HttpsTransport',historical_transport)
    monkeypatch.setattr(stage,'acquire',historical_acquire)
    with local_wire(monkeypatch,f.bodies()):assert c.main(args)==0
    output=json.loads(capsys.readouterr().out)
    assert set(output)=={'status','reason','attempts','bundle_sha256'}
    assert output['status']=='CAPTURED_UNREVIEWED' and output['attempts']==4
    assert not (tmp_path/(p['sha256']+'.transport-diagnostics.json')).exists()
