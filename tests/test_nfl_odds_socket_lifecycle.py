"""SYNTHETIC real socketpair/HTTPResponse; external application transports blocked."""
from contextlib import contextmanager
from functools import partial
import hashlib
import http.client
import json
import socket
import threading
import time
import pytest
from app_core import nfl_odds_acquisition as lane
from scripts import nfl_odds_acquisition as cli
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_odds_acquisition import NOW, SECRET, security
from test_nfl_odds_listing_acquisition import fixture, response


@contextmanager
def local_http(monkeypatch, raw, *, closing=True, framing='length', declared=None, truncated=False, stall=False, journal=None):
    # Windows creates socketpair over loopback: do this before stricter blocking.
    client, peer = socket.socketpair(); stop = threading.Event(); sent=[]; states=[]
    real_response = http.client.HTTPSConnection.getresponse
    def serve():
        try:
            peer.settimeout(3); request=b''
            while b'\r\n\r\n' not in request:
                part=peer.recv(4096)
                if not part:return
                request+=part; assert len(request)<=16384
            sent.append(request)
            header=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: '+(b'close' if closing else b'keep-alive')+b'\r\n'
            if framing=='length':
                header+=b'Content-Length: '+str(len(raw) if declared is None else declared).encode()+b'\r\n';encoded=raw
            elif framing=='chunked':
                header+=b'Transfer-Encoding: chunked\r\n'
                blocks=[raw[i:i+4093] for i in range(0,len(raw),4093)]
                encoded=b''.join(hex(len(b))[2:].encode()+b'\r\n'+b+b'\r\n' for b in blocks)+(b'' if truncated else b'0\r\n\r\n')
            else:encoded=raw
            peer.sendall(header+b'\r\n'+(b'' if stall else encoded))
            if stall:stop.wait(3)
            elif closing or framing=='eof' or truncated:peer.shutdown(socket.SHUT_WR)
            else:stop.wait(3)
        except OSError:
            # Fixture cleanup may interrupt a blocked peer after intentional rejection.
            assert stop.is_set()
    thread=threading.Thread(target=serve,daemon=True);thread.start()
    def connect(conn):
        assert type(conn) is http.client.HTTPSConnection and conn.host==lane.SOURCE['host']
        if journal is not None:
            rows=[json.loads(line) for line in journal.read_bytes().splitlines()]
            assert [r['attempt'] for r in rows if 'attempt' in r]==[1]
        assert not states, 'Unexpected retry'
        conn.sock=client
    def observed(conn):
        result=real_response(conn);states.append(dict(detached=conn.sock is None,will_close=result.will_close))
        return result
    try:
        with blocked_network(),monkeypatch.context() as m:
            m.setattr(http.client.HTTPSConnection,'connect',connect)
            m.setattr(http.client.HTTPSConnection,'getresponse',observed)
            yield dict(sent=sent,states=states)
    finally:
        stop.set()
        for sock in (client,peer):
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            sock.close()
        thread.join(3);assert not thread.is_alive()


def transport():return lane.HttpsTransport(lambda:SECRET,version=lane.LISTING_VERSION)


@pytest.mark.parametrize('closing',[False,True])
@pytest.mark.parametrize('framing',['length','chunked','eof'])
def test_actual_acquisition_retains_complete_original_and_one_attempt(monkeypatch,tmp_path,closing,framing):
    plan,auth,config=fixture(tmp_path);raw=b' '*17000+json.dumps(response(),indent=2).encode()+b'\n'
    journal=tmp_path/(plan['sha256']+'.nfl-odds-attempts.jsonl')
    with local_http(monkeypatch,raw,closing=closing,framing=framing,journal=journal) as wire:
        with lane.controlled(plan,config,hashlib.sha256(config).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security) as approved:
            result=lane.acquire(plan,approved,root=tmp_path,transport=transport(),wall=lambda:NOW)
            p=result['payload'];assert p['status']=='CAPTURED_UNREVIEWED',p['reason']
            assert p['attempted']==1 and lane.custody.read(p['object'])[0]==raw
            assert p['object']['body_sha256']==hashlib.sha256(raw).hexdigest()
            path=tmp_path/(plan['sha256']+'.nfl-odds-unreviewed.json')
            assert lane.read_back(path,journal)==result
            before=journal.read_bytes()
            with pytest.raises(ValueError,match='ALREADY_ATTEMPTED_NO_RESUME'):
                lane.acquire(plan,approved,root=tmp_path,transport=transport(),wall=lambda:NOW)
            assert before==journal.read_bytes() and len(wire['sent'])==1
            assert not any(p[k] for k in ('owner_reviewed','source_acceptance','inference','public_export','qualified'))
            assert p['action']=='PASS' and p['stake']==0 and SECRET.encode() not in path.read_bytes()+before
    assert wire['states'][0]['detached']==(closing or framing=='eof')


@pytest.mark.parametrize('closing',[False,True])
@pytest.mark.parametrize('bad',['length','chunked','json','oversize'])
def test_actual_rejection_preserves_spent_attempt_and_failure_readback(monkeypatch,tmp_path,closing,bad):
    plan,auth,config=fixture(tmp_path);raw=json.dumps(response()).encode()
    if bad=='json':raw=b'SYNTHETIC not JSON'
    if bad=='oversize':raw=b' '*(lane.LIMITS['max_body_bytes']+1)
    journal=tmp_path/(plan['sha256']+'.nfl-odds-attempts.jsonl')
    with local_http(monkeypatch,raw,closing=closing,framing='chunked' if bad=='chunked' else 'length',
                    declared=len(raw)+1 if bad=='length' else None,truncated=bad in {'length','chunked'},journal=journal) as wire:
        with lane.controlled(plan,config,hashlib.sha256(config).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security) as approved:
            result=lane.acquire(plan,approved,root=tmp_path,transport=transport(),wall=lambda:NOW)
            assert result['payload']['status']=='INCOMPLETE' and result['payload']['attempted']==1
            assert result['payload']['reason'].startswith('NFL_ACQUIRE_')
            path=tmp_path/(plan['sha256']+'.nfl-odds-unreviewed.json')
            assert lane.read_back(path,journal)==result and len(wire['sent'])==1
            before=journal.read_bytes()
            with pytest.raises(ValueError,match='ALREADY_ATTEMPTED_NO_RESUME'):
                lane.acquire(plan,approved,root=tmp_path,transport=transport(),wall=lambda:NOW)
            assert before==journal.read_bytes()


@pytest.mark.parametrize('closing',[False,True])
def test_detached_read_timeout_is_bounded_and_cleanup_interrupts(monkeypatch,tmp_path,closing):
    plan,_,_=fixture(tmp_path)
    with local_http(monkeypatch,b'[]',closing=closing,stall=True):
        start=time.monotonic();deadline=start+.15
        stream=transport()(plan['payload']['request'],lane.LIMITS,deadline,time.monotonic)
        try:
            with pytest.raises(Exception):lane.read_body(stream,deadline,time.monotonic)
        finally:stream.close()
        assert time.monotonic()-start<2


@pytest.mark.parametrize('expired',[False,True])
def test_completion_still_checks_original_deadline(monkeypatch,tmp_path,expired):
    plan,_,_=fixture(tmp_path);clock=[10.0];deadline=20.0
    with local_http(monkeypatch,b'[]',closing=True):
        stream=transport()(plan['payload']['request'],lane.LIMITS,deadline,lambda:clock[0])
        try:
            chunks=stream.chunks();assert next(chunks)==b'[]';clock[0]=deadline if expired else deadline-.001
            if expired:
                with pytest.raises(Exception,match='NFL_ACQUIRE_DEADLINE'):next(chunks)
            else:
                with pytest.raises(StopIteration):next(chunks)
        finally:stream.close()


@pytest.mark.parametrize('closing',[False,True])
def test_cancelled_stream_cannot_report_completion(monkeypatch,tmp_path,closing):
    plan,_,_=fixture(tmp_path)
    with local_http(monkeypatch,b'[]',closing=closing):
        stream=transport()(plan['payload']['request'],lane.LIMITS,time.monotonic()+3,time.monotonic)
        stream.close()
        with pytest.raises(Exception,match='NFL_ACQUIRE_DEADLINE'):next(stream.chunks())


@pytest.mark.parametrize('closing',[False,True])
def test_synthetic_cli_real_http_framing_and_private_readback(monkeypatch,tmp_path,capsys,closing):
    from app_core import ncaaf_pilot_setup
    plan,auth,config=fixture(tmp_path);paths=[tmp_path/(n+'.json') for n in ('SYNTHETIC-plan','SYNTHETIC-auth','SYNTHETIC-config')]
    for path,raw in zip(paths,(json.dumps(plan).encode(),json.dumps(auth).encode(),config)):path.write_bytes(raw)
    journal=tmp_path/(plan['sha256']+'.nfl-odds-attempts.jsonl')
    with local_http(monkeypatch,json.dumps(response()).encode(),closing=closing,journal=journal):
        # Same labelled synthetic CLI harness as the preserved original tests.
        # Runtime fail-closed behavior is asserted separately below.
        monkeypatch.setattr(cli,'verify_runtime',lambda:None)
        monkeypatch.setattr(cli,'private_credential',lambda:SECRET)
        monkeypatch.setattr(ncaaf_pilot_setup,'inspect_security',security)
        monkeypatch.setattr(lane,'controlled',partial(lane.controlled,wall=lambda:NOW))
        monkeypatch.setattr(lane,'acquire',partial(lane.acquire,wall=lambda:NOW))
        assert cli.main([str(paths[0]),'--execute','--authorization',str(paths[1]),'--custody-root',str(tmp_path),
                         '--controlled-bindings',str(paths[2]),'--bindings-sha256',hashlib.sha256(config).hexdigest()])==0
        report=json.loads(capsys.readouterr().out)
        retained=lane.read_back(tmp_path/(plan['sha256']+'.nfl-odds-unreviewed.json'),journal)
        assert report==lane.execution_report(retained) and retained['payload']['status']=='CAPTURED_UNREVIEWED'


def test_real_runtime_gate_rejects_blocked_test_environment(monkeypatch,tmp_path,capsys):
    plan,_,_=fixture(tmp_path);path=tmp_path/'SYNTHETIC-plan.json';path.write_text(json.dumps(plan))
    with blocked_network():
        assert cli.main([str(path),'--execute'])==1
    assert json.loads(capsys.readouterr().out)==dict(status='REJECTED',reason='NFL_ACQUIRE_RUNTIME_UNSUPPORTED')
    assert not list(tmp_path.glob('*.jsonl'))
