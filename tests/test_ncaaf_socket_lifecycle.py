"""Real local socket/http.client lifecycle; labelled synthetic evidence only.

Only DNS/TCP/TLS connect is replaced with already-connected socket pairs. The
real HTTPSConnection request, HTTPResponse framing, timed workers, acquisition,
durable journal, original-byte custody and read-back remain in use.
"""
import base64
from contextlib import contextmanager
import hashlib
import http.client
import json
import socket
import threading
import time

import pytest

from app_core import ncaaf_pilot as pilot, ncaaf_pilot_discovery as discovery
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic
from test_ncaaf_pilot import Clock, fixture
from test_ncaaf_pilot_discovery import fixture as discovery_fixture, body, event


@contextmanager
def local_http(monkeypatch, bodies, *, closing=True, framing='length', declared=None,
               stall=False, end_body=False, journal=None):
    # Windows socketpair uses loopback; create pairs before the stricter block.
    pairs=[socket.socketpair() for _ in bodies]
    stop=threading.Event();sent=[];connected=[];responses=[];errors=[]
    original_response=http.client.HTTPSConnection.getresponse
    def observed_response(conn):
        response=original_response(conn)
        responses.append((conn,response))
        return response
    def serve(peer, raw):
        try:
            peer.settimeout(3);request=b''
            while b'\r\n\r\n' not in request:
                part=peer.recv(4096)
                if not part: return
                request+=part
                assert len(request)<=16384
            sent.append(request)
            header=b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
            header+=b'Connection: close\r\n' if closing else b'Connection: keep-alive\r\n'
            if framing=='length':
                header+=b'Content-Length: '+str(len(raw) if declared is None else declared).encode()+b'\r\n'
                encoded=raw
            elif framing=='chunked':
                header+=b'Transfer-Encoding: chunked\r\n'
                blocks=[raw[i:i+4093] for i in range(0,len(raw),4093)]
                encoded=b''.join(hex(len(b))[2:].encode()+b'\r\n'+b+b'\r\n' for b in blocks)+b'0\r\n\r\n'
            else: encoded=raw
            peer.sendall(header+b'\r\n'+encoded)
            if stall:stop.wait(3)
            elif closing or end_body:peer.shutdown(socket.SHUT_WR)
            else:stop.wait(3)
        except OSError as exc:
            if not stop.is_set():errors.append(type(exc).__name__)
    workers=[threading.Thread(target=serve,args=(peer,raw),daemon=True) for (_,peer),raw in zip(pairs,bodies)]
    for worker in workers:worker.start()
    def connect(conn):
        assert type(conn) is http.client.HTTPSConnection
        assert conn.host in pilot.HOSTS.values()
        if journal is not None:
            records=[json.loads(line) for line in journal.read_bytes().splitlines()]
            assert len([r for r in records if 'attempt' in r])==len(connected)+1
        assert len(connected)<len(pairs),'Unexpected retry or extra request'
        conn.sock=pairs[len(connected)][0]
        connected.append(conn.host)
    try:
        with blocked_network(),monkeypatch.context() as m:
            m.setattr(http.client.HTTPSConnection,'connect',connect)
            m.setattr(http.client.HTTPSConnection,'getresponse',observed_response)
            yield dict(pairs=pairs,sent=sent,connected=connected,responses=responses,errors=errors)
    finally:
        stop.set()
        for pair in pairs:
            for sock in pair:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                sock.close()
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()


def transport():return pilot.HttpsTransport(lambda provider:'SYNTHETIC_CREDENTIAL_ONLY')


def first_request(synthetic,monkeypatch,tmp_path):
    plan,auth,packet,_,_,clock,_,_=fixture(synthetic,monkeypatch,root=tmp_path)
    return plan,auth,packet,clock,plan['payload']['requests'][0]


@pytest.mark.parametrize('closing',[False,True])
@pytest.mark.parametrize('framing',['length','chunked','eof'])
def test_actual_acquisition_preserves_complete_bytes_and_spent_journal(synthetic,monkeypatch,tmp_path,closing,framing):
    if framing=='eof' and not closing:
        # EOF framing necessarily closes; use HTTP/1.1's inferred close semantics.
        end_body=True
    else:end_body=False
    plan,auth,packet,clock,_=first_request(synthetic,monkeypatch,tmp_path)
    bodies=[base64.b64decode(o['payload']['body_b64']) for o in packet['payload']['response_objects']]
    journal=tmp_path/(plan['sha256']+'.attempts.jsonl')
    with local_http(monkeypatch,bodies,closing=closing,framing=framing,end_body=end_body,journal=journal) as wire:
        result=pilot.acquire(plan,auth,root=tmp_path,transport=transport(),wall=clock.wall,monotonic=clock.monotonic)
        assert result['payload']['status']=='CAPTURED_UNACCEPTED',result['payload']['reason']
        assert result['payload']['attempted_requests']==result['payload']['completed_objects']==len(bodies)
        for item,original in zip(result['payload']['objects'],bodies):
            assert base64.b64decode(item['body_b64'])==original
            assert item['body_bytes']==len(original)
            assert item['body_sha256']==hashlib.sha256(original).hexdigest()
            assert item['requested_at']==item['metadata']['received_at']==clock.wall()
        path=tmp_path/(plan['sha256']+'.unaccepted.json')
        assert path.read_bytes()==pilot.encode(result)
        assert json.loads(path.read_bytes())==result
        before=journal.read_bytes()
        records=[json.loads(line) for line in before.splitlines()]
        assert [r['attempt'] for r in records if 'attempt' in r]==list(range(1,len(bodies)+1))
        with pytest.raises(ValueError,match='ALREADY_ATTEMPTED_NO_RESUME'):
            pilot.acquire(plan,auth,root=tmp_path,transport=transport(),wall=clock.wall,monotonic=clock.monotonic)
        assert journal.read_bytes()==before
        assert len(wire['sent'])==len(wire['connected'])==len(bodies) and not wire['errors']
        assert all(not result['payload'][k] for k in ('accepted','inference','scientific_acceptance','wagering_authority'))
        assert b'SYNTHETIC_CREDENTIAL_ONLY' not in path.read_bytes()+before


@pytest.mark.parametrize('closing',[False,True])
def test_discovery_uses_same_real_transport_and_remains_unaccepted(monkeypatch,tmp_path,closing):
    plan,auth=discovery_fixture(monkeypatch,tmp_path)
    raw=body([event()]);clock=Clock()
    journal=tmp_path/(plan['sha256']+'.discovery-attempts.jsonl')
    with local_http(monkeypatch,[raw],closing=closing) as wire:
        result=discovery.acquire(plan,auth,root=tmp_path,transport=transport(),wall=clock.wall,monotonic=clock.monotonic)
        assert result['payload']['status']=='DISCOVERED_UNACCEPTED',result['payload'].get('reason')
        assert result['payload']['attempted_requests']==1
        saved=discovery.read_bundle(tmp_path/(plan['sha256']+'.discovery-unaccepted.json'))
        assert saved==result
        assert base64.b64decode(result['payload']['objects'][0]['body_b64'])==raw
        assert not result['payload']['accepted'] and not result['payload']['inference']
        assert len(wire['sent'])==1


@pytest.mark.parametrize('closing',[False,True])
@pytest.mark.parametrize('size',[8192,8193,32769,524288])
def test_real_multichunk_length_boundary(monkeypatch,size,closing):
    raw=b'[]'+b' '*(size-2)
    request=dict(id='SYNTHETIC',provider='cfbd',host=pilot.HOSTS['cfbd'],endpoint='games',params=dict(year=2026,seasonType='both',classification='fbs'),max_attempts=1)
    with local_http(monkeypatch,[raw],closing=closing) as wire:
        deadline=time.monotonic()+3
        response=transport()(request,dict(connect_seconds=2,read_seconds=2),deadline,time.monotonic)
        try:assert pilot._read(response,524288,deadline,time.monotonic)==raw
        finally:response.close()
        assert len(wire['sent'])==1 and not wire['errors']


@pytest.mark.parametrize('closing',[False,True])
@pytest.mark.parametrize('attack',['truncated','short_length','malformed_length','oversized_length','oversized_chunked'])
def test_invalid_body_stays_incomplete_and_spent(synthetic,monkeypatch,tmp_path,closing,attack):
    plan,auth,_,clock,_=first_request(synthetic,monkeypatch,tmp_path)
    raw=b'[{"SYNTHETIC":"complete"}]';declared=len(raw);framing='length'
    if attack=='truncated':declared+=9
    elif attack=='short_length':declared-=3
    elif attack=='malformed_length':declared='unknown'
    elif attack=='oversized_length':declared=524289;raw=b''
    else:raw=b'[]'+b' '*524287;framing='chunked'
    with local_http(monkeypatch,[raw],closing=closing,declared=declared,framing=framing,end_body=True) as wire:
        result=pilot.acquire(plan,auth,root=tmp_path,transport=transport(),wall=clock.wall,monotonic=clock.monotonic)
        assert result['payload']['status']=='INCOMPLETE'
        assert result['payload']['attempted_requests']==1 and result['payload']['completed_objects']==0
        expected=('INCOMPLETE_BODY' if attack=='truncated' else 'NCAAF_CUSTODY_BODY_SCHEMA' if attack=='short_length' else 'BODY_LIMIT')
        assert result['payload']['reason'].endswith(expected)
        journal=tmp_path/(plan['sha256']+'.attempts.jsonl');before=journal.read_bytes()
        assert json.loads((tmp_path/(plan['sha256']+'.unaccepted.json')).read_bytes())==result
        with pytest.raises(ValueError,match='ALREADY_ATTEMPTED_NO_RESUME'):
            pilot.acquire(plan,auth,root=tmp_path,transport=transport(),wall=clock.wall,monotonic=clock.monotonic)
        assert journal.read_bytes()==before and len(wire['sent'])==len(wire['connected'])==1


@pytest.mark.parametrize('gate',['before_deadline','at_deadline','after_deadline','cancelled'])
def test_completed_response_checks_clock_and_cancellation_first(synthetic,monkeypatch,tmp_path,gate):
    plan,_,_,clock,request=first_request(synthetic,monkeypatch,tmp_path)
    with local_http(monkeypatch,[b'[]']) as wire:
        stream=transport()(request,plan['payload']['limits'],30,clock.monotonic)
        chunks=stream.chunks()
        try:
            assert next(chunks)==b'[]'
            conn,response=wire['responses'][0]
            assert conn.sock is None and response.isclosed() and wire['pairs'][0][0].fileno()==-1
            clock.elapsed=29.999 if gate=='before_deadline' else 30 if gate=='at_deadline' else 30.001
            if gate=='cancelled':clock.elapsed=0;stream.close()
            if gate=='before_deadline':assert list(chunks)==[]
            else:
                with pytest.raises(ValueError,match='NCAAF_PILOT_DEADLINE'):next(chunks)
        finally:stream.close()


def test_pending_detached_read_deadline_and_cleanup(synthetic,monkeypatch,tmp_path):
    plan,_,_,_,request=first_request(synthetic,monkeypatch,tmp_path)
    with local_http(monkeypatch,[b''],declared=20,stall=True) as wire:
        started=time.monotonic()
        deadline=started+.15
        stream=transport()(request,plan['payload']['limits'],deadline,time.monotonic)
        try:
            with pytest.raises((ValueError,TimeoutError)):
                pilot._read(stream,524288,deadline,time.monotonic)
        finally:stream.close()
        assert time.monotonic()-started<1
        assert wire['responses'][0][1].isclosed() and len(wire['sent'])==1


def test_close_cancels_pending_read_within_shared_deadline(synthetic,monkeypatch,tmp_path):
    plan,_,_,_,request=first_request(synthetic,monkeypatch,tmp_path)
    entered=threading.Event();original=http.client.HTTPResponse.read1;errors=[]
    def observed_read(response,*args):entered.set();return original(response,*args)
    with local_http(monkeypatch,[b''],declared=20,stall=True) as wire:
        monkeypatch.setattr(http.client.HTTPResponse,'read1',observed_read)
        deadline=time.monotonic()+2
        stream=transport()(request,plan['payload']['limits'],deadline,time.monotonic)
        def read():
            try:pilot._read(stream,524288,deadline,time.monotonic)
            except Exception as exc:errors.append(exc)
        worker=threading.Thread(target=read,daemon=True);worker.start()
        assert entered.wait(1)
        started=time.monotonic();stream.close();worker.join(1)
        # Windows may finish an already-pending select at its bounded timeout;
        # cancellation must reject and finish within the original shared budget.
        assert not worker.is_alive() and time.monotonic()<deadline+1
        assert len(errors)==1 and isinstance(errors[0],ValueError) and str(errors[0])=='NCAAF_PILOT_DEADLINE'
        assert len(wire['sent'])==1 and wire['responses'][0][1].isclosed()


def test_unexpected_socket_error_is_not_suppressed(synthetic,monkeypatch,tmp_path):
    plan,_,_,_,request=first_request(synthetic,monkeypatch,tmp_path)
    with local_http(monkeypatch,[b'[]'],closing=False) as wire:
        deadline=time.monotonic()+2
        stream=transport()(request,plan['payload']['limits'],deadline,time.monotonic)
        response=wire['responses'][0][1]
        response.fp.close();wire['pairs'][0][0].close()
        assert not response.isclosed()
        try:
            with pytest.raises(OSError) as error:list(stream.chunks())
            assert error.value.errno in {9,10038}
        finally:response.fp=None;stream.close()


def test_real_read_exception_is_sanitized_in_failure_bundle(synthetic,monkeypatch,tmp_path):
    plan,auth,_,clock,_=first_request(synthetic,monkeypatch,tmp_path)
    with local_http(monkeypatch,[b'[]']) as wire:
        def failure(*args):raise OSError(5,'SYNTHETIC_SENSITIVE_ERROR_CANARY')
        monkeypatch.setattr(http.client.HTTPResponse,'read1',failure)
        result=pilot.acquire(plan,auth,root=tmp_path,transport=transport(),wall=clock.wall,monotonic=clock.monotonic)
        assert result['payload']['status']=='INCOMPLETE' and result['payload']['reason']=='NCAAF_PILOT_TRANSPORT_FAILURE'
        assert result['payload']['attempted_requests']==1 and len(wire['sent'])==1
        assert all(b'SYNTHETIC_SENSITIVE_ERROR_CANARY' not in p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
