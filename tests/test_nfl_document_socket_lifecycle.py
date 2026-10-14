"""Real local sockets and http.client; no external DNS, TLS or HTTP request.

The only synthetic connection boundary installs an already-connected socketpair.
The real HTTPSConnection request/getresponse, HTTPResponse framing, socket file
ownership and timed worker are exercised rather than replaced by BytesIO.
"""
from contextlib import contextmanager
from functools import partial
import hashlib
import http.client
import json
import socket
import threading
import time

import pytest

from app_core import nfl_document_acquisition as lane
from scripts import nfl_document_acquisition as cli
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_document_acquisition import fixture, security, NOW, URL


BODY = b'<!DOCTYPE html><html><body>SYNTHETIC complete original terms.</body></html>'


@contextmanager
def local_transport(monkeypatch, body=BODY, *, closing=True, framing='length', declared=None,
                    stall=False, end_body=False):
    # Windows implements socketpair with local loopback; conftest still denies
    # external connections. Create this pair before the stricter boundary block.
    read_socket, peer = socket.socketpair()
    stop = threading.Event()
    sent = []
    errors = []

    def serve():
        try:
            peer.settimeout(2)
            request = b''
            while b'\r\n\r\n' not in request:
                request += peer.recv(4096)
            sent.append(request)
            headers = b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n'
            headers += b'Connection: close\r\n' if closing else b'Connection: keep-alive\r\n'
            if framing == 'length':
                headers += b'Content-Length: ' + str(len(body) if declared is None else declared).encode() + b'\r\n'
                encoded = body
            elif framing == 'chunked':
                headers += b'Transfer-Encoding: chunked\r\n'
                encoded = hex(len(body))[2:].encode() + b'\r\n' + body + b'\r\n0\r\n\r\n'
            else:
                encoded = body
            peer.sendall(headers + b'\r\n' + encoded)
            if stall:
                stop.wait(2)
            elif closing or end_body:
                peer.shutdown(socket.SHUT_WR)
            else:
                stop.wait(2)
        except OSError as exc:
            if not stop.is_set():
                errors.append(type(exc).__name__)

    server = threading.Thread(target=serve, daemon=True)
    server.start()

    def connect(conn):
        assert type(conn) is http.client.HTTPSConnection
        assert conn.host == 'ludlow-filings.s3.us-east-1.amazonaws.com'
        conn.sock = read_socket  # Synthetic DNS/TCP/TLS boundary only.

    try:
        with blocked_network():
            with monkeypatch.context() as m:
                m.setattr(socket, 'getaddrinfo', lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))])
                m.setattr(http.client.HTTPSConnection, 'connect', connect)
                yield read_socket, sent, errors
    finally:
        stop.set()
        for sock in (read_socket, peer):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        server.join(2)
        assert not server.is_alive()


@pytest.mark.parametrize('closing', [False, True])
@pytest.mark.parametrize('size', [len(BODY), 8192, 8193])
def test_real_content_length_lifecycle_through_cli_custody_readback(monkeypatch, tmp_path, capsys, closing, size):
    from app_core import ncaaf_pilot_setup
    body = BODY + b' ' * (size - len(BODY))
    plan, auth, config = fixture(tmp_path, format='html')
    plan_path = tmp_path / 'SYNTHETIC-plan.json'
    plan_path.write_text(json.dumps(plan))
    config_path = tmp_path / 'SYNTHETIC-bindings.json'
    config_path.write_bytes(config)
    with local_transport(monkeypatch, body, closing=closing) as (sock, sent, errors):
        monkeypatch.setattr(cli, 'verify_runtime', lambda: None)  # Only the labelled synthetic CLI boundary.
        monkeypatch.setattr(ncaaf_pilot_setup, 'inspect_security', security)
        monkeypatch.setattr(lane, 'controlled', partial(lane.controlled, wall=lambda: NOW))
        monkeypatch.setattr(lane, 'acquire', partial(lane.acquire, wall=lambda: NOW))
        result = cli.main([str(plan_path), '--execute', '--controlled-bindings', str(config_path),
                           '--bindings-sha256', hashlib.sha256(config).hexdigest(), '--custody-root', str(tmp_path)])
        report = json.loads(capsys.readouterr().out)
        assert result == 0, report
        assert report['status'] == 'CAPTURED_UNREVIEWED' and report['attempts'] == 1
        bundle = lane.read_back(tmp_path / (plan['sha256'] + '.documents-unreviewed.json'),
                                tmp_path / (plan['sha256'] + '.documents-attempts.jsonl'))
        assert lane.read_bundle(bundle)['rules'][0] == body
        assert bundle['payload']['objects'][0]['body_sha256'] == hashlib.sha256(body).hexdigest()
        assert bundle['payload']['inference'] is bundle['payload']['source_acceptance'] is False
        assert bundle['payload']['stake'] == 0 and bundle['payload']['wager_action'] == 'PASS'
        assert len(sent) == 1 and not errors
        assert b'GET /SYNTHETIC.pdf HTTP/1.1\r\n' in sent[0]


@pytest.mark.parametrize('framing', ['chunked', 'eof'])
def test_real_complete_closing_response_other_framing(monkeypatch, framing):
    with local_transport(monkeypatch, framing=framing):
        response = lane.HttpsTransport()(URL, lane.LIMITS, time.monotonic() + 2, time.monotonic)
        try:
            assert b''.join(response.chunks()) == BODY
        finally:
            response.close()


@pytest.mark.parametrize('closing', [False, True])
def test_real_truncated_content_length_stays_unavailable_and_spent(monkeypatch, tmp_path, closing):
    plan, auth, config = fixture(tmp_path, format='html')
    # The peer really ends the body, even for a keep-alive header.
    with local_transport(monkeypatch, closing=closing, end_body=True, declared=len(BODY) + 7) as (_, sent, errors):
        with lane.controlled(plan, config, hashlib.sha256(config).hexdigest(), root=tmp_path, wall=lambda: NOW, security=security):
            result = lane.acquire(plan, auth, root=tmp_path, wall=lambda: NOW)
        assert result['payload']['status'] == 'INCOMPLETE'
        assert result['payload']['reason'] == 'DOCUMENT_INCOMPLETE'
        assert not result['payload']['objects'] and result['payload']['attempted'] == 1
        journal = tmp_path / (plan['sha256'] + '.documents-attempts.jsonl')
        original = journal.read_bytes()
        assert lane.read_back(tmp_path / (plan['sha256'] + '.documents-unreviewed.json'), journal) == result
        with lane.controlled(plan, config, hashlib.sha256(config).hexdigest(), root=tmp_path, wall=lambda: NOW, security=security):
            with pytest.raises(ValueError, match='DOCUMENT_SPENT_NO_RESUME'):
                lane.acquire(plan, auth, root=tmp_path, wall=lambda: NOW)
        assert journal.read_bytes() == original and len(sent) == 1 and not errors


@pytest.mark.parametrize('gate', ['expired', 'cancelled'])
def test_completion_does_not_bypass_deadline_or_cancellation(monkeypatch, gate):
    with local_transport(monkeypatch):
        response = lane.HttpsTransport()(URL, lane.LIMITS, time.monotonic() + 2, time.monotonic)
        chunks = response.chunks()
        try:
            assert next(chunks) == BODY
            assert response.response.isclosed() and response.read_socket.fileno() == -1
            if gate == 'expired':
                response.deadline = time.monotonic() - 1
            else:
                response.cancel.set()
            with pytest.raises(lane.diagnostics.Failure) as failure:
                next(chunks)
            assert str(failure.value) == 'DOCUMENT_READ_FAILURE'
            assert failure.value.diagnostic['exception_category'] == 'TIMEOUT'
        finally:
            response.close()


def test_pending_real_read_stops_at_deadline_without_waiting_for_peer(monkeypatch):
    with local_transport(monkeypatch, body=b'', declared=len(BODY), stall=True):
        response = lane.HttpsTransport()(URL, lane.LIMITS, time.monotonic() + 2, time.monotonic)
        started = time.monotonic()
        response.deadline = started + .15
        try:
            with pytest.raises(lane.diagnostics.Failure, match='DOCUMENT_READ_FAILURE') as failure:
                list(response.chunks())
            # OS read timeout and the caller deadline can race. Both must stop;
            # explicit cancellation is separately asserted above, not inferred.
            assert failure.value.diagnostic['exception_category'] == 'TIMEOUT'
            assert time.monotonic() - started < 1
        finally:
            response.close()
        assert response.response.isclosed()


def test_unexpected_closed_socket_is_not_suppressed(monkeypatch):
    with local_transport(monkeypatch, closing=False):
        response = lane.HttpsTransport()(URL, lane.LIMITS, time.monotonic() + 2, time.monotonic)
        try:
            # Deliberately damage a live unread response, not normal EOF framing.
            response.response.fp.close()
            response.read_socket.close()
            assert not response.response.isclosed()
            with pytest.raises(lane.diagnostics.Failure, match='DOCUMENT_READ_FAILURE') as failure:
                list(response.chunks())
            assert failure.value.diagnostic['exception_category'] == 'OS'
            assert failure.value.diagnostic['os_errno'] in {9, 10038}
        finally:
            # The deliberate broken-file attack above is not a normal close;
            # restore only this synthetic object's pointer for safe cleanup.
            response.response.fp = None
            response.close()
