"""Labelled synthetic local sockets; no external requests or authentic evidence."""
from contextlib import contextmanager
import hashlib
import http.client
from pathlib import Path
import sys
import threading
import time

import pytest

import test_ncaaf_transport_diagnostics as diagnostic_fixture
from test_ncaaf_reviewed_target_pipeline import (
    test_actual_manual_operator_two_prompts_native_capture_and_same_context_readback as _operator_assertions,
)


@pytest.fixture
def server_threads(monkeypatch):
    """Observe only this fixture's workers; all other starts stay native."""
    original = threading.Thread.start
    started = []
    source = Path(diagnostic_fixture.__file__).resolve()

    def start(thread, *args, **kwargs):
        target = getattr(thread, '_target', None)
        code = getattr(target, '__code__', None)
        if code is not None and code.co_name == 'serve' and Path(code.co_filename).resolve() == source:
            started.append(thread)
        return original(thread, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, 'start', start)
        yield started
        assert all(not worker.is_alive() for worker in started)
    assert threading.Thread.start is original


@pytest.mark.parametrize('closing', [False, True])
def test_delayed_initial_and_later_connections_preserve_originals(monkeypatch, server_threads, closing):
    bodies = [b'{"SYNTHETIC":1}', b'{"SYNTHETIC":2}']
    native_connect = http.client.HTTPSConnection.connect
    native_request = http.client.HTTPSConnection.request
    with diagnostic_fixture.local_wire(monkeypatch, bodies, closing=closing) as wire:
        time.sleep(2.1)  # Longer than the unchanged server receive bound.
        assert server_threads == [] and wire['connections'] == []
        for index, body in enumerate(bodies):
            if index:
                time.sleep(2.1)
                assert len(server_threads) == index
            connection = http.client.HTTPSConnection('synthetic.invalid', timeout=2)
            try:
                connection.request('GET', '/SYNTHETIC')
                response = connection.getresponse()
                received = response.read()
                assert response.status == 200 and received == body
                assert hashlib.sha256(received).digest() == hashlib.sha256(body).digest()
                assert wire['pairs'][index][1].gettimeout() == 2
                assert len(server_threads) == len(wire['connections']) == index + 1
            finally:
                connection.close()
        assert len(wire['sent']) == 2
    assert all(sock.fileno() == -1 for pair in wire['pairs'] for sock in pair)
    assert all(not worker.is_alive() for worker in server_threads)
    assert http.client.HTTPSConnection.connect is native_connect
    assert http.client.HTTPSConnection.request is native_request


@pytest.mark.parametrize('after_connect,exception', [
    (False, None), (False, RuntimeError), (False, KeyboardInterrupt),
    (True, RuntimeError), (True, KeyboardInterrupt),
])
def test_failure_cancellation_and_unused_worker_cleanup(monkeypatch, server_threads, after_connect, exception):
    native_connect = http.client.HTTPSConnection.connect
    native_request = http.client.HTTPSConnection.request
    wire = None
    connection = None

    def operation():
        nonlocal wire, connection
        with diagnostic_fixture.local_wire(monkeypatch, [b'[]'] * 4) as active:
            wire = active
            assert server_threads == []
            if after_connect:
                connection = http.client.HTTPSConnection('synthetic.invalid', timeout=2)
                connection.connect()
                assert len(server_threads) == len(wire['connections']) == 1
            if exception is not None:
                raise exception('SYNTHETIC_EXIT')

    try:
        if exception is None:
            operation()
        else:
            with pytest.raises(exception, match='SYNTHETIC_EXIT'):
                operation()
    finally:
        if connection is not None:
            connection.close()
    assert wire is not None and wire['sent'] == []
    assert len(server_threads) == int(after_connect)
    assert all(sock.fileno() == -1 for pair in wire['pairs'] for sock in pair)
    assert all(not worker.is_alive() for worker in server_threads)
    assert http.client.HTTPSConnection.connect is native_connect
    assert http.client.HTTPSConnection.request is native_request


@pytest.mark.parametrize('delay', [0, 2.1], ids=['normal', 'delayed-preflight'])
def test_actual_operator_assertions_keep_authentication_and_cleanup(monkeypatch, tmp_path, capsys, server_threads, delay):
    original_wire = diagnostic_fixture.local_wire
    original_profile = sys.getprofile()
    operator = Path(__file__).resolve().parents[1] / 'scripts/ncaaf_reviewed_target_operator.py'
    events = []
    delayed = False

    @contextmanager
    def observed_wire(*args, **kwargs):
        nonlocal delayed
        with original_wire(*args, **kwargs) as wire:
            def profile(frame, event, value):
                nonlocal delayed
                if frame.f_code.co_name == 'verify_checkout' and Path(frame.f_code.co_filename).resolve() == operator:
                    if event in ('call', 'return'):
                        events.append(event)  # Fixed codes only; no arguments or values.
                    if event == 'call' and not delayed:
                        delayed = True
                        assert server_threads == []
                        time.sleep(delay)
                if original_profile is not None:
                    original_profile(frame, event, value)
            sys.setprofile(profile)
            try:
                yield wire
            finally:
                sys.setprofile(original_profile)

    with monkeypatch.context() as patch:
        patch.setattr(diagnostic_fixture, 'local_wire', observed_wire)
        _operator_assertions(monkeypatch, tmp_path, capsys)
    assert delayed and events.count('call') == events.count('return') >= 3
    assert len(server_threads) == 4 and all(not worker.is_alive() for worker in server_threads)
    assert diagnostic_fixture.local_wire is original_wire and sys.getprofile() is original_profile
