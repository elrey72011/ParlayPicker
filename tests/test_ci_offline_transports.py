"""Synthetic transport acceptance; no external request is made."""
import errno
import socket

import pytest
import requests
from pathlib import Path


@pytest.mark.parametrize("host", ["provider.example.invalid", "8.8.8.8", "2001:4860:4860::8888"])
def test_external_dns_rejected_before_resolution(host):
    with pytest.raises(socket.gaierror, match="External network blocked"):
        socket.getaddrinfo(host, 443)


def test_external_http_rejected_before_transport():
    with pytest.raises(requests.ConnectionError, match="External network blocked"):
        requests.get("https://provider.example.invalid/retained", timeout=1)


def test_external_socket_and_datagram_rejected():
    with socket.socket() as client:
        with pytest.raises(ConnectionRefusedError):
            client.connect(("8.8.8.8", 443))
        assert client.connect_ex(("8.8.8.8", 443)) == errno.ENETUNREACH
    with socket.socket(type=socket.SOCK_DGRAM) as client:
        with pytest.raises(ConnectionRefusedError):
            client.sendto(b"synthetic", ("8.8.8.8", 53))


def test_loopback_service_delegation(monkeypatch, request):
    offline = request.config.pluginmanager.get_plugin(str(Path(__file__).parent / "conftest.py"))
    assert offline is not None
    calls = []
    monkeypatch.setattr(offline, "_original_resolve", lambda *a, **k: calls.append(a) or [])
    assert offline._resolve("localhost", 5432) == []
    assert offline._resolve("127.0.0.1", 5432) == []
    assert offline._resolve("::1", 5432) == []
    marker = object()
    monkeypatch.setattr(offline, "_original_request", lambda *a, **k: marker)
    assert offline._request(None, "GET", "http://localhost:5432/") is marker
    assert len(calls) == 3


def test_existing_explicit_synthetic_response_fixture(monkeypatch):
    marker = object()
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: marker)
    assert requests.get("https://provider.example.invalid/synthetic") is marker
