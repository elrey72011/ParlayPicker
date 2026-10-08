"""Repository tests deny external transports; local service fixtures remain usable.

Production modules and clocks are untouched. Requests fail with their normal
connection error so callers retain real outage diagnostics. Tests may provide
explicit synthetic clients/responses through their existing fixtures.
"""
import errno
import ipaddress
import socket
from urllib.parse import urlsplit

import requests

_original_resolve = socket.getaddrinfo
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_sendto = socket.socket.sendto
_original_request = requests.sessions.Session.request


def _loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def _resolve(host, *args, **kwargs):
    if not _loopback(host):
        raise socket.gaierror(socket.EAI_FAIL, "External network blocked in offline tests")
    return _original_resolve(host, *args, **kwargs)


def _connect(self, address):
    if isinstance(address, tuple) and not _loopback(address[0]):
        raise ConnectionRefusedError(errno.ECONNREFUSED, "External network blocked in offline tests")
    return _original_connect(self, address)


def _connect_ex(self, address):
    if isinstance(address, tuple) and not _loopback(address[0]):
        return errno.ENETUNREACH
    return _original_connect_ex(self, address)


def _sendto(self, *args):
    address = args[-1]
    if isinstance(address, tuple) and not _loopback(address[0]):
        raise ConnectionRefusedError(errno.ECONNREFUSED, "External network blocked in offline tests")
    return _original_sendto(self, *args)


def _request(self, method, url, *args, **kwargs):
    if not _loopback(urlsplit(url).hostname):
        raise requests.ConnectionError("External network blocked in offline tests")
    return _original_request(self, method, url, *args, **kwargs)


def pytest_sessionstart(session):
    socket.getaddrinfo = _resolve
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.socket.sendto = _sendto
    requests.sessions.Session.request = _request


def pytest_sessionfinish(session, exitstatus):
    socket.getaddrinfo = _original_resolve
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
    socket.socket.sendto = _original_sendto
    requests.sessions.Session.request = _original_request
