"""Private allowlisted failure facts. Never serialize exception text or headers."""
import http.client
import math
import socket
import ssl

VERSION = 'nfl-acquisition-failure-diagnostics-v1'
UNKNOWN = 'UNKNOWN'
PHASES = frozenset(('CONNECTION_SETUP', 'HTTPS_CONNECT', 'REQUEST_SEND', 'RESPONSE_RECEIVE',
                    'RESPONSE_HEADERS', 'BODY_READ', 'BODY_VALIDATION', 'RESPONSE_VALIDATION',
                    'AUTHORIZATION', 'JOURNAL', 'CUSTODY', 'TRANSPORT', 'UNKNOWN'))
CATEGORIES = frozenset(('TLS_CERTIFICATE_VERIFICATION', 'TLS', 'DNS', 'TIMEOUT', 'CONNECTION',
                        'HTTP_PROTOCOL', 'OS', 'POLICY', 'UNKNOWN'))
FIELDS = frozenset(('version', 'phase', 'exception_category', 'os_errno', 'os_winerror',
                    'tls_error_code', 'tls_verify_code', 'http_status', 'elapsed_seconds'))


def integer(value):
    return value if type(value) is int and -(2**31) <= value < 2**32 else UNKNOWN


def status(value):
    return value if type(value) is int and 100 <= value <= 599 else UNKNOWN


def elapsed(start, end):
    if type(start) in (int, float) and type(end) in (int, float):
        value = end - start
        if math.isfinite(value) and 0 <= value <= 86400:
            return round(value, 6)
    return UNKNOWN


def category(exc):
    # Class membership only: no class names, messages, SSL reason strings,
    # certificate text, URLs, partial bodies or arbitrary exception attributes.
    if isinstance(exc, ssl.SSLCertVerificationError): return 'TLS_CERTIFICATE_VERIFICATION'
    if isinstance(exc, ssl.SSLError): return 'TLS'
    if isinstance(exc, socket.gaierror): return 'DNS'
    if isinstance(exc, TimeoutError): return 'TIMEOUT'
    if isinstance(exc, ConnectionError): return 'CONNECTION'
    if isinstance(exc, http.client.HTTPException): return 'HTTP_PROTOCOL'
    if isinstance(exc, OSError): return 'OS'
    if isinstance(exc, ValueError) and str(exc) in {'NCAAF_PILOT_DEADLINE', 'NFL_ACQUIRE_DEADLINE'}: return 'TIMEOUT'
    if isinstance(exc, ValueError): return 'POLICY'
    return UNKNOWN


def facts(exc, *, phase, http_status=UNKNOWN, started=None, ended=None):
    return dict(version=VERSION, phase=phase if phase in PHASES else UNKNOWN,
                exception_category=category(exc),
                os_errno=integer(getattr(exc, 'errno', None)) if isinstance(exc, OSError) else UNKNOWN,
                os_winerror=integer(getattr(exc, 'winerror', None)) if isinstance(exc, OSError) else UNKNOWN,
                tls_error_code=integer(exc.errno) if isinstance(exc, ssl.SSLError) else UNKNOWN,
                tls_verify_code=integer(getattr(exc, 'verify_code', None)) if isinstance(exc, ssl.SSLCertVerificationError) else UNKNOWN,
                http_status=status(http_status), elapsed_seconds=elapsed(started, ended))


def checked(value):
    """Reject extra/free-text fields before any private execution report."""
    if not isinstance(value, dict) or set(value) != FIELDS or value['version'] != VERSION:
        raise ValueError('NFL_ACQUIRE_DIAGNOSTICS_INVALID')
    if value['phase'] not in PHASES or value['exception_category'] not in CATEGORIES:
        raise ValueError('NFL_ACQUIRE_DIAGNOSTICS_INVALID')
    for key in ('os_errno', 'os_winerror', 'tls_error_code', 'tls_verify_code'):
        if value[key] != UNKNOWN and integer(value[key]) == UNKNOWN:
            raise ValueError('NFL_ACQUIRE_DIAGNOSTICS_INVALID')
    if value['http_status'] != UNKNOWN and status(value['http_status']) == UNKNOWN:
        raise ValueError('NFL_ACQUIRE_DIAGNOSTICS_INVALID')
    time = value['elapsed_seconds']
    if time != UNKNOWN and (type(time) not in (int, float) or not math.isfinite(time) or not 0 <= time <= 86400):
        raise ValueError('NFL_ACQUIRE_DIAGNOSTICS_INVALID')
    return dict(value)


class Failure(ValueError):
    """Only the stable rejection code is exposed by str(); facts are separate."""
    def __init__(self, code, diagnostic):
        super().__init__(code)
        self.diagnostic = checked(diagnostic)
