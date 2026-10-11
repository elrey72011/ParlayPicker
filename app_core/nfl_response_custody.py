"""Bounded retained NFL response bytes. No acquisition, store or acceptance."""
import base64
import csv
import hashlib
import io
import json
from urllib.parse import urlsplit
from app_core.research_estimate_trace import encode
from app_core.producer_provenance import clock

VERSION = 'nfl-original-response-custody-v1'
MAX_BODY = 512 * 1024
MAX_OBJECTS = 16
MAX_TOTAL = 2 * 1024 * 1024


def require(ok, code):
    if not ok:
        raise ValueError(code)


def known(value):
    from app_core.source_evidence_intake import absent
    return isinstance(value, str) and bool(value.strip()) and not absent(value.strip())


def digest(obj):
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def unique(pairs):
    out = {}
    for k, v in pairs:
        require(k not in out, 'NFL_PRIVATE_RESPONSE_SCHEMA')
        out[k] = v
    return out


def retained(body, *, receipt):
    """Wrap supplied original decoded bytes; receipt clocks are supplied facts."""
    require(isinstance(body, bytes) and 0 < len(body) <= MAX_BODY, 'NFL_PRIVATE_RESPONSE_SIZE')
    value = dict(version=VERSION, body_sha256=hashlib.sha256(body).hexdigest(),
                 bytes_base64=base64.b64encode(body).decode(), receipt=receipt)
    read(value)
    return value


def read(obj):
    require(isinstance(obj, dict) and set(obj) == {'version', 'body_sha256', 'bytes_base64', 'receipt'}
            and obj['version'] == VERSION, 'NFL_PRIVATE_CUSTODY_MISSING')
    try:
        raw = base64.b64decode(obj['bytes_base64'], validate=True)
    except (ValueError, TypeError):
        raise ValueError('NFL_PRIVATE_CUSTODY_CORRUPT') from None
    require(0 < len(raw) <= MAX_BODY, 'NFL_PRIVATE_RESPONSE_SIZE')
    require(hashlib.sha256(raw).hexdigest() == obj['body_sha256'], 'NFL_PRIVATE_CUSTODY_CORRUPT')
    r = obj['receipt']
    require(isinstance(r, dict) and set(r) == set('source_id endpoint format evidence_label request_started_at received_at observed_at observation_meaning provider_clock_meaning'.split()),
            'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    require(r['evidence_label'] in {'RETAINED', 'SYNTHETIC'} and known(r['source_id']), 'NFL_PRIVATE_RESPONSE_SCHEMA')
    u = urlsplit(r['endpoint'])
    require(u.scheme in {'https', 'http'} and bool(u.hostname) and not u.username and not u.password
            and not u.query and not u.fragment, 'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')
    require(r['format'] in {'csv', 'json'} and
            r['observation_meaning'] == 'actual_local_first_observation_of_complete_decoded_body'
            and r['provider_clock_meaning'] in {'publisher_available_at_per_record', 'provider_market_last_update', 'official_schedule_effective_at'},
            'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    times = [clock(r[k]) for k in ('request_started_at', 'received_at', 'observed_at')]
    require(all(times) and times[0] <= times[1] <= times[2], 'NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT')
    # Secrets must never be retained even if echoed by a provider. Reject, do not redact originals.
    import re
    require(not re.search(rb'(?i)(api[_-]?key|access[_-]?token|authorization|client_secret|password)\s*["\x27:=]', raw),
            'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')
    try:
        text = raw.decode('utf-8')
        if r['format'] == 'json':
            decoded = json.loads(text, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        else:
            reader = csv.DictReader(io.StringIO(text, newline=''))
            require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), 'NFL_PRIVATE_RESPONSE_SCHEMA')
            decoded = list(reader)
            require(decoded and all(None not in row and all(v is not None for v in row.values()) for row in decoded), 'NFL_PRIVATE_RESPONSE_SCHEMA')
    except (UnicodeError, ValueError, TypeError, csv.Error):
        raise ValueError('NFL_PRIVATE_RESPONSE_SCHEMA') from None
    return raw, decoded


def objects(items):
    require(isinstance(items, dict) and 0 < len(items) <= MAX_OBJECTS, 'NFL_PRIVATE_CUSTODY_MISSING')
    decoded = {}
    total = 0
    identities = set()
    for key, obj in items.items():
        raw, value = read(obj)
        require(obj['body_sha256'] not in identities, 'NFL_PRIVATE_DUPLICATE_RESPONSE')
        identities.add(obj['body_sha256'])
        total += len(raw)
        decoded[key] = value
    require(total <= MAX_TOTAL, 'NFL_PRIVATE_RESPONSE_SIZE')
    return decoded
