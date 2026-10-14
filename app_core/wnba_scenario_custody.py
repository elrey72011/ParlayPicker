"""Bounded retained inputs only. No acquisition, credentials or acceptance."""
import base64
from copy import deepcopy
import hashlib
import json
import re
from urllib.parse import urlsplit

import pandas as pd

VERSION = 'wnba-original-scenario-evidence-v1'
SOURCE = 'balldontlie-wnba-games-v1'
ENDPOINT = 'https://api.balldontlie.io/wnba/v1/games'
MAX_BODY = 512 * 1024
MAX_OBJECTS = 16
MAX_AGGREGATE = 2 * 1024 * 1024
MAX_PACKET = 12 * 1024 * 1024
MAX_SELECTION = 16 * 1024 * 1024


def require(condition, code):
    if not condition:
        raise ValueError('WNBA_SCENARIO_' + code)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'DUPLICATE_FIELD')
        out[key] = value
    return out


def clock(value):
    require(isinstance(value, str) and re.search(r'(Z|[+-]\d\d:\d\d)$', value), 'CLOCK_UNKNOWN')
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise ValueError('WNBA_SCENARIO_CLOCK_UNKNOWN') from None
    require(pd.notna(stamp) and stamp.tzinfo is not None, 'CLOCK_UNKNOWN')
    return stamp.tz_convert('UTC')


def official_url(value):
    require(isinstance(value, str), 'SOURCE_IDENTITY')
    u = urlsplit(value)
    require(u.scheme == 'https' and u.hostname and (u.hostname == 'wnba.com' or u.hostname.endswith('.wnba.com'))
            and u.username is None and u.password is None and u.port is None and not u.query and not u.fragment,
            'SOURCE_IDENTITY')
    return value


def retain(raw, receipt):
    """Caller supplies original decoded bytes and actual acquisition receipts."""
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_BODY, 'BODY_LIMIT')
    value = dict(version=VERSION, body_base64=base64.b64encode(raw).decode(),
                 body_sha256=hashlib.sha256(raw).hexdigest(), body_size=len(raw), receipt=deepcopy(receipt))
    read(value)
    return value


def read(value):
    require(isinstance(value, dict) and set(value) == {'version', 'body_base64', 'body_sha256', 'body_size', 'receipt'}
            and value['version'] == VERSION, 'CUSTODY_MISSING')
    require(isinstance(value['body_base64'], str) and len(value['body_base64']) <= 4 * ((MAX_BODY + 2) // 3), 'BODY_LIMIT')
    try:
        raw = base64.b64decode(value['body_base64'], validate=True)
    except (ValueError, TypeError):
        raise ValueError('WNBA_SCENARIO_CUSTODY_CORRUPT') from None
    require(0 < len(raw) <= MAX_BODY and type(value['body_size']) is int and len(raw) == value['body_size']
            and hashlib.sha256(raw).hexdigest() == value['body_sha256'], 'CUSTODY_CORRUPT')
    r = value['receipt']
    require(isinstance(r, dict) and set(r) == set('source_id endpoint parameters format evidence_label request_started_at received_at observed_at observation_meaning publisher_availability status'.split())
            and r['evidence_label'] in {'SYNTHETIC', 'RETAINED'} and type(r['status']) is int and r['status'] == 200
            and r['observation_meaning'] == 'actual_local_first_observation_of_complete_decoded_body'
            and r['publisher_availability'] == 'UNKNOWN', 'RECEIPT_MISSING')
    require(clock(r['request_started_at']) <= clock(r['received_at']) <= clock(r['observed_at']), 'RECEIPT_CLOCK')
    if r['source_id'] == SOURCE:
        require(r['endpoint'] == ENDPOINT and r['format'] == 'json', 'SOURCE_IDENTITY')
        q = r['parameters']
        require(isinstance(q, dict) and set(q) == {'seasons', 'per_page', 'cursor'} and isinstance(q['seasons'], list)
                and len(q['seasons']) == 1 and type(q['seasons'][0]) is int
                and q['per_page'] == 100 and type(q['per_page']) is int
                and (q['cursor'] is None or type(q['cursor']) is int and q['cursor'] >= 0), 'SOURCE_SCOPE')
    else:
        require(r['source_id'] == 'official-wnba-event-document-v1' and r['format'] == 'utf8-text'
                and r['parameters'] == {}, 'SOURCE_IDENTITY')
        official_url(r['endpoint'])
    try:
        text = raw.decode('utf-8')
    except UnicodeError:
        raise ValueError('WNBA_SCENARIO_BODY_FORMAT') from None
    require('\x00' not in text, 'BODY_FORMAT')
    # Original content is never written to diagnostics. Credential echoes are
    # rejected, rather than modified and mislabelled byte-exact evidence.
    require(not re.search(r'(?i)(api[_-]?key|authorization|bearer\s|set-cookie|access_token)', text), 'CREDENTIAL_ECHO')
    if r['format'] == 'json':
        try:
            body = json.loads(text, object_pairs_hook=unique)
        except (ValueError, TypeError):
            raise ValueError('WNBA_SCENARIO_BODY_FORMAT') from None
        require(isinstance(body, dict) and set(body) == {'data', 'meta'} and isinstance(body['data'], list)
                and 0 < len(body['data']) <= 100 and isinstance(body['meta'], dict)
                and set(body['meta']) <= {'next_cursor', 'per_page'} and body['meta'].get('per_page') == 100
                and (body['meta'].get('next_cursor') is None or type(body['meta']['next_cursor']) is int
                     and body['meta']['next_cursor'] >= 0), 'BODY_FORMAT')
        return raw, body
    return raw, text
