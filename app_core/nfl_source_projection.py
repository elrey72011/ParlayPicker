"""Attributable projections of original formats; never replacement source bodies."""
from copy import deepcopy
from app_core import nfl_observed_custody as custody, producer_provenance as producer

VERSION = 'nfl-original-source-projection-v1'
require = custody.require
digest = custody.digest


def seal(value):
    return dict(payload=deepcopy(value), sha256=digest(value))


def proof(objects, fact, original):
    require(isinstance(fact, dict) and set(fact) == {'object', 'path', 'text', 'operation'}, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    require(fact['object'] in objects, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    if fact['operation'] == 'receipt_source_namespace':
        require(fact['object'] == 'odds' and fact['path'] is None and fact['text'] is None
            and original['odds']['receipt']['source_id'] == 'the-odds-api:v4:americanfootball_nfl', 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
        return 'odds_api'  # Receipt attribution, never an asserted provider body column.
    value = objects[fact['object']]
    if fact['text'] is not None:
        require(isinstance(value, str) and fact['path'] is None and set(fact['text']) == {'start', 'end', 'exact'}, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
        t = fact['text']
        require(type(t['start']) is int and type(t['end']) is int and 0 <= t['start'] < t['end'] <= len(value)
            and value[t['start']:t['end']] == t['exact'], 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
        value = t['exact']
    else:
        require(isinstance(fact['path'], list) and len(fact['path']) <= 16, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
        for key in fact['path']:
            require(type(key) in {int, str} and (not isinstance(value, list) or type(key) is int and 0 <= key < len(value)), 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
            value = value[key]
    op = fact['operation']
    if op == 'team': value = producer.team(value, 'NFL')
    elif op == 'clock':
        value = producer.clock(value)
        require(value is not None, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    elif op == 'integer':
        require(str(value).isdigit(), 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT'); value = int(value)
    elif op == 'boolean':
        require(value in {'true', 'false'}, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT'); value = value == 'true'
    else: require(op == 'identity', 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    return value


def document_projection(p, objects, name):
    sealed = p['projections'][name]
    require(isinstance(sealed, dict) and set(sealed) == {'payload', 'sha256'} and digest(sealed['payload']) == sealed['sha256'], 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    v = sealed['payload']
    require(set(v) == {'version', 'facts', 'values', 'original_objects'} and v['version'] == VERSION, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    expected = {k:dict(body_sha256=obj['body_sha256'], receipt_sha256=digest(obj['receipt'])) for k,obj in p['objects'].items()}
    require(v['original_objects'] == expected, 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    values = {k:proof(objects, f, p['objects']) for k,f in v['facts'].items()}
    if name == 'clock_document':
        # These are explicitly local UNKNOWNs, not claims extracted from a provider.
        values.update(publisher_field=None, publisher_meaning='UNKNOWN')
    require(values == v['values'], 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    return values


def score_projection(p, objects):
    rows = objects['scores']
    require(isinstance(rows, list) and bool(rows), 'NFL_PRIVATE_RESPONSE_SCHEMA')
    # Retain complete original body, including other seasons and unplayed games.
    # Deterministic selected-season view preserves source order and all columns.
    require(all(isinstance(r, dict) and str(r.get('season', '')).isdigit() for r in rows), 'NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT')
    require(all(not r.get('source_available_at') for r in rows), 'NFL_PRIVATE_PUBLISHER_AVAILABILITY_CONFLICT')
    require(all(custody.known(r.get('game_id')) for r in rows) and len({r['game_id'] for r in rows}) == len(rows), 'NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT')
    selected = [i for i,r in enumerate(rows) if int(r['season']) == p['event']['season']]
    v = dict(version=VERSION, body_sha256=p['objects']['scores']['body_sha256'], receipt_sha256=digest(p['objects']['scores']['receipt']),
        season=p['event']['season'], selected_row_indices=selected, selection='exact_season_source_order_then_native_completed_before_utc_day',
        home=p['event']['home'], away=p['event']['away'], publisher_available_at=None, publisher_availability='UNKNOWN')
    require(p['projections']['scores'] == seal(v), 'NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT')
    require(selected, 'NFL_PRIVATE_HISTORY_MISSING')
    return [deepcopy(rows[i]) for i in selected]


def verify(p, objects):
    return {name:document_projection(p, objects, name) for name in ('schedule', 'listing', 'clock_document')} | {'scores':score_projection(p, objects)}
