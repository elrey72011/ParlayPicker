"""Explicit prospective OWNER_REVIEWED / PRIVATE_RESEARCH NFL inputs.

Separate from every independent acceptance catalog and legacy reader. Static
inspection never executes saved artifacts or reconstructs historical inference.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
import math
from pathlib import Path
import pandas as pd
from app_core import nfl_response_custody as custody, nfl_native_provenance as native
from app_core import nfl_inference_evidence as evidence, producer_provenance as producer
from app_core.research_estimate_trace import encode, fact

VERSION = 'nfl-owner-reviewed-private-inputs-v1'
RESULT = 'nfl-owner-reviewed-private-result-v1'
MAX_PACKET = 8 * 1024 * 1024
MAX_PACKETS = 4
MAX_SELECTED = 16 * 1024 * 1024
SELECTED = ContextVar('nfl_owner_private_packets', default=())
IN_NATIVE = ContextVar('nfl_owner_private_native_math', default=False)
ROOT = Path(__file__).resolve().parents[1]
REASONS = frozenset(('NFL_PRIVATE_' + name) for name in '''PACKET_SCHEMA PACKET_INTEGRITY CUSTODY_MISSING
CUSTODY_CORRUPT RESPONSE_SIZE RESPONSE_SCHEMA CREDENTIAL_FORBIDDEN CUSTODY_CLOCK_UNKNOWN CUSTODY_CLOCK_CONFLICT
DUPLICATE_RESPONSE EVENT_MAPPING_MISSING EVENT_MAPPING_CONFLICT EXACT_OFFER_NOT_SELECTED PACKET_AMBIGUOUS
OFFER_IDENTITY_CONFLICT PERIOD_RULES_LISTING_MISSING PERIOD_RULES_UNSUPPORTED PERMISSION_MISSING
PERMISSION_CLOCK_CONFLICT OWNER_REVIEW_MISSING OWNER_REVIEW_CONFLICT OWNER_REVIEW_CLOCK_CONFLICT
PUBLISHER_AVAILABILITY_MISSING DEPENDENCY_IDENTITY_CONFLICT DEPENDENCY_CLOCK_CONFLICT
HISTORY_MISSING FEATURE_ORDER_CONFLICT FEATURE_DERIVATION_CONFLICT FEATURE_CLOCK_CONFLICT
MODEL_BINDING_CONFLICT QUOTE_CLOCK_UNKNOWN QUOTE_CLOCK_CONFLICT INTEGER_PUSH_UNVALIDATED
INFERENCE_FAILED PROBABILITY_CONFLICT RUN_CONFLICT AUTHORITY_FORBIDDEN RUNTIME_CONFLICT'''.split())
require = custody.require
digest = custody.digest


def model_binding():
    return dict(predictor_id='score-distribution-v1:nfl', target='selected_side_full_game_half_point_spread_cover',
        configuration=evidence.configuration(), artifacts=evidence.artifacts(),
        predictor_callables=evidence.predictor_callables(), feature_order=list(evidence.FEATURES))


def reader_binding():
    return dict(runtime=evidence.runtime(), sources={p: __import__('hashlib').sha256((ROOT/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in ('app_core/nfl_owner_research.py', 'app_core/nfl_response_custody.py', 'app_core/nfl_native_provenance.py',
                  'app_core/market_probability_model.py', 'core/streamlit_pipeline.py')})


def load(raw, *, owner_upload=False):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_PACKET, 'NFL_PRIVATE_PACKET_SCHEMA')
    packet = json.loads(raw, object_pairs_hook=custody.unique)
    require(isinstance(packet, dict) and set(packet) == {'payload', 'sha256'}, 'NFL_PRIVATE_PACKET_SCHEMA')
    p = packet['payload']
    require(digest(p) == packet['sha256'], 'NFL_PRIVATE_PACKET_INTEGRITY')
    require(set(p) == set('version evidence_label event quote objects projections permissions owner_review model features feature_order feature_available_at original_inference_time'.split())
            and p['version'] == VERSION and p['evidence_label'] in {'RETAINED', 'SYNTHETIC'}
            and p['original_inference_time'] is None, 'NFL_PRIVATE_PACKET_SCHEMA')
    require(not owner_upload or p['evidence_label'] == 'RETAINED', 'NFL_PRIVATE_PACKET_SCHEMA')
    # Staging verifies bytes only. It creates no permission/review/acceptance.
    custody.objects(p['objects'])
    return packet


@contextmanager
def selected(packets=()):
    require(isinstance(packets, (tuple, list)) and len(packets) <= MAX_PACKETS, 'NFL_PRIVATE_PACKET_SCHEMA')
    packets = tuple(load(encode(p).encode()) for p in packets)
    require(sum(len(encode(p).encode()) for p in packets) <= MAX_SELECTED and
            len({p['sha256'] for p in packets}) == len(packets), 'NFL_PRIVATE_PACKET_AMBIGUOUS')
    token = SELECTED.set(deepcopy(packets))
    try:
        yield
    finally:
        SELECTED.reset(token)


def selection_requested():
    return bool(SELECTED.get()) and not IN_NATIVE.get()


def private(row):
    try:
        item = json.loads(row.get('ml_estimate_metadata', ''))
        return 'nfl_private_inputs' in item
    except (ValueError, TypeError, AttributeError):
        return False


def subject(p):
    return digest({k:v for k,v in p.items() if k != 'owner_review'})


def _path(value, path):
    require(isinstance(path, list), 'NFL_PRIVATE_PACKET_SCHEMA')
    for part in path:
        value = value[part]
    return value


def _clock(value, code):
    result = producer.clock(value)
    require(result is not None, code)
    return pd.Timestamp(result)


def _event(event):
    from core.nfl_teams import NFL_TEAMS
    require(isinstance(event, dict) and set(event) == set('canonical_event_id provider_namespace provider_event_id home away start season neutral_site'.split())
            and all(custody.known(event.get(k)) for k in ('canonical_event_id', 'provider_namespace', 'provider_event_id', 'home', 'away', 'start')),
            'NFL_PRIVATE_EVENT_MAPPING_MISSING')
    require(event['provider_namespace'] == 'odds_api' and
            producer.team(event['home'], 'NFL') == event['home'] and producer.team(event['away'], 'NFL') == event['away']
            and event['home'] != event['away'] and event['neutral_site'] is False
            and all(native.nfl_stats_identity(event[s]) in {v.upper() for v in NFL_TEAMS.values()} for s in ('home', 'away'))
            and type(event['season']) is int, 'NFL_PRIVATE_EVENT_MAPPING_CONFLICT')
    return event


def matches(p, source):
    q = p['quote']; e = p['event']
    # Separate exact quote identity from local schedule/candidate identifiers.
    return (producer.text(source.get('league') or source.get('League')).upper() == 'NFL'
        and source.get('market_type') == q['market_type'] and producer.number(source.get('spread_line')) == q['point']
        and producer.number(source.get('odds_american')) == q['price']
        and producer.team(source.get('home_team') or source.get('Home'), 'NFL') == e['home']
        and producer.team(source.get('away_team') or source.get('Away'), 'NFL') == e['away']
        and producer.clock(source.get('game_start_utc') or source.get('commence_time')) == producer.clock(e['start'])
        and any(all(m.get(k) == q.get(k) for k in ('book', 'market_type', 'point', 'price', 'recorded_at', 'provider_event_id', 'provider_namespace'))
                for m in producer._matches(source)))


def checked(packet, at, *, source=None):
    """All original facts, permissions and owner verification before mathematics."""
    p = load(encode(packet).encode())['payload']; e = _event(p['event']); q = p['quote']
    now = _clock(at, 'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT')
    start = _clock(e['start'], 'NFL_PRIVATE_EVENT_MAPPING_CONFLICT')
    require(q.get('market_type') in {'spread_home', 'spread_away'} and type(q.get('point')) in {float, int}
            and math.isfinite(q['point']) and abs(q['point'] % 1) == .5, 'NFL_PRIVATE_INTEGER_PUSH_UNVALIDATED')
    require(type(q.get('price')) in {float, int} and abs(q['price']) >= 100 and math.isfinite(q['price']), 'NFL_PRIVATE_OFFER_IDENTITY_CONFLICT')
    require(q.get('provider_quote_id') is None or custody.known(q['provider_quote_id']), 'NFL_PRIVATE_OFFER_IDENTITY_CONFLICT')
    quote = _clock(q.get('recorded_at'), 'NFL_PRIVATE_QUOTE_CLOCK_UNKNOWN')
    require(0 <= (now-quote).total_seconds() <= 900 and now < start, 'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT')
    objects = custody.objects(p['objects'])
    require(set(p['objects']) == {'schedule', 'odds', 'scores', 'listing', 'clock_document'}, 'NFL_PRIVATE_CUSTODY_MISSING')
    require(all(o['receipt']['evidence_label'] == p['evidence_label'] for o in p['objects'].values()), 'NFL_PRIVATE_PACKET_SCHEMA')
    proj = p['projections']
    require(set(proj) == {'schedule_path', 'odds_path', 'listing_path', 'clock_document_path'}, 'NFL_PRIVATE_PACKET_SCHEMA')
    scheduled = _path(objects['schedule'], proj['schedule_path'])
    require(scheduled == e, 'NFL_PRIVATE_EVENT_MAPPING_CONFLICT')
    game = _path(objects['odds'], proj['odds_path'])
    require(game.get('id') == e['provider_event_id'] and game.get('sport_key') == 'americanfootball_nfl'
            and producer.team(game.get('home_team'), 'NFL') == e['home'] and producer.team(game.get('away_team'), 'NFL') == e['away']
            and producer.clock(game.get('commence_time')) == producer.clock(e['start']), 'NFL_PRIVATE_EVENT_MAPPING_CONFLICT')
    supplied = []
    for book in game.get('bookmakers', []):
        for market in book.get('markets', []):
            for outcome in market.get('outcomes', []):
                if book.get('key') == q['book'] and market.get('key') == 'spreads' and outcome.get('name') == game['home_team' if q['market_type'] == 'spread_home' else 'away_team']:
                    supplied.append((book, market, outcome))
    require(len(supplied) == 1, 'NFL_PRIVATE_OFFER_IDENTITY_CONFLICT')
    b, m, o = supplied[0]
    require(o.get('point') == q['point'] and o.get('price') == q['price'] and m.get('last_update') == q['recorded_at']
            and q['provider_event_id'] == e['provider_event_id'] and q['provider_namespace'] == e['provider_namespace']
            and q.get('provider_quote_id') == o.get('quote_id'), 'NFL_PRIVATE_OFFER_IDENTITY_CONFLICT')
    listing = _path(objects['listing'], proj['listing_path'])
    require(isinstance(listing, dict) and all(custody.known(listing.get(k)) for k in ('listing_id', 'product', 'jurisdiction', 'rule_edition', 'source_reference')),
            'NFL_PRIVATE_PERIOD_RULES_LISTING_MISSING')
    require(listing.get('event') == e and listing.get('quote') == q and listing.get('period') == 'full_game'
            and listing.get('overtime') is True and listing.get('payoff') in {'binary_win_loss', 'novig_fvs_decided_game'},
            'NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED')
    from app_core.source_contract import RULES
    require(q['book'] == 'novig' and listing['payoff'] == 'novig_fvs_decided_game' and listing['rule_edition'] == RULES,
            'NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED')
    effective = [_clock(listing.get(k), 'NFL_PRIVATE_PERIOD_RULES_LISTING_MISSING') for k in ('effective_from', 'effective_until')]
    require(effective[0] <= quote <= now < effective[1], 'NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED')
    clocks = _path(objects['clock_document'], proj['clock_document_path'])
    require(clocks.get('provider_quote_field') == 'markets.last_update' and clocks.get('quote_meaning') == 'provider_market_last_update'
            and clocks.get('publisher_field') == 'source_available_at' and clocks.get('publisher_meaning') == 'publisher_available_at_per_record'
            and custody.known(clocks.get('source_reference')), 'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    observations = []
    for name, obj in p['objects'].items():
        r = obj['receipt']; observed = _clock(r['observed_at'], 'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
        require(observed <= now, 'NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT')
        observations.append(observed)
        permission = p['permissions'].get(name)
        require(isinstance(permission, dict) and set(permission) == set('reviewer reviewed_at effective_from effective_until source_id endpoint private_retention private_research_use credential_exclusion basis'.split())
                and permission['reviewer'] == 'Robert Velarde' and permission['private_retention'] is True and permission['private_research_use'] is True and permission['credential_exclusion'] is True
                and isinstance(permission['basis'], str) and len(permission['basis'].strip()) >= 16
                and permission['source_id'] == r['source_id'] and permission['endpoint'] == r['endpoint'], 'NFL_PRIVATE_PERMISSION_MISSING')
        reviewed, begin, end = [_clock(permission[k], 'NFL_PRIVATE_PERMISSION_CLOCK_CONFLICT') for k in ('reviewed_at', 'effective_from', 'effective_until')]
        request = _clock(r['request_started_at'], 'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
        require(reviewed <= request and begin <= request <= now < end, 'NFL_PRIVATE_PERMISSION_CLOCK_CONFLICT')
    require(set(p['permissions']) == set(p['objects']), 'NFL_PRIVATE_PERMISSION_MISSING')
    require(quote <= _clock(p['objects']['odds']['receipt']['observed_at'], 'NFL_PRIVATE_QUOTE_CLOCK_UNKNOWN') <= now,
            'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT')
    require(p['objects']['scores']['receipt']['format'] == 'csv' and
            p['objects']['scores']['receipt']['provider_clock_meaning'] == 'publisher_available_at_per_record', 'NFL_PRIVATE_PUBLISHER_AVAILABILITY_MISSING')
    scores = objects['scores']; identifiers = set(); typed = []
    from core.nfl_teams import NFL_TEAMS
    from app_core.research_replay import cell
    for row in scores:
        require(all(row.get(k) for k in ('game_id', 'season', 'gameday', 'home_team', 'away_team'))
                and all(k in row for k in ('home_score', 'away_score', 'result', 'source_available_at')), 'NFL_PRIVATE_PUBLISHER_AVAILABILITY_MISSING')
        require(custody.known(row['game_id']) and row['game_id'] not in identifiers, 'NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT')
        identifiers.add(row['game_id'])
        # Native gameday is a date, not a fabricated original kickoff/availability clock.
        day = pd.to_datetime(row['gameday'], errors='coerce', utc=True)
        require(pd.notna(day), 'NFL_PRIVATE_DEPENDENCY_CLOCK_CONFLICT')
        require(int(row['season']) == e['season'] and native.nfl_stats_identity(row['home_team'], schedule_code=True) in {v.upper() for v in NFL_TEAMS.values()}
                and native.nfl_stats_identity(row['away_team'], schedule_code=True) in {v.upper() for v in NFL_TEAMS.values()}
                and native.nfl_stats_identity(row['home_team'], schedule_code=True) != native.nfl_stats_identity(row['away_team'], schedule_code=True), 'NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT')
        converted = dict(row, season=int(row['season']), **{k:float(row[k]) if row[k] else None for k in ('home_score', 'away_score', 'result')})
        values = [converted[k] for k in ('home_score', 'away_score', 'result')]
        require(all(v is None or math.isfinite(v) for v in values), 'NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT')
        if any(v is not None for v in values):
            require(all(v is not None for v in values) and converted['home_score'] >= 0 and converted['away_score'] >= 0
                    and converted['result'] == converted['home_score']-converted['away_score']
                    and row['game_id'] != e['provider_event_id'], 'NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT')
            av = _clock(row['source_available_at'], 'NFL_PRIVATE_PUBLISHER_AVAILABILITY_MISSING')
            ob = _clock(p['objects']['scores']['receipt']['observed_at'], 'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
            require(day.date() <= av.date() and av <= ob <= now, 'NFL_PRIVATE_DEPENDENCY_CLOCK_CONFLICT')
        # Retain unscored schedule rows in the full body; unchanged native
        # selection ignores their explicit missing results. Never fabricate clocks.
        typed.append({k:cell(v) for k,v in converted.items()})
    original = dict(columns=list(scores[0]), rows=typed)
    # Native unchanged strict completed-before-current-UTC-day and stable last5 mathematics.
    stats = {}
    for side in ('home', 'away'):
        rows = native.selected_rows(original, e['season'], now.isoformat(), native.nfl_stats_identity(e[side]))
        require(bool(rows), 'NFL_PRIVATE_HISTORY_MISSING')
        stats[side] = native.aggregates(rows)
    features = {}
    for name in evidence.FEATURES:
        if name == 'feature_diff_last5':
            features[name] = stats['home']['last5_win_pct']-stats['away']['last5_win_pct']
        else:
            _, side, key = name.split('_', 2)
            features[name] = stats[side][native.MAPPING[key]]
    require(p['feature_order'] == list(evidence.FEATURES) and list(p['features']) == list(sorted(evidence.FEATURES)), 'NFL_PRIVATE_FEATURE_ORDER_CONFLICT')
    require(set(features) == set(p['features']) and all(fact(features[k]) == fact(p['features'][k]) for k in features), 'NFL_PRIVATE_FEATURE_DERIVATION_CONFLICT')
    available = _clock(p['feature_available_at'], 'NFL_PRIVATE_FEATURE_CLOCK_CONFLICT')
    require(max(observations) <= available <= now, 'NFL_PRIVATE_FEATURE_CLOCK_CONFLICT')
    require(p['model'] == model_binding(), 'NFL_PRIVATE_MODEL_BINDING_CONFLICT')
    review = p['owner_review']
    require(isinstance(review, dict) and set(review) == set('version reviewer reviewed_at subject_sha256 status purpose attestation findings'.split())
            and review['version'] == 'nfl-owner-verification-v1' and review['reviewer'] == 'Robert Velarde'
            and review['status'] == 'OWNER_REVIEWED' and review['purpose'] == 'PRIVATE_RESEARCH'
            and review['attestation'] == 'I verified the original evidence and applicable facts for this exact subject.', 'NFL_PRIVATE_OWNER_REVIEW_MISSING')
    require(review['subject_sha256'] == subject(p) and set(review['findings']) == set('event_mapping offer_and_settlement permissions clocks_and_original_custody feature_provenance model_binding'.split())
            and all(isinstance(v, dict) and set(v) == {'conclusion', 'basis'} and v['conclusion'] == 'VERIFIED'
                    and isinstance(v['basis'], str) and len(v['basis'].strip()) >= 16 for v in review['findings'].values()), 'NFL_PRIVATE_OWNER_REVIEW_CONFLICT')
    verification = _clock(review['reviewed_at'], 'NFL_PRIVATE_OWNER_REVIEW_CLOCK_CONFLICT')
    require(available <= verification < now, 'NFL_PRIVATE_OWNER_REVIEW_CLOCK_CONFLICT')
    if source is not None:
        require(matches(p, source), 'NFL_PRIVATE_OFFER_IDENTITY_CONFLICT')
        require(all(not producer.text(source.get(k)) or source[k] == e[field] for k, field in
                    (('provider_event_id', 'provider_event_id'), ('provider_namespace', 'provider_namespace'))),
                'NFL_PRIVATE_EVENT_MAPPING_CONFLICT')
        # Contradictory parallel named-side, period and price descriptors are not repaired.
        for fields, expected in ((('Home', 'home_team'), e['home']), (('Away', 'away_team'), e['away'])):
            require(all(not producer.text(source.get(k)) or producer.team(source[k], 'NFL') == expected for k in fields), 'NFL_PRIVATE_EVENT_MAPPING_CONFLICT')
        require(all(not producer.text(source.get(k)) or producer.clock(source[k]) == producer.clock(e['start'])
                    for k in ('game_start_utc', 'commence_time')), 'NFL_PRIVATE_EVENT_MAPPING_CONFLICT')
        require(all(not producer.text(source.get(k)) or source[k] == expected for k, expected in
                    (('market_period', listing['period']), ('period', listing['period']), ('settlement_rules', listing['rule_edition']))),
                'NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED')
    return features, listing


def predict(source):
    from app_core.research_estimate_trace import generated_time, origin_metadata
    at = generated_time()
    result = dict(ml_probability=float('nan'), ml_probability_source='', ml_target='', ml_projection=float('nan'),
        ml_residual_scale=float('nan'), ml_feature_quality='unavailable', ml_inference_status='unavailable', ml_unavailable_reason='NFL_PRIVATE_EXACT_OFFER_NOT_SELECTED')
    packet = None
    try:
        observed = producer._matches(source)
        matching = [p for p in SELECTED.get() if p['payload']['quote']['market_type'] == source.get('market_type')
                    and any(m.get('provider_event_id') == p['payload']['event']['provider_event_id']
                            and m.get('provider_namespace') == p['payload']['event']['provider_namespace'] for m in observed)]
        require(len(matching) == 1, 'NFL_PRIVATE_PACKET_AMBIGUOUS' if matching else 'NFL_PRIVATE_EXACT_OFFER_NOT_SELECTED')
        packet = matching[0]
        features, listing = checked(packet, at, source=source)
        native_row = dict(features, League='NFL', league='NFL', market_type=source['market_type'], spread_line=packet['payload']['quote']['point'],
                          ml_feature_eligible=True, stats_resolution_status='resolved')
        token = IN_NATIVE.set(True)
        try:
            from app_core.market_probability_model import predict_market_probabilities
            computed = predict_market_probabilities(pd.DataFrame([native_row])).iloc[0]
        finally:
            IN_NATIVE.reset(token)
        require(computed['ml_inference_status'] == 'success', 'NFL_PRIVATE_INFERENCE_FAILED')
        result.update({k:computed[k] for k in result})
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, ArithmeticError) as exc:
        result['ml_unavailable_reason'] = str(exc) if str(exc) in REASONS else 'NFL_PRIVATE_PACKET_SCHEMA'
    origin_source = dict(source)
    if result['ml_inference_status'] == 'success':
        q = packet['payload']['quote']; e = packet['payload']['event']
        # New presentation bindings cite the separate reviewed original listing;
        # they never amend the retained provider body or old producer packet.
        fields = dict(quote_id=q.get('provider_quote_id') or 'nfl-private-derived:' + digest(q),
            market_period=listing['period'], settlement_rules=listing['rule_edition'],
            game_start_utc=e['start'], provider_namespace=e['provider_namespace'], provider_event_id=e['provider_event_id'])
        result.update(fields)
        origin_source.update(fields)
    metadata = json.loads(origin_metadata(origin_source, result, producer._line(source), generated_at=at))
    p = dict(version=RESULT, status=result['ml_inference_status'], reason=result['ml_unavailable_reason'], inference_time=at,
        review_status='OWNER_REVIEWED' if result['ml_inference_status'] == 'success' else 'UNVERIFIED', purpose='PRIVATE_RESEARCH', source_acceptance=False, calibration=False,
        scientific_qualification=False, wagering_authority=False, wager_action='PASS', live_stake=0)
    if packet is not None:
        p['original_packet'] = deepcopy(packet)
    if result['ml_inference_status'] == 'success':
        p.update(raw_probability=fact(result['ml_probability']), consumed_reader=reader_binding(),
                 original_blend=None, ui_refresh=[], consumed_features=features, payoff=listing['payoff'])
    metadata['nfl_private_inputs'] = dict(payload=p, sha256=digest(p))
    result['ml_estimate_metadata'] = encode(metadata)
    # The original current producer clock, never a reconstructed historical time.
    result['prediction_generated_at'] = at
    return result


def diagnose(source):
    """Static read-back only. No numeric inference or accepted-catalog mutation."""
    try:
        item = json.loads(source['ml_estimate_metadata']); saved = item['nfl_private_inputs']; p = saved['payload']
        require(digest(p) == saved['sha256'] and p['version'] == RESULT, 'NFL_PRIVATE_PACKET_INTEGRITY')
        require(p['source_acceptance'] is False and p['calibration'] is False and p['scientific_qualification'] is False
                and p['wagering_authority'] is False and p['wager_action'] == 'PASS' and p['live_stake'] == 0, 'NFL_PRIVATE_AUTHORITY_FORBIDDEN')
        require(p['purpose'] == 'PRIVATE_RESEARCH' and p['review_status'] == ('OWNER_REVIEWED' if p['status'] == 'success' else 'UNVERIFIED'), 'NFL_PRIVATE_OWNER_REVIEW_CONFLICT')
        if p['status'] != 'success':
            return dict(status='INCOMPLETE', reason=p['reason'] if p['reason'] in REASONS else 'NFL_PRIVATE_INFERENCE_FAILED')
        features, listing = checked(p['original_packet'], p['inference_time'], source=source)
        require(p['payoff'] == listing['payoff'], 'NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED')
        require(p['consumed_features'] == features and p['consumed_reader'] == reader_binding(), 'NFL_PRIVATE_RUNTIME_CONFLICT')
        require(p['raw_probability'] == item['probability'] == fact(source['ml_probability']) and p['inference_time'] == item['generated_at']
                == source.get('prediction_generated_at') and source['ml_probability_source'] == 'score-distribution-v1:nfl', 'NFL_PRIVATE_PROBABILITY_CONFLICT')
        require(isinstance(p['ui_refresh'], list), 'NFL_PRIVATE_PACKET_SCHEMA')
        latest = p['original_blend']
        previous_clock = _clock(p['inference_time'], 'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT')
        for receipt in p['ui_refresh']:
            r = receipt['payload']
            require(digest(r) == receipt['sha256'] and r['version'] == 'nfl-owner-private-ui-refresh-v1'
                    and r['original_packet_sha256'] == p['original_packet']['sha256'], 'NFL_PRIVATE_PACKET_INTEGRITY')
            require(latest is not None and r['original_raw_probability'] == p['raw_probability']
                    and r['previous_probability'] == latest['probability'] and r['consumed_blend'] == evidence.consumed_blend()
                    and r['semantics'] == 'recorded_ui_refresh_not_calibration', 'NFL_PRIVATE_PROBABILITY_CONFLICT')
            q = p['original_packet']['payload']['quote']; e = p['original_packet']['payload']['event']
            refresh = _clock(r['refreshed_at'], 'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT')
            require(previous_clock <= refresh and 0 <= (refresh-_clock(q['recorded_at'], 'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT')).total_seconds() <= 900
                    and refresh < _clock(e['start'], 'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'), 'NFL_PRIVATE_QUOTE_CLOCK_CONFLICT')
            previous_clock = refresh
            latest = r
        if latest is not None:
            require(latest['probability'] == fact(source.get('calibrated_probability'))
                    and latest['estimated_ev'] == fact(source.get('expected_value')), 'NFL_PRIVATE_PROBABILITY_CONFLICT')
        return dict(status='COMPLETE', reason='AVAILABLE', probability=p['raw_probability']['value'], review_status='OWNER_REVIEWED', purpose='PRIVATE_RESEARCH')
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError) as exc:
        return dict(status='REJECTED', reason=str(exc) if str(exc) in REASONS else 'NFL_PRIVATE_PACKET_SCHEMA')


def finish(frame):
    for i, row in frame.iterrows():
        if not private(row):
            continue
        item = json.loads(row['ml_estimate_metadata']); p = item['nfl_private_inputs']['payload']
        if p['status'] == 'success' and p['original_blend'] is None:
            p['original_blend'] = dict(probability=fact(row.get('calibrated_probability')), estimated_ev=fact(row.get('expected_value')),
                semantics='original_market_context_blend_not_calibration')
        item['nfl_private_inputs']['sha256'] = digest(p)
        frame.at[i, 'ml_estimate_metadata'] = encode(item)
    return frame


def private_display(source):
    assessment = diagnose(source)
    result = dict(version='nfl-owner-private-display-v1', status=assessment['status'], reason=assessment['reason'],
        review_status='OWNER_REVIEWED' if assessment['status'] == 'COMPLETE' else 'UNVERIFIED', purpose='PRIVATE_RESEARCH', raw_probability=None, original_blend=None, ui_refresh=None,
        probability_semantics='decided_game_cover_probability', ev=None, edge=None, stake=0, wager_action='PASS', calibrated=False)
    if assessment['status'] == 'COMPLETE':
        p = json.loads(source['ml_estimate_metadata'])['nfl_private_inputs']['payload']
        result.update(raw_probability=assessment['probability'], original_blend=p['original_blend'], ui_refresh=p['ui_refresh'],
            original_packet_sha256=p['original_packet']['sha256'], inference_time=p['inference_time'],
            reviewer=p['original_packet']['payload']['owner_review']['reviewer'],
            reviewed_at=p['original_packet']['payload']['owner_review']['reviewed_at'])
    return result


def retain_ui_refresh(before, after, inputs):
    """Record the actual existing UI blend; never replace raw/original clocks."""
    at = evidence.ui_reblend_time()
    for i, row in after.iterrows():
        if not private(row):
            continue
        item = json.loads(row['ml_estimate_metadata']); p = item['nfl_private_inputs']['payload']
        if p['status'] != 'success':
            continue
        require(i in before.index and before.at[i, 'ml_estimate_metadata'] == row['ml_estimate_metadata'], 'NFL_PRIVATE_RUN_CONFLICT')
        r = dict(version='nfl-owner-private-ui-refresh-v1', refreshed_at=at, original_packet_sha256=p['original_packet']['sha256'],
            original_raw_probability=p['raw_probability'], previous_probability=fact(before.at[i, 'calibrated_probability']),
            inputs={k:fact(v.at[i]) for k,v in inputs.items()}, probability=fact(row.get('calibrated_probability')),
            estimated_ev=fact(row.get('expected_value')), semantics='recorded_ui_refresh_not_calibration',
            consumed_blend=evidence.consumed_blend())
        p['ui_refresh'].append(dict(payload=r, sha256=digest(r)))
        item['nfl_private_inputs']['sha256'] = digest(p)
        after.at[i, 'ml_estimate_metadata'] = encode(item)
