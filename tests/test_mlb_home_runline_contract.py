"""Synthetic source replay only: no collector, SQLite store, fit or inference."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json

import pytest

from app_core import mlb_home_runline_contract as contract
from app_core import mlb_pregame_receipts as receipts
from app_core import mlb_production_readiness as legacy
from app_core import mlb_spread_total_model as old
from scripts.benchmark_drive_history_loading import blocked_network
from test_mlb_pregame_receipts import NOW, START, schedule_game, feed, odds_game


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def packet_input(line=-1.5, declared=False):
    """Hand-build immutable synthetic sources; never capture or register them."""
    observations = {}

    def source(payload, name="mlb_statsapi"):
        observation = {"source": name, "endpoint": "synthetic://offline", "params": {},
                       "observed_at": NOW.isoformat(), "payload": payload}
        key = old.digest(observation)
        observations[key] = observation
        return key, observation

    game = schedule_game(100, START)
    game['gameNumber'] = 1
    schedule_key, _ = source({"dates": [{"games": [game]}]})
    raw = odds_game()
    market = raw['bookmakers'][0]['markets'][0]
    market['outcomes'][0]['point'] = line
    quote = {"market_type": "spread_home", "line": line, "decimal_odds": 2.1,
             "sportsbook": "Novig", "provider_namespace": "odds_api", "provider_event_id": "odds-100",
             "observed_at": NOW.isoformat(), "provider_updated_at": (NOW-timedelta(minutes=1)).isoformat(),
             "source_game_start_utc": START.isoformat()}
    if declared:
        market['period'] = 'FULL_GAME'
        document = "Synthetic rules only: ordinary full-game integer-score half-line settlement."
        key, _ = source({"sportsbook": "Novig", "market_period": "FULL_GAME", "rule_version": "SYNTHETIC-v1",
                         "document_utf8": document, "document_sha256": hashlib.sha256(document.encode()).hexdigest(),
                         "settlement_rules": "SYNTHETIC_FULL_GAME_VALID_FINAL_ONLY"}, 'novig_rules_document')
        quote.update(market_period='FULL_GAME', settlement_rules_hash=key)
    quotes_key, _ = source(raw, 'odds_api')
    prior = []
    for i in range(1, 11):
        _, observation = source(feed(schedule_game(i, START-timedelta(days=i), True)))
        prior.append(receipts.prior_from_observation(observation))
    refs = {'schedule': schedule_key, 'quotes': quotes_key}
    if declared:
        refs['settlement_rules'] = quote['settlement_rules_hash']
    payload = {"schema_version": old.SCHEMA, "provider_namespace": "mlb", "provider_event_id": "100",
               "home_team_id": "mlb:112", "away_team_id": "mlb:134", "season": 2026,
               "game_start_utc": START.isoformat(), "captured_at": NOW.isoformat(),
               "prediction_cutoff": NOW.isoformat(), "quote": quote, "prior_games": prior,
               "source_observations": refs}
    return {'payload': payload, 'sha256': old.digest(payload)}, observations


def reseal(snapshot):
    snapshot['sha256'] = old.digest(snapshot['payload'])


def rebind_source(snapshot, observations, kind, change):
    source = deepcopy(observations[snapshot['payload']['source_observations'][kind]])
    change(source['payload'])
    key = old.digest(source)
    observations[key] = source
    snapshot['payload']['source_observations'][kind] = key
    reseal(snapshot)


MAPPING_ATTACKS = ('unrelated_matchup', 'swapped_teams', 'shared_city_team', 'bare_shared_city', 'schedule_names',
                   'doubleheader', 'final_doubleheader', 'provider_claim')


def mapping_fixture(attack):
    snapshot, observations = packet_input()
    if attack in ('unrelated_matchup', 'swapped_teams', 'shared_city_team', 'bare_shared_city', 'provider_claim'):
        def change(raw):
            if attack == 'provider_claim':
                raw['provider_ids'] = {'mlb': '999'}
                return
            pair = ('Seattle Mariners', 'Los Angeles Angels') if attack == 'unrelated_matchup' else (
                raw['away_team'], raw['home_team'])
            if attack == 'shared_city_team': pair = ('Chicago White Sox', raw['away_team'])
            if attack == 'bare_shared_city': pair = ('Chicago', raw['away_team'])
            raw['home_team'], raw['away_team'] = pair
            for entry, name in zip(raw['bookmakers'][0]['markets'][0]['outcomes'], pair):
                entry['name'] = name
        rebind_source(snapshot, observations, 'quotes', change)
    else:
        def change(raw):
            game = raw['dates'][0]['games'][0]
            if attack == 'schedule_names':
                game['teams']['home']['team']['name'] = 'Seattle Mariners'
                return
            companion = schedule_game(101, START + timedelta(hours=6))
            companion['gameNumber'] = 2
            if attack == 'final_doubleheader':
                companion = schedule_game(101, START - timedelta(hours=10), final=True)
                companion['gameNumber'] = 1
                game['gameNumber'] = 2
            raw['dates'][0]['games'].append(companion)
        rebind_source(snapshot, observations, 'schedule', change)
    return snapshot, observations


@pytest.mark.parametrize('attack', MAPPING_ATTACKS)
def test_ordered_odds_schedule_mapping_rejects_consistently_rehashed_sources(attack):
    snapshot, observations = mapping_fixture(attack)
    payload = snapshot['payload']
    assert old.digest(payload) == snapshot['sha256']
    assert all(old.digest(v) == k for k, v in observations.items())
    # Independent source checks accept these intact chains; replay their relationship.
    assert legacy._quote_source_check(payload, observations) is None
    assert legacy._schedule_check(payload, observations)[0] is None
    reason = 'DOUBLEHEADER_IDENTITY_AMBIGUOUS' if 'doubleheader' in attack else (
        'PROVIDER_ID_CONFLICT' if attack == 'provider_claim' else 'ODDS_SCHEDULE_MATCHUP_MISMATCH')
    with pytest.raises(ValueError, match=reason):
        contract.replay_features(snapshot, observations)


def test_original_ordered_pair_aliases_replay_without_live_capture_clock():
    snapshot, observations = packet_input()
    def change(raw):
        game = raw['dates'][0]['games'][0]
        game['teams']['home']['team']['name'] = 'Chi. Cubs'
        game['teams']['away']['team']['name'] = 'Pittsburgh'
    rebind_source(snapshot, observations, 'schedule', change)
    before = deepcopy((snapshot, observations))
    assert contract.replay_features(snapshot, observations)['payload']['identity']['game_number'] == 1
    assert (snapshot, observations) == before


def clock_fixture(case):
    snapshot, observations = packet_input()
    selected = (NOW - timedelta(minutes=2)).isoformat()
    def change(raw):
        book = raw['bookmakers'][0]
        market = book['markets'][0]
        if case != 'book_fallback':
            market['last_update'] = selected
        if case in ('market_without_book', 'missing_both', 'wrong_market_only'):
            book.pop('last_update')
        if case in ('missing_both', 'wrong_market_only'):
            market.pop('last_update')
        if case == 'wrong_market_only':
            book['markets'][1]['last_update'] = selected
        if case == 'unmatched_clocks':
            book['last_update'] = (NOW + timedelta(minutes=1)).isoformat()
            book['markets'][1]['last_update'] = (NOW + timedelta(minutes=2)).isoformat()
            other = deepcopy(market)
            other['outcomes'][0]['point'] = -2.5
            other['last_update'] = (NOW + timedelta(minutes=3)).isoformat()
            book['markets'].append(other)
        if case == 'invalid_market':
            market['last_update'] = 'not-a-clock'
        if case == 'duplicate_exact_different_clock':
            book['last_update'] = selected
            other = deepcopy(market)
            other['last_update'] = (NOW - timedelta(minutes=3)).isoformat()
            book['markets'].append(other)
    rebind_source(snapshot, observations, 'quotes', change)
    if case != 'book_fallback':
        snapshot['payload']['quote']['provider_updated_at'] = selected
    if case == 'book_instead_of_market':
        snapshot['payload']['quote']['provider_updated_at'] = (NOW - timedelta(minutes=1)).isoformat()
    reseal(snapshot)
    return snapshot, observations


@pytest.mark.parametrize('case', ['market_over_book', 'market_without_book', 'book_fallback', 'unmatched_clocks'])
def test_exact_quote_market_update_precedes_book_fallback(case):
    snapshot, observations = clock_fixture(case)
    before = deepcopy((snapshot, observations))
    packet = contract.replay_features(snapshot, observations)
    assert packet['payload']['identity']['quote']['provider_updated_at'] == snapshot['payload']['quote']['provider_updated_at']
    assert contract.verify_packet(packet, snapshot, observations) == packet
    assert (snapshot, observations) == before


@pytest.mark.parametrize('case', ['missing_both', 'wrong_market_only', 'book_instead_of_market',
                                'invalid_market', 'duplicate_exact_different_clock'])
def test_missing_invalid_or_conflicting_exact_quote_update_fails_closed(case):
    snapshot, observations = clock_fixture(case)
    assert all(old.digest(v) == k for k, v in observations.items())
    with pytest.raises(ValueError):
        contract.replay_features(snapshot, observations)


@pytest.mark.parametrize('line', [-1.5, 1.5])
def test_actual_legacy_replay_names_signed_orientation_and_round_trip(line):
    snapshot, observations = packet_input(line)
    before = deepcopy((snapshot, observations))
    packet = contract.replay_features(snapshot, observations)
    values = packet['payload']
    assert values['ordered_names'] == list(old.FEATURE_COLUMNS[:6]) + ['exact_line', 'price_implied_probability']
    assert values['named_values'] == {'home_ppg': 5., 'away_ppg': 2., 'home_oppg': 2., 'away_oppg': 5.,
                                      'home_win_pct': 1., 'away_win_pct': 0., 'exact_line': line,
                                      'price_implied_probability': 1 / 2.1}
    assert values['ordered_values'] == legacy.exact_feature_values(snapshot)[1]
    # Demonstrate the old report-name/index defect without rewriting history.
    old_report_names = [f['name'] for f in legacy.feature_contract()['scopes']['MLB/RUN_LINE']['features']]
    mislabeled = dict(zip(old_report_names, values['ordered_values']))
    assert mislabeled['home_win_pct'] == 2. and values['named_values']['home_win_pct'] == 1.
    assert old.FEATURE_COLUMNS == list(contract.LEGACY_NAMES)
    assert legacy.FEATURE_VERSION == 'mlb-receipt-asof-exact-market-v1'
    assert values['target_binding']['status'] == 'UNKNOWN'
    assert values['inference_status'] == 'NOT_EXECUTED'
    assert values['production_eligible'] is values['wager_approved'] is False
    assert values['recommended_stake'] == 0
    assert contract.verify_packet(json.loads(json.dumps(packet)), snapshot, observations) == packet
    assert (snapshot, observations) == before
    with pytest.raises(ValueError, match='TARGET_BINDING_UNKNOWN'):
        contract.verify_packet(packet, snapshot, observations, require_target_binding=True)


def test_captured_target_and_rules_replay_still_grants_no_authority():
    snapshot, observations = packet_input(declared=True)
    packet = contract.replay_features(snapshot, observations)
    checked = contract.verify_packet(packet, snapshot, observations, require_target_binding=True)
    assert checked['payload']['target_binding']['status'] == 'CAPTURED_TARGET_BOUND'
    assert checked['payload']['target_binding']['source_rights_approved'] is False
    assert checked['payload']['target_binding']['accepted_production_reader'] is False
    assert checked['payload']['inference_status'] == 'NOT_EXECUTED'


@pytest.mark.parametrize('change', ['names', 'vector', 'named_values', 'version', 'schema_hash', 'identity', 'stake'])
def test_consumer_rejects_resealed_vector_or_identity_reinterpretation(change):
    snapshot, observations = packet_input()
    packet = contract.replay_features(snapshot, observations)
    p = packet['payload']
    if change == 'names': p['ordered_names'][1], p['ordered_names'][2] = p['ordered_names'][2], p['ordered_names'][1]
    if change == 'vector': p['ordered_values'][2] = 99.
    if change == 'named_values': p['named_values']['home_win_pct'] = .7
    if change == 'version': p['feature_version'] = legacy.FEATURE_VERSION
    if change == 'schema_hash': p['schema_hash'] = '0' * 64
    if change == 'identity': p['identity']['home_team_id'] = 'mlb:999'
    if change == 'stake': p['recommended_stake'] = 1
    packet['sha256'] = old.digest(p)
    with pytest.raises(ValueError, match='FEATURE_PACKET_CONTRACT_MISMATCH'):
        contract.verify_packet(packet, snapshot, observations)


@pytest.mark.parametrize('field,value', [('market_type','spread_away'), ('market_type','moneyline_home'),
    ('line', -2.5), ('line', 1), ('sportsbook', 'FanDuel'), ('provider_namespace', 'espn')])
def test_unsupported_side_target_line_book_or_provider_fails(field, value):
    snapshot, observations = packet_input()
    snapshot['payload']['quote'][field] = value
    reseal(snapshot)
    with pytest.raises(ValueError):
        contract.replay_features(snapshot, observations)


@pytest.mark.parametrize('attack', ['receipt_hash', 'observation_hash', 'missing_source', 'source_identity',
                                  'schedule_team', 'missing_number', 'late_feature', 'post_start',
                                  'wrong_price', 'duplicate_quote'])
def test_source_corruption_incomplete_identity_and_chronology_rejected(attack):
    snapshot, observations = packet_input()
    p = snapshot['payload']
    if attack == 'receipt_hash': snapshot['sha256'] = '0' * 64
    elif attack == 'observation_hash': observations[p['source_observations']['quotes']]['payload']['id'] = 'other'
    elif attack == 'missing_source': observations.pop(p['prior_games'][0]['observation_hash'])
    elif attack in ('source_identity', 'duplicate_quote'):
        src = deepcopy(observations[p['source_observations']['quotes']])
        if attack == 'source_identity': src['payload']['id'] = 'other'
        else: src['payload']['bookmakers'][0]['markets'][0]['outcomes'].append(deepcopy(src['payload']['bookmakers'][0]['markets'][0]['outcomes'][0]))
        key = old.digest(src); observations[key] = src; p['source_observations']['quotes'] = key; reseal(snapshot)
    elif attack in ('schedule_team', 'missing_number'):
        src = deepcopy(observations[p['source_observations']['schedule']])
        game = src['payload']['dates'][0]['games'][0]
        if attack == 'schedule_team': game['teams']['home']['team']['id'] = 999
        else: game.pop('gameNumber')
        key = old.digest(src); observations[key] = src; p['source_observations']['schedule'] = key; reseal(snapshot)
    elif attack == 'late_feature': p['prior_games'][0]['available_at'] = (NOW+timedelta(seconds=1)).isoformat(); reseal(snapshot)
    elif attack == 'post_start': p['quote']['observed_at'] = START.isoformat(); reseal(snapshot)
    elif attack == 'wrong_price': p['quote']['decimal_odds'] = 3.; reseal(snapshot)
    with pytest.raises((ValueError, KeyError)):
        contract.replay_features(snapshot, observations)


@pytest.mark.parametrize('attack', ['unsupported_period', 'period_conflict', 'missing_rules', 'rule_hash', 'future_rules', 'document_hash'])
def test_period_and_settlement_facts_never_default_or_authorize(attack):
    snapshot, observations = packet_input(declared=True)
    p = snapshot['payload']
    if attack == 'unsupported_period': p['quote']['market_period'] = 'FIRST_FIVE'
    if attack == 'missing_rules': observations.pop(p['quote']['settlement_rules_hash'])
    if attack == 'rule_hash': observations[p['quote']['settlement_rules_hash']]['payload']['rule_version'] = 'changed'
    if attack == 'document_hash':
        src = deepcopy(observations[p['quote']['settlement_rules_hash']]); src['payload']['document_utf8'] = 'changed'
        key = old.digest(src); observations[key] = src
        p['quote']['settlement_rules_hash'] = key; p['source_observations']['settlement_rules'] = key
    if attack == 'future_rules':
        src = deepcopy(observations[p['quote']['settlement_rules_hash']]); src['observed_at'] = (NOW+timedelta(seconds=1)).isoformat()
        key = old.digest(src); observations[key] = src
        p['quote']['settlement_rules_hash'] = key; p['source_observations']['settlement_rules'] = key
    if attack == 'period_conflict':
        src = deepcopy(observations[p['source_observations']['quotes']]); src['payload']['bookmakers'][0]['markets'][0]['period'] = 'FIRST_FIVE'
        key = old.digest(src); observations[key] = src; p['source_observations']['quotes'] = key
    reseal(snapshot)
    with pytest.raises(ValueError):
        contract.replay_features(snapshot, observations)


def test_exact_ten_not_silent_reweighting_or_legacy_reordering():
    snapshot, observations = packet_input()
    game = schedule_game(50, START-timedelta(days=11), True)
    observation = {'source':'mlb_statsapi','endpoint':'synthetic://offline','params':{},'observed_at':NOW.isoformat(),'payload':feed(game)}
    key = old.digest(observation); observations[key] = observation
    snapshot['payload']['prior_games'].append(receipts.prior_from_observation(observation)); reseal(snapshot)
    assert len(legacy.exact_feature_values(snapshot)[1]) == 8  # historical interpretation remains readable
    with pytest.raises(ValueError, match='EXACT_TEN_PRIOR_GAMES_REQUIRED'):
        contract.replay_features(snapshot, observations)


@pytest.mark.parametrize('field,value', [('provider_updated_at',(NOW+timedelta(seconds=1)).isoformat()),
    ('provider_updated_at',NOW.isoformat()), ('observed_at',NOW.replace(tzinfo=None).isoformat())])
def test_original_update_and_timezone_clocks_are_not_reconstructed(field, value):
    snapshot, observations = packet_input()
    snapshot['payload']['quote'][field] = value; reseal(snapshot)
    with pytest.raises(ValueError): contract.replay_features(snapshot, observations)


@pytest.mark.parametrize('vector', [{'COVER':.4,'PUSH':.1,'NO_COVER':.5},
    {'COVER':True,'PUSH':0,'NO_COVER':0}, {'WIN':.6,'PUSH':0,'LOSS':.4},
    {'COVER':float('nan'),'PUSH':0,'NO_COVER':.4}])
def test_invalid_or_wrong_target_probability_vector(vector):
    with pytest.raises(ValueError): contract.verify_probability_vector(vector)


def test_valid_vector_can_have_negative_ev_without_inference_or_wager_assertion():
    snapshot, observations = packet_input()
    packet = contract.replay_features(snapshot, observations)
    vector = contract.verify_probability_vector({'COVER':.4,'PUSH':0.,'NO_COVER':.6})
    assert vector['COVER'] * (2.1 - 1) - vector['NO_COVER'] < 0
    assert packet['payload']['inference_status'] == 'NOT_EXECUTED'
    assert packet['payload']['recommended_stake'] == 0
