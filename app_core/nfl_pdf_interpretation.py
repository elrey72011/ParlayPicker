"""Explicit, bounded OWNER_REVIEWED interpretation of attributable NFL prose.

No network, acceptance catalog, free-form inference, or permission creation.
UTF-8 character locations refer to complete original custody bodies, not edited
website summaries or PDF extraction presented as an original response.
"""
from copy import deepcopy
import re
from app_core import nfl_source_projection as old, nfl_pdf_evidence as custody
from app_core import producer_provenance as producer

VERSION = 'nfl-pdf-source-interpretation-v3'
REVIEW = 'nfl-owner-pdf-source-interpretation-review-v3'
METHOD = 'nfl-bounded-pdf-prose-and-attribution-v1'
ADAPTER = 'nfl-novig-decided-half-point-full-game-ot-v1'
require, digest = custody.require, custody.digest
seal = old.seal
CONFLICT = 'NFL_PRIVATE_INTERPRETATION_CONFLICT'
UNSUPPORTED = 'NFL_PRIVATE_INTERPRETATION_UNSUPPORTED'
METHODS = {
    'schedule': dict(canonical_event_id='literal',provider_event_id='literal',provider_namespace='receipt_namespace',
        home='team',away='team',start='utc_instant',season='integer',neutral_site='neutral_site'),
    'listing': dict(listing_id='literal',product='literal',jurisdiction='literal',period='period',overtime='overtime',
        payoff='payoff',rule_edition='edition',amendments='amendments',effective_from='effective_from',effective_until='effective_until',
        source_reference='literal',reference_team='team',opponent='team',threshold='half_point',comparison='comparison',position='position',
        feed_kind='feed_kind',feed_identity='literal',applicability='applicability'),
    'clock_document': dict(provider_quote_field='quote_field',quote_meaning='quote_meaning',source_reference='literal'),
    'rules': dict(rulebook='literal',ticker='literal',rule_edition='edition',amendments='amendments',period='period',
        overtime='overtime',payoff='payoff',reference_definition='reference_definition',
        effective_from='edition_effective_from',effective_until='edition_effective_until')}


def clock(value):
    require(isinstance(value, str) and re.search(r'(Z|[+-]\d\d:\d\d)$', value), 'NFL_PRIVATE_INTERPRETATION_CLOCK_CONFLICT')
    result = producer.clock(value)
    require(result is not None, 'NFL_PRIVATE_INTERPRETATION_CLOCK_CONFLICT')
    return result


def subject(value):
    return digest({k:v for k,v in value.items() if k != 'review'})


def normalize(method, excerpt):
    """Small reviewed vocabulary; unsupported prose stays unavailable."""
    if method=='receipt_namespace':
        require(excerpt=='odds_api',CONFLICT);return excerpt
    if method=='integer' and type(excerpt) is int:
        require(0<=excerpt<=9999,UNSUPPORTED);return excerpt
    if method=='neutral_site' and type(excerpt) is bool:return excerpt
    require(isinstance(excerpt, str) and 0 < len(excerpt) <= 2048, UNSUPPORTED)
    text = ' '.join(excerpt.split())
    if method == 'quote_field':
        # Inert exact source spans may include the guide's inline code markup.
        text = re.sub(r'</?code>', '', text)
    lower = text.lower().rstrip('.')
    if method == 'literal':
        return text
    if method == 'team':
        result = producer.team(text, 'NFL')
        require(bool(result), UNSUPPORTED)
        return result
    if method == 'utc_instant':
        return clock(text)
    if method == 'integer':
        require(re.fullmatch(r'\d{1,4}', text), UNSUPPORTED)
        return int(text)
    if method == 'half_point':
        require(re.fullmatch(r'[+-]?\d{1,3}\.5', text), UNSUPPORTED)
        return float(text)
    if method == 'american_price':
        require(re.fullmatch(r'[+-]\d{3,5}', text) and abs(int(text)) >= 100, UNSUPPORTED)
        return int(text)
    if method == 'reference_definition' and re.fullmatch(
            r"calculated as <team> ?'s points minus <opponent> ?'s points recorded within <time period>", lower):
        # Exact NFL Spread template variables, not named-team or period inference.
        # The listing still binds their identities and full-game/OT applicability.
        return 'reference_minus_opponent_final_points'
    vocabulary = {
        'period': {'entire game':'full_game', 'full game':'full_game', 'first half':'first_half'},
        'overtime': {'including overtime':True, 'inclusive of overtime':True, 'excluding overtime':False},
        'neutral_site': {'not a neutral-site game':False, 'neutral-site game':True},
        'quote_field': {'market-level last_update':'markets.last_update'},
        'quote_meaning': {
            'the last time our system saw odds for that market from the bookmaker':'provider_market_observation',
            "when our system last saw the bookmaker's market odds":'provider_market_observation',
            "when our system last saw the bookmaker's odds for this market":'provider_market_observation'},
        'payoff': {'fair value settlement':'novig_fvs_decided_game'},
        'reference_definition': {"reference team's final score minus opposing team's final score":'reference_minus_opponent_final_points'},
        'comparison': {'greater than':'greater_than', 'less than':'less_than'},
        'position': {'yes':'YES', 'no':'NO'},
        'feed_kind': {'outcome sid':'sid', 'outcome link':'link'},
        'amendments': {'no applicable amendments':[]},
    }
    if method in vocabulary:
        require(lower in vocabulary[method], UNSUPPORTED)
        return deepcopy(vocabulary[method][lower])
    if method == 'edition':
        match = re.fullmatch(r'Rule edition: (NFL-001 edition (\d{4}-\d{2}-\d{2}))\.?', text)
        require(match is not None, UNSUPPORTED)
        clock(match[2]+'T00:00:00Z')  # Valid date; NOT an effective-date inference.
        return match[1]
    if method in {'effective_from', 'effective_until','edition_effective_from','edition_effective_until'}:
        word = 'from' if method.endswith('_from') else 'until'
        scope = 'rule edition' if method.startswith('edition_') else 'listing'
        match = re.fullmatch(r'Effective for this '+scope+' '+word+r' (\S+)', text)
        require(match is not None, 'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
        return clock(match[1])
    if method == 'applicability':
        match = re.fullmatch(r'Listing (\S+) governs provider event (\S+) for (.+) versus (.+)\.', text)
        require(match is not None, 'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
        return dict(listing_id=match[1], provider_event_id=match[2], reference_team=producer.team(match[3],'NFL'), opponent=producer.team(match[4],'NFL'))
    raise ValueError(UNSUPPORTED)


def document(p, objects, name, at):
    wrapped = p['projections'][name]
    require(isinstance(wrapped, dict) and set(wrapped) == {'payload','sha256'} and digest(wrapped['payload']) == wrapped['sha256'], CONFLICT)
    v = wrapped['payload']
    require(set(v) == {'version','method_version','original_objects','facts','values','applicability','prepared_at','review'}
            and v['version'] == VERSION and v['method_version'] == METHOD, CONFLICT)
    identities = {k:dict(body_sha256=o['body_sha256'],receipt_sha256=digest(o['receipt'])) for k,o in p['objects'].items()}
    for k,o in p['objects'].items():
        if o.get('version')==custody.VERSION:identities[k]['extraction_sha256']=digest(o['extraction'])
    require(v['original_objects'] == identities and v['applicability'] == dict(event=p['event'],quote=p['quote']), CONFLICT)
    require(isinstance(v['facts'],dict) and 0 < len(v['facts']) <= 32 and set(v['values']) == set(v['facts']), CONFLICT)
    require(set(v['facts']) == set(METHODS[name]), 'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
    values = {}
    for field, f in v['facts'].items():
        require(set(f) == {'supports','method','status'} and f['status'] in {'DOCUMENTED','OWNER_INTERPRETED'}
                and isinstance(f['supports'],list) and 1 <= len(f['supports']) <= 4, CONFLICT)
        require(f['method'] == METHODS[name][field], UNSUPPORTED)
        require(all(s.get('operation') in {'identity','pdf_page_text','receipt_source_namespace'} for s in f['supports']), CONFLICT)
        require(all((s.get('operation')=='receipt_source_namespace') == (name=='schedule' and field=='provider_namespace') for s in f['supports']),CONFLICT)
        require((f['status']=='DOCUMENTED') == (f['method']=='literal'), CONFLICT)
        extracted = [custody.proof(objects, support, p['objects']) for support in f['supports']]
        interpreted = [normalize(f['method'], value) for value in extracted]
        require(all(value == interpreted[0] for value in interpreted), CONFLICT)
        values[field] = interpreted[0]
        # Listing-specific facts cannot be borrowed from a generic rule template.
        if name == 'listing':
            require(all(s['object'] == 'listing' for s in f['supports']), 'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
        if name == 'rules':
            require(all(s['object'] == 'rules' for s in f['supports']), CONFLICT)
    require(values == v['values'], CONFLICT)
    prepared, now = clock(v['prepared_at']), clock(at)
    require(all(o.get('version')!=custody.VERSION or clock(o['extraction']['prepared_at'])<=prepared for o in p['objects'].values()), 'NFL_PRIVATE_INTERPRETATION_CLOCK_CONFLICT')
    require(max(clock(o['receipt']['observed_at']) for o in p['objects'].values()) <= prepared <= now, 'NFL_PRIVATE_INTERPRETATION_CLOCK_CONFLICT')
    r = v['review']
    require(isinstance(r,dict) and set(r) == {'version','reviewer','reviewed_at','subject_sha256','status','purpose','findings','limitations'}
            and r['version'] == REVIEW and r['reviewer'] == 'Robert Velarde' and r['status'] == 'OWNER_REVIEWED'
            and r['purpose'] == 'PRIVATE_RESEARCH', 'NFL_PRIVATE_INTERPRETATION_REVIEW_MISSING')
    require(r['subject_sha256'] == subject(v) and set(r['findings']) == set(values)
            and all(set(f) == {'conclusion','basis'} and f['conclusion'] == 'VERIFIED'
                    and isinstance(f['basis'],str) and 24 <= len(f['basis'].strip()) <= 4096 for f in r['findings'].values())
            and isinstance(r['limitations'],str) and 24 <= len(r['limitations'].strip()) <= 4096,
            'NFL_PRIVATE_INTERPRETATION_REVIEW_CONFLICT')
    require(prepared <= clock(r['reviewed_at']) <= clock(p['owner_review'].get('reviewed_at')) < now,
            'NFL_PRIVATE_INTERPRETATION_CLOCK_CONFLICT')
    return values


def verify(p, objects, at):
    result = {name:document(p,objects,name,at) for name in ('schedule','listing','clock_document','rules')}
    result['clock_document'].update(publisher_field=None,publisher_meaning='UNKNOWN')
    result['scores'] = old.score_projection(p,objects)
    return result


def listing_contract(p, projections, market, outcome, at):
    """Exact external edition/content applicability, not an accepted-source catalog."""
    l, r = projections['listing'], projections['rules']
    fields = set('listing_id product jurisdiction period overtime payoff rule_edition amendments effective_from effective_until source_reference reference_team opponent threshold comparison position feed_kind feed_identity applicability'.split())
    require(set(l) == fields, 'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
    require(set(r) == set(METHODS['rules']), 'NFL_PRIVATE_RULE_EDITION_CONFLICT')
    require(r['rulebook'] == 'NFL-001' and r['ticker'] == 'NFLSPR'
            and r['rule_edition'] == l['rule_edition'] and re.fullmatch(r'NFL-001 edition \d{4}-\d{2}-\d{2}',r['rule_edition']), 'NFL_PRIVATE_RULE_EDITION_CONFLICT')
    # A nonempty/unknown amendment chain is deliberately unsupported, never omitted.
    require(r['amendments'] == l['amendments'] == [], 'NFL_PRIVATE_RULE_AMENDMENT_UNSUPPORTED')
    require(l['period'] == r['period'] == 'full_game' and l['overtime'] is r['overtime'] is True
            and l['payoff'] == r['payoff'] == 'novig_fvs_decided_game'
            and r['reference_definition'] == 'reference_minus_opponent_final_points', 'NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED')
    e, q = p['event'], p['quote']
    selected = e['home' if q['market_type'] == 'spread_home' else 'away']
    require({l['reference_team'],l['opponent']} == {e['home'],e['away']} and l['applicability'] == dict(
        listing_id=l['listing_id'],provider_event_id=e['provider_event_id'],reference_team=l['reference_team'],opponent=l['opponent']),
        'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
    reference_selected = l['reference_team'] == selected
    require(l['comparison'] == 'greater_than' and l['position'] == ('YES' if reference_selected else 'NO')
            and l['threshold'] == (-q['point'] if reference_selected else q['point']), 'NFL_PRIVATE_POSITION_MAPPING_CONFLICT')
    require(l['feed_kind'] in {'sid','link'} and custody.known(l['feed_identity'])
            and l['feed_identity'] == outcome.get(l['feed_kind']), 'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
    for key in ('listing_id','product','jurisdiction','source_reference'):
        require(custody.known(l[key]), 'NFL_PRIVATE_LISTING_APPLICABILITY_MISSING')
    require(clock(l['effective_from']) <= clock(q['recorded_at']) <= clock(at) < clock(l['effective_until']),
            'NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED')
    require(clock(r['effective_from']) <= clock(q['recorded_at']) <= clock(at) < clock(r['effective_until']),
            'NFL_PRIVATE_RULE_EDITION_CONFLICT')
    return dict(l,event=e,quote=q,adapter_identity=ADAPTER,applicable_rule_document_sha256=p['objects']['rules']['body_sha256'],
                amendments=[],probability_scope='conditional_on_decided_full_game_result',operator_update_at=None)
