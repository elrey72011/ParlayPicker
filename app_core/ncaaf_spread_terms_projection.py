"""Explicit owner-v2 scope interpretation; never a provider declaration/catalog.

Only exact already reviewed quote/terms subjects may supply missing period and
settlement semantics. Original raw bodies, selected side, price and clocks stay
unchanged. The strict original routes never select this context.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
VERSION='ncaaf-owner-applicable-spread-terms-v3'
from app_core.ncaaf_owner_terms_projection import SELECTED
CONDITIONAL_VERSION='ncaaf-private-mathematical-assumptions-v1'
def require(ok,code):
    if not ok:raise ValueError(code)


def valid_line(packet,line):
    from app_core.ncaaf_spread_mass import target_contract
    from app_core.ncaaf_owner_spread import integer
    target_contract('spread_home',line,integer=integer(packet))
    return True


def subjects(packet):
    from app_core.ncaaf_owner_spread import conditional
    if conditional(packet):
        return conditional_subjects(packet)
    o=packet['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']
    q=o['quote'];t=o['source_review']['terms_review']
    require(packet['payload']['version'] in {'ncaaf-owner-integer-inputs-v1','ncaaf-owner-named-half-point-inputs-v1'} and q['market_type'] in {'spread_home','spread_away'}
        and valid_line(packet,q['point']) and t['reviewer']=='Robert Velarde' and not q['book'].lower().startswith('novig')
        and all(q.get(k)==t[k] for k in ('operator','product','listing_id','period'))
        and t['period']=='full_game' and q['rules']==t['settlement']=='full_game_including_overtime_binary_win_push_loss'
        and q['period_source']==q['rules_source']==VERSION,'NCAAF_CUSTODY_QUOTE_CONFLICT')
    from app_core.source_evidence_intake import absent
    require(all(isinstance(t[k],str) and t[k].strip() and not absent(t[k]) for k in ('operator','product','listing_id','rule_edition')),'NCAAF_CUSTODY_QUOTE_CONFLICT')
    return deepcopy(q),deepcopy(t)


@contextmanager
def selected(packet):
    token=SELECTED.set(subjects(packet))
    try:yield
    finally:SELECTED.reset(token)


def transport_facts(game,book,market,outcome):
    scope=SELECTED.get()
    if scope is None:return None
    q,t=scope
    identity=(game.get('id'),game.get('sport_key'),game.get('home_team'),game.get('away_team'),game.get('commence_time'),book.get('key'),market.get('key'),outcome.get('name'),outcome.get('point'),outcome.get('price'),market.get('last_update'))
    expected=(q['provider_event_id'],'americanfootball_ncaaf',q['event_home_team'],q['event_away_team'],q['event_start_utc'],q['book'],'spreads',q['event_home_team'] if q['market_type']=='spread_home' else q['event_away_team'],q['point'],q['price'],q['recorded_at'])
    if identity!=expected:return None
    require(market.get('period') in (None,'',t['period']) and market.get('settlement_rules') in (None,'',t['settlement']), 'NCAAF_CUSTODY_QUOTE_CONFLICT')
    if q['period_source'] in (VERSION,CONDITIONAL_VERSION):
        return dict(period=t['period'],period_source=q['period_source'],rules=t['settlement'],rules_source=q['rules_source'])
    return dict(period=t['period'],period_source=VERSION,rules=t['settlement'],rules_source=VERSION)


def matches(packet,quotes):
    q,t=subjects(packet);out=[]
    marker=CONDITIONAL_VERSION if q['period_source']==CONDITIONAL_VERSION else VERSION
    for actual in quotes:
        v=deepcopy(actual)
        if all(v.get(k)==q.get(k) for k in ('book','market_type','point','price','recorded_at','provider_event_id','provider_namespace','event_home_team','event_away_team','event_start_utc')):
            require(v.get('period') in (None,'',q['period']) and v.get('rules') in (None,'',q['rules']), 'NCAAF_CUSTODY_QUOTE_CONFLICT')
            for field,source in (('period','period_source'),('rules','rules_source')):
                require(v.get(source) in (None,'','market.period','market.settlement_rules',VERSION,CONDITIONAL_VERSION),'NCAAF_CUSTODY_QUOTE_CONFLICT')
                v[field]=q[field];v[source]=marker
        out.append(v)
    return out


def presentation(packet,source):
    from app_core import producer_provenance as producer
    value=dict(source);original=producer._matches(source);projected=matches(packet,original)
    require(len(original)==len(projected)==1,'NCAAF_CUSTODY_QUOTE_CONFLICT')
    value['provider_quotes']=json.dumps(projected,sort_keys=True,separators=(',',':'))
    # Original provider JSON remains in the exact reviewed custody packet. These
    # derived descriptors explicitly name their owner-terms interpretation.
    return value


def conditional_subjects(packet):
    """Exact reviewed mathematical assumptions, never verified offer admission."""
    from app_core.ncaaf_owner_spread import conditional
    o=packet['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']
    q=o['quote'];t=o['source_review']['terms_review']
    require(conditional(packet) and valid_line(packet,q['point'])
        and q['market_type'] in {'spread_home','spread_away'} and t['reviewer']=='Robert Velarde'
        and q['period_source']==q['rules_source']==CONDITIONAL_VERSION
        and all(q.get(k)==t[k] for k in ('operator','product','listing_id','period'))
        and q['operator']==q['book'] and q['period']=='full_game'
        and q['rules']==t['settlement']=='full_game_including_overtime_binary_win_push_loss'
        and t['product'] is None and t['listing_id'] is None and t['rule_edition'] is None
        and t['rules_document_sha256'] is None,'NCAAF_PRIVATE_ASSUMPTIONS_CONFLICT')
    return deepcopy(q),deepcopy(t)
