"""Canonical parlay qualification. Estimated tickets never carry stake authority."""
from collections import Counter
from datetime import datetime, timezone
import itertools
import logging
import math
from core.wager_decisions import finite, aware
from core.market_policy import production_market, sport_market_family
from app_core.public_quote_policy import supported_quote


def leg_checks(row, now):
    c=row.get('wager_contract') or {}
    odds=finite(row.get('odds')); p=finite(c.get('conservative_probability'))
    start=aware(row.get('start')); quote=aware(row.get('quote_time'))
    from app_core.public_history import resolved_pick, event_key
    checks=[('spread_total_candidates',production_market(row.get('market')),'moneyline_excluded'),
        ('valid_prices',odds is not None and 100<=abs(odds)<=10000,'invalid_price'),
        ('valid_lines',finite(c.get('line')) is not None and resolved_pick(row),'invalid_line'),
        ('positive_conservative_ev',(finite(c.get('conservative_ev')) or 0)>0 and (finite(c.get('conservative_edge')) or 0)>0 and p is not None and 0<p<1,'negative_conservative_ev'),
        ('fresh_quotes',quote is not None and 0<=(now-quote).total_seconds()<=1800,'stale_quote'),
        ('supported_books',supported_quote(row) and not row.get('quote_time_basis'),'unsupported_book'),
        ('pregame',start is not None and start>now,'game_started'),
        ('identity_verified',c.get('identity_verified') is True and c.get('quote_verified') is True and event_key(row) is not None and c.get('selection')==row.get('pick') and c.get('market_type')==row.get('market') and c.get('odds')==odds and c.get('sportsbook')==row.get('quote_source'),'identity_failure'),
        ('exact_market_validated',c.get('market_family')==sport_market_family(row.get('sport'),row.get('market'))
         and c.get('deployment_state') in {'STANDARD_VALIDATED','PREMIUM_VALIDATED'}
         and all(isinstance(c.get(field),str) and c[field].strip() for field in
                 ('model_id','model_version','calibration_id','calibration_version',
                  'validation_id','validation_artifact_id')),'unvalidated_straight_leg'),
        ('production_eligible',c.get('wager_contract_version')=='live-v1' and c.get('production_eligible') is True and (finite(c.get('production_bet_amount')) or 0)>0,'research_only'),
        ('standard_premium',c.get('maturity') in {'STANDARD','PREMIUM'},'provisional_straight_only' if c.get('maturity')=='PROVISIONAL' else 'research_only'),
        ('secondary_review',c.get('gemini_review_status') in {'APPROVE','CONFIRM','REDUCE'},'gemini_hard_veto' if c.get('gemini_review_status')=='HARD_VETO' else 'gemini_unavailable')]
    return checks


def production_parlay_leg_eligible(row, now=None):
    return all(ok for _,ok,_ in leg_checks(row,now or datetime.now(timezone.utc)))


def correlation_status(legs):
    from app_core.public_history import event_key
    seen=set()
    for row in legs:
        key=event_key(row)
        if key is None: return 'UNKNOWN'
        teams={(key[0],t) for t in key[1:3]}
        if seen & teams: return 'HIGH'
        seen.update(teams)
    # Common scoring environments/model errors are not estimated as independent.
    totals=[r for r in legs if str(r['market']).startswith('total')]
    if len({r['sport'] for r in totals})<len(totals): return 'UNKNOWN'
    return 'LOW'


def canonical_funnel(rows, now=None):
    now=now or datetime.now(timezone.utc)
    counts=Counter({key:0 for key in 'spread_total_candidates valid_prices valid_lines positive_conservative_ev fresh_quotes supported_books pregame identity_verified exact_market_validated production_eligible standard_premium secondary_review provisional standard premium research qualified parlay_eligible same_book_candidates valid_2leg_pairs valid_3leg_combinations'.split()}); counts['total_best_picks']=len(rows); exclusions=Counter(); pool=[]
    for row in rows:
        checks=leg_checks(row,now); reached=True
        for stage,ok,reason in checks:
            reached=reached and bool(ok)
            if reached: counts[stage]+=1
        exclusions.update({reason for _,ok,reason in checks if not ok})
        maturity=(row.get('wager_contract') or {}).get('maturity','RESEARCH')
        counts[maturity.lower()]+=1
        if all(ok for _,ok,_ in checks): pool.append(row)
    # Bounded deterministic pool. Duplicate exact legs do not generate duplicate tickets.
    from app_core.public_history import digest
    pool=list({digest(r):r for r in pool}.values())
    pool.sort(key=lambda r:(-(r['wager_contract']['conservative_probability']),r['game'],r['pick']))
    pool=pool[:20]
    counts['parlay_eligible']=len(pool)
    combinations=[]; partners=set()
    for n in (2,3):
        for indices in itertools.combinations(range(len(pool)),n):
            legs=[pool[i] for i in indices]
            if len({r['quote_source'] for r in legs})!=1: continue
            risk=correlation_status(legs)
            if risk in {'HIGH','UNKNOWN'}:
                exclusions['correlation_excluded']+=1
                if risk=='HIGH': exclusions['same_team_conflict']+=1
                continue
            combinations.append(legs);partners.update(indices)
            counts['valid_2leg_pairs' if n==2 else 'valid_3leg_combinations']+=1
    counts['same_book_candidates']=len(partners)
    exclusions['no_same_book_partner']=len(pool)-len(partners)
    return {'counts':dict(counts),'exclusions':dict(exclusions),'combinations':combinations}


def build_production_parlays(rows, now=None):
    from app_core.public_history import digest
    now=now or datetime.now(timezone.utc)
    funnel=canonical_funnel(rows,now)
    tickets=[]
    for legs in funnel['combinations']:
        legs=sorted(legs,key=lambda r:(r['sport'],r['game'],r['pick']))
        probability=math.prod(r['wager_contract']['conservative_probability'] for r in legs)
        decimal=math.prod(1+(r['odds']/100 if r['odds']>0 else 100/-r['odds']) for r in legs)
        tickets.append(dict(parlay_id=digest(legs),created_at=now.isoformat(),legs=legs,
            game_ids=[r['wager_contract']['game_id'] for r in legs],sportsbook=legs[0]['quote_source'],
            win_estimate=probability,decimal_odds_estimate=decimal,ev_estimate=probability*decimal-1,
            estimated_decimal_odds=decimal,estimated_combined_probability=probability,estimated_ev=probability*decimal-1,
            actual_ticket_price_verified=False,actual_ticket_odds=None,recommended_stake=0.0,
            correlation_status=correlation_status(legs),approved_legs=True,status='QUALIFIED — VERIFY TICKET PRICE'))
    tickets.sort(key=lambda t:(-t['win_estimate'],-t['ev_estimate'],t['parlay_id']))
    # Reuse is bounded independently of the broader research display.
    usage=Counter();selected=[]
    leaders=[next((t for t in tickets if len(t['legs'])==n),None) for n in (2,3)]
    ordered=[t for t in leaders if t is not None]+sorted([t for t in tickets if t not in leaders],key=lambda t:(-t['ev_estimate'],t['parlay_id']))
    for ticket in ordered:
        if any(usage[g]>=2 for g in ticket['game_ids']):continue
        ticket['category']='Conservative' if ticket is leaders[0] else 'Balanced' if ticket is leaders[1] else 'Best EV'
        selected.append(ticket);usage.update(ticket['game_ids'])
        if len(selected)==5:break
    logging.getLogger(__name__).info('PARLAY_FUNNEL %s qualified=%s',funnel['counts'],len(selected))
    logging.getLogger(__name__).info('PARLAY_EXCLUSIONS %s',funnel['exclusions'])
    return selected
