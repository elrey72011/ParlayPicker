"""Owner-confirmed ticket prices produce recommendations, never orders."""
from core.exposure_ledger import digest,verify_snapshot
from core.wager_decisions import aware,finite
from app_core.production_parlays import production_parlay_leg_eligible,correlation_status

def recommend(ticket,confirmation,policy,exposure,*,now):
    result={'ticket_id':ticket['parlay_id'],'status':'QUALIFIED - VERIFY TICKET PRICE','recommended_stake':0.,'automated_bet_placement':False}
    if not confirmation:return result
    reasons=[];legs=ticket['legs']
    if confirmation.get('ticket_hash')!=digest(ticket):reasons.append('ticket_changed')
    if confirmation.get('sportsbook')!=ticket['sportsbook']:reasons.append('sportsbook_mismatch')
    at=aware(confirmation.get('confirmed_at'))
    if at is None or not 0<=(now-at).total_seconds()<=1800:reasons.append('stale_confirmation')
    if not all(production_parlay_leg_eligible(r,now) for r in legs):reasons.append('leg_not_eligible')
    if correlation_status(legs)!='LOW':reasons.append('correlation_unknown')
    # Frechet lower bound does not assume independent legs or invent correlation.
    # Its use and stake policy must still be validated, not merely switched on.
    if policy.get('joint_method')!='frechet_lower' or not policy.get('validation_id'):reasons.append('unvalidated_joint_policy')
    if aware(policy.get('expires_at')) is None or aware(policy['expires_at'])<=now:reasons.append('expired_parlay_policy')
    odds=finite(confirmation.get('decimal_odds'))
    if odds is None or odds<=1:reasons.append('invalid_actual_price')
    try:verify_snapshot(exposure,now=now)
    except (ValueError,TypeError,KeyError):reasons.append('invalid_exposure')
    if reasons:return dict(result,blockers=reasons)
    p=max(0.,sum(r['wager_contract']['conservative_probability'] for r in legs)-len(legs)+1)
    ev=p*odds-1
    if ev<=0:return dict(result,blockers=['nonpositive_actual_price_ev'],conservative_ev=ev)
    cap=finite(policy.get('stake_cap'));fraction=finite(policy.get('kelly_fraction'))
    if cap is None or not 0<cap<=.01 or fraction is None or not 0<fraction<=1:return dict(result,blockers=['invalid_parlay_cap'])
    used=exposure['committed'];limits=[cap,ev/(odds-1)*fraction]
    for key,limit in [('total','total_cap'),('daily','daily_cap'),('weekly','weekly_cap')]:limits.append(exposure[limit]-used.get(key,0))
    for r in legs:
        c=r['wager_contract'];sport=c['sport'];game=c['game_id']
        limits.append(exposure['game_cap']-used.get(f'game:{sport}:{game}',0))
        sport_cap=finite(policy.get('sport_caps',{}).get(sport))
        limits.append((sport_cap or 0)-used.get(f'sport:{sport}',0))
        teams=c.get('team_ids') or r.get('team_ids')
        if not isinstance(teams,list) or len(teams)!=2:return dict(result,blockers=['missing_team_exposure_identity'])
        for team in teams:limits.append(exposure['team_cap']-used.get(f'team:{sport}:{team}',0))
    stake=max(0,min(limits))*exposure['bankroll']
    return dict(result,status='ACTIONABLE PARLAY RECOMMENDATION' if stake>0 else 'PASS',recommended_stake=stake,conservative_probability=p,conservative_ev=ev,actual_decimal_odds=odds,confirmation_hash=digest(confirmation))

