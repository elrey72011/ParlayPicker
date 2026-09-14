"""Owner-reviewable, hash-bound policy artifacts; activation is always explicit."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json
from core.exposure_ledger import digest
from core.sport_policy import research_policies,SportPolicy
from core.wager_decisions import aware

def build(validation_dir):
    results={}
    for path in Path(validation_dir).glob('*.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('sport') not in research_policies():continue
        if digest({k:v for k,v in r.items() if k!='validation_hash'})!=r.get('validation_hash'):raise ValueError('Validation hash mismatch')
        results[r['sport']]=r
    policies={s:asdict(p) for s,p in research_policies().items()}
    for sport,r in results.items():
        if r['deployment_state']!='UNVALIDATED':
            values=dict(r['supported_policy']['policy']);values['validation_id']=r['validation_hash'];policies[sport]=values
    value={'schema':'activation-v1','sports':policies,'validation_results':results}
    value['policy_hash']=digest(value)
    return value

def verify(value,*,now=None,test_mode=False):
    now=now or datetime.now(timezone.utc)
    if value.get('schema')!='activation-v1' or digest({k:v for k,v in value.items() if k not in {'policy_hash','activation'}})!=value.get('policy_hash'):raise ValueError('Policy hash mismatch')
    for sport in value['sports']:
        verify_sport(value,sport,now=now,test_mode=test_mode)
    return value


def verify_sport(value,sport,*,now=None,test_mode=False):
    now=now or datetime.now(timezone.utc)
    if digest({k:v for k,v in value.items() if k not in {'policy_hash','activation'}})!=value.get('policy_hash'):raise ValueError('Policy hash mismatch')
    values=value['sports'][sport]
    p=SportPolicy(**values)
    if p.sport!=sport:raise ValueError('Sport mismatch')
    if not test_mode and any(x in p.validation_id.upper() for x in ('TEST','SYNTHETIC')):raise ValueError('Test policy prohibited in production')
    if p.deployment_state=='UNVALIDATED':
        if any(getattr(p,k)!=0 for k in ('provisional_stake_cap','standard_stake_cap','premium_stake_cap','kelly_fraction','sport_exposure_cap')):raise ValueError('Nonzero unvalidated cap')
        return
    r=value['validation_results'].get(sport,{})
    if r.get('test_only') and not test_mode:raise ValueError('Test validation prohibited in production')
    if r.get('blockers') or r.get('tier_results',{}).get(p.deployment_state) is None or r['tier_results'][p.deployment_state]:raise ValueError('Validation gates have not passed')
    if digest({k:v for k,v in r.items() if k!='validation_hash'})!=r.get('validation_hash') or p.validation_id!=r.get('validation_hash'):raise ValueError('Validation binding mismatch')
    if r.get('deployment_state')!=p.deployment_state:raise ValueError('State mismatch')
    if aware(r.get('expires_at')) is None or aware(r['expires_at'])<=now:raise ValueError('Expired validation: '+sport)
    if not test_mode:
        if r.get('validation_schema')!='activation-validation-v2':raise ValueError('Incomplete validation schema')
        if r.get('uncertainty',{}).get('coverage') is None or r.get('uncertainty',{}).get('blockers'):raise ValueError('Uncertainty study unavailable')
        if not r.get('prior_validation') or r['prior_validation'].get('blockers'):raise ValueError('Prior validation unavailable')
    from core.wager_decisions import finite
    tier=p.deployment_state.split('_')[0].lower()
    if getattr(p,tier+'_stake_cap')<=0:raise ValueError('Missing active maturity cap')
    limits=r['supported_policy'].get('exposure_limits',{})
    if any(finite(limits.get(k)) is None or not 0<=finite(limits[k])<=1 for k in ('total_cap','daily_cap','weekly_cap','game_cap','team_cap')):raise ValueError('Missing required exposure cap')
    expected=dict(r['supported_policy']['policy'],validation_id=r['validation_hash'])
    if expected!=values:raise ValueError('Policy differs from validated settings')
    outage=r['supported_policy'].get('gemini_outage')
    if outage:
        from core.wager_decisions import finite
        cap,mult=finite(outage.get('cap')),finite(outage.get('multiplier'))
        if outage.get('mode')!='capped' or cap is None or not 0<cap<=.01 or mult is None or not 0<mult<1:raise ValueError('Invalid Gemini outage configuration')

def activate(candidate,destination,*,confirm=False,now=None):
    if not confirm:raise ValueError('Owner --confirm required')
    value=verify(json.loads(Path(candidate).read_text(encoding='utf-8')),now=now)
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    previous=digest(json.loads(path.read_text())) if path.exists() else None
    value['activation']={'at':(now or datetime.now(timezone.utc)).isoformat(),'candidate_hash':value['policy_hash'],'prior_policy_hash':previous}
    receipt=path.parent/('activation-'+digest(value)+'.json')
    if not receipt.exists():receipt.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)
    return value
