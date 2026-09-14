"""Shared implementation for owner activation commands. No execution APIs."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.exposure_ledger import digest

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str),encoding='utf-8')

def main(kind):
    p=argparse.ArgumentParser()
    if kind=='validate_sport_deployment':
        p.add_argument('--database',required=True);p.add_argument('--all-sports',action='store_true');p.add_argument('--sport');p.add_argument('--output',required=True);p.add_argument('--plan-dir',default='data/validation_plans');p.add_argument('--freeze-plan')
        a=p.parse_args();from core.activation_validation import validate,freeze_plan
        if a.freeze_plan:
            value=freeze_plan(a.database,json.loads(Path(a.freeze_plan).read_text()))
            write(Path(a.plan_dir)/(value['sport']+'.json'),value);print('FROZEN_PLAN',value['plan_hash']);return 0
        from core.sport_policy import SPORTS
        for sport in SPORTS if a.all_sports else [a.sport]:
            if sport not in SPORTS:raise ValueError('Unsupported sport')
            plan=Path(a.plan_dir)/(sport+'.json');r=validate(a.database,sport,json.loads(plan.read_text()) if plan.exists() else None)
            write(Path(a.output)/(sport+'.json'),r)
            Path(a.output,sport+'.md').write_text('# '+sport+'\n\n'+r['deployment_state']+'\n\n'+json.dumps(r,indent=2),encoding='utf-8')
            print(sport,r['deployment_state'],r['blockers'])
    elif kind in {'build_wager_policy','verify_wager_policy','activate_wager_policy'}:
        from core.activation_policy import build,verify,activate
        if kind=='build_wager_policy':
            p.add_argument('--validation-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=build(a.validation_dir);write(a.output,r);print({s:v['deployment_state'] for s,v in r['sports'].items()})
        elif kind=='verify_wager_policy':
            p.add_argument('--policy',required=True);p.add_argument('--validation-dir',required=True);a=p.parse_args();r=verify(json.loads(Path(a.policy).read_text()))
            for sport,result in r['validation_results'].items():
                source=json.loads(Path(a.validation_dir,sport+'.json').read_text())
                if source!=result:raise ValueError('Validation directory mismatch')
            print('POLICY_VERIFIED',r['policy_hash'])
        else:
            p.add_argument('--candidate',required=True);p.add_argument('--destination',required=True);p.add_argument('--confirm',action='store_true');a=p.parse_args();r=activate(a.candidate,a.destination,confirm=a.confirm);print('ACTIVATED',r['policy_hash'])
    elif kind=='exposure_ledger':
        from core import exposure_ledger as ledger
        p.add_argument('action',choices=['status','snapshot','verify-snapshot','configure','record']);p.add_argument('--ledger',default='data/exposure/exposure.sqlite3');p.add_argument('--output');p.add_argument('--snapshot');p.add_argument('--input');p.add_argument('--confirm',action='store_true');a=p.parse_args()
        if a.action=='status':print(json.dumps({'events':len(ledger.events(a.ledger)),'bankroll_configured':any(e['status']=='CONFIGURED' for e in ledger.events(a.ledger))}))
        elif a.action=='snapshot':r=ledger.snapshot(a.ledger);write(a.output,r);print('SNAPSHOT',r['snapshot_hash'])
        elif a.action=='verify-snapshot':ledger.verify_snapshot(json.loads(Path(a.snapshot).read_text()));print('FRESH_EXPOSURE_VERIFIED')
        else:
            r=json.loads(Path(a.input).read_text());print(ledger.append(a.ledger,r,confirmed=a.confirm))
    elif kind=='validate_live_activation':
        p.add_argument('--mode',required=True,choices=['hermetic','production','replay','parlay']);p.add_argument('--database',default='data/prediction_evidence/evidence.sqlite3');p.add_argument('--policy',default='data/policies/active_wager_policy.json');p.add_argument('--ledger',default='data/exposure/exposure.sqlite3');p.add_argument('--require-funded',action='store_true');p.add_argument('--fixtures');p.add_argument('--output',required=True);a=p.parse_args()
        if a.mode in {'hermetic','parlay'}:
            fixture=Path(a.fixtures or 'tests/fixtures/activation')/'config.json'
            if not fixture.exists() or json.loads(fixture.read_text()).get('test_only') is not True:raise ValueError('Explicit test-only fixture required')
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
            from activation_fixture import engineering_proof
            r=engineering_proof(parlay=a.mode=='parlay')
        elif a.mode=='replay':
            import sqlite3,hashlib
            before=Path(a.database).read_bytes()
            with sqlite3.connect(Path(a.database).resolve().as_uri()+'?mode=ro',uri=True) as db:
                records=db.execute('SELECT candidates,decisions,inputs,payload_hash FROM snapshots').fetchall()
                for ca,de,inp,expected in records:
                    if hashlib.sha256('\0'.join([ca,de,inp]).encode()).hexdigest()!=expected:raise ValueError('Immutable snapshot hash mismatch')
            after=Path(a.database).read_bytes()
            if before!=after:raise ValueError('Replay modified evidence database')
            r={'historical_snapshots':len(records),'historical_promotions':0,'automated_bet_placement':False}
        else:
            import os
            os.environ['PARLAYPICKER_WAGER_POLICY_PATH']=a.policy;os.environ['PARLAYPICKER_EXPOSURE_LEDGER']=a.ledger
            from core.live_wager_contract import load_configuration
            policies,config,reason=load_configuration(datetime.now(timezone.utc))
            active=[s for s,v in policies.items() if v.deployment_state!='UNVALIDATED']
            from app_core.prediction_evidence import load_snapshots
            from core.live_wager_contract import finalize_live_wagers
            from core.wager_decisions import aware
            import pandas as pd
            current=datetime.now(timezone.utc);funded=[];blocked={}
            # Evaluate one current cohort, never sum independently allocated historical runs.
            snapshots=load_snapshots(a.database)
            for sid,audit,final in snapshots[-1:]:
                if audit.empty or not any(aware(x) and aware(x)>current for x in audit.get('game_start_utc',[])):continue
                bankroll=config.get('exposure',{}).get('bankroll')
                if bankroll is None:blocked['bankroll_not_configured']=1;continue
                output,_=finalize_live_wagers(audit,final,bankroll,now=current,policies=policies,config=config)
                for row in output.to_dict('records'):
                    amount=float(row.get('production_bet_amount') or 0)
                    if amount>0:funded.append(row)
                    else:
                        why=row.get('production_gate_reason') or 'missing_authority'
                        blocked[why]=blocked.get(why,0)+1
            if not funded and not blocked:blocked[reason or 'no_current_strict_candidates']=1
            r={'REAL_SPORT_AUTHORITY_ACTIVE':bool(active and funded),'ACTIVE_VALIDATED_SPORTS':active,'FUNDED_CANDIDATES':len(funded),'TOTAL_RECOMMENDED_STAKE':sum(float(x['production_bet_amount']) for x in funded),'BLOCKED_BY_REASON':blocked,'AUTOMATED_BET_PLACEMENT':False}

        write(a.output,r);print(json.dumps(r,indent=2))
        if a.require_funded and not(r.get('ENGINEERING_NONZERO_PATH_VERIFIED') or r.get('FUNDED_CANDIDATES',0)>0):return 1
    return 0

if __name__=='__main__':raise SystemExit(main(Path(sys.argv[0]).stem))
