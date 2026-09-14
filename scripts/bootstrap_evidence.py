"""Explicit forward evidence job; never activates policy or places wagers."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--database',default='data/prediction_evidence/evidence.sqlite3');p.add_argument('--plan-dir',default='data/validation_plans');p.add_argument('--output',default='output/sport-validation');p.add_argument('--capture-closes',action='store_true');p.add_argument('--grade',action='store_true');p.add_argument('--backup',action='store_true');a=p.parse_args()
    from core.activation_validation import validate
    from core.sport_policy import SPORTS
    result={}
    if a.capture_closes:
        from app_core.activation_closing import capture_live
        result['closing']=capture_live(a.database)
    if a.grade:
        from app_core.prediction_evidence import refresh_outcomes
        result['grading']=refresh_outcomes(a.database)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    for sport in SPORTS:
        plan=Path(a.plan_dir,sport+'.json');r=validate(a.database,sport,json.loads(plan.read_text()) if plan.exists() else None)
        (output/(sport+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
        (output/(sport+'.md')).write_text('# '+sport+'\n\n'+json.dumps(r,indent=2),encoding='utf-8')
        result[sport]={'state':r['deployment_state'],'blockers':r['blockers']}
    if a.backup:
        from app_core.evidence_remote import sync,remote_status
        sync(a.database);result['remote']=remote_status()
    print(json.dumps(result,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
