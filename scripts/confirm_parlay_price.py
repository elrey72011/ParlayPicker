"""Record owner-confirmed ticket odds and calculate a recommendation. No execution."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
 from core.activation_policy import verify_sport
 from core.exposure_ledger import snapshot,digest
 from core.parlay_confirmation import recommend
 from app_core.public_board import validate_package
 from app_core.prediction_evidence import load_snapshots
 p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--ticket-id',required=True);p.add_argument('--decimal-odds',type=float,required=True);p.add_argument('--sportsbook',required=True);p.add_argument('--policy',default='data/policies/active_wager_policy.json');p.add_argument('--ledger',default='data/exposure/exposure.sqlite3');p.add_argument('--database',default='data/prediction_evidence/evidence.sqlite3');p.add_argument('--output',required=True);p.add_argument('--confirm',action='store_true');a=p.parse_args()
 if not a.confirm:raise ValueError('Explicit owner confirmation required')
 from core.owner_parlay_confirmation import confirm_ticket
 record=confirm_ticket(json.loads(Path(a.package).read_text()),a.ticket_id,a.sportsbook,a.decimal_odds,a.policy,a.ledger,a.database,confirmed=a.confirm)
 target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(record,indent=2),encoding='utf-8')
 result=record['recommendation']
 print(json.dumps(result,indent=2))
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (ValueError,KeyError,OSError,StopIteration) as e:print('BLOCKED:',str(e));raise SystemExit(1)
