"""Owner-confirmed ticket service shared by CLI and authenticated UI."""
import json
from pathlib import Path
from datetime import datetime,timezone
from core.activation_policy import verify_sport
from core.exposure_ledger import snapshot,digest
from core.parlay_confirmation import recommend
from app_core.public_board import validate_package
from app_core.prediction_evidence import load_snapshots


def confirm_ticket(package, ticket_id, sportsbook, decimal_odds, policy_path, ledger, database, *, confirmed=False, now=None):
 if not confirmed: raise ValueError('Explicit owner confirmation required')
 # Do not mutate the saved package when enriching private exposure identity.
 package=json.loads(json.dumps(package))
 now=now or datetime.now(timezone.utc);validate_package(package)
 ticket=next(t for t in package['parlays'] if t['parlay_id']==ticket_id)
 active=json.loads(Path(policy_path).read_text());specs=[]
 if not active.get('activation'):raise ValueError('Policy is not owner activated')
 for sport in {r['sport'] for r in ticket['legs']}:
  verify_sport(active,sport,now=now)
  r=active['validation_results'][sport];spec=r['supported_policy'].get('parlay_policy')
  if not spec:raise ValueError('No validated parlay policy: '+sport)
  for leg in ticket['legs']:
   if leg['sport']==sport and any(leg['wager_contract'].get(k)!=r['versions'].get(k) for k in ('model_version','calibration_version')):raise ValueError('Ticket model/calibration version changed')
  specs.append(dict(spec,validation_id=r['validation_hash'],expires_at=r['expires_at']))
 # Multi-sport ticket uses the strictest validated caps and expiry.
 policy=dict(specs[0]);policy['stake_cap']=min(s['stake_cap'] for s in specs);policy['kelly_fraction']=min(s['kelly_fraction'] for s in specs);policy['expires_at']=min(s['expires_at'] for s in specs)
 policy['sport_caps']={s:v['sport_exposure_cap'] for s,v in active['sports'].items()}
 import ast
 saved=[row for sid,audit,final in load_snapshots(database) for row in audit.to_dict('records')]
 for leg in ticket['legs']:
  c=leg['wager_contract'];matches=[r for r in saved if str(r.get('game_id',r.get('matchup_id')))==str(c['game_id']) and r.get('market_type')==c['market_type'] and r.get('best_pick')==c['selection'] and r.get('odds_american')==c['odds']]
  teams={str(r.get('team_ids')) for r in matches}
  if len(teams)!=1:raise ValueError('Missing or ambiguous immutable team exposure identity')
  ids=ast.literal_eval(teams.pop());leg['team_ids']=ids
 confirmation={'ticket_hash':digest(ticket),'sportsbook':sportsbook,'decimal_odds':decimal_odds,'confirmed_at':now.isoformat()}
 result=recommend(ticket,confirmation,policy,snapshot(ledger,now=now),now=now)
 record={'confirmation':confirmation,'recommendation':result}
 receipt=Path(ledger).parent/'ticket-confirmations'/(digest(record)+'.json');receipt.parent.mkdir(parents=True,exist_ok=True)
 if not receipt.exists():receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
 return record
