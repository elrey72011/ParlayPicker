"""Run activation checks and preserve successes and explicit blockers."""
import subprocess,sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
commands=[
 ['scripts/prediction_evidence.py','status','--database','data/prediction_evidence/evidence.sqlite3'],
 ['scripts/prediction_evidence.py','restore','--database','data/prediction_evidence/evidence.sqlite3'],
 ['scripts/prediction_evidence.py','refresh','--database','data/prediction_evidence/evidence.sqlite3'],
 ['scripts/validate_sport_deployment.py','--database','data/prediction_evidence/evidence.sqlite3','--all-sports','--output','output/sport-validation'],
 ['scripts/build_wager_policy.py','--validation-dir','output/sport-validation','--output','output/wager-policy-candidate.json'],
 ['scripts/verify_wager_policy.py','--policy','output/wager-policy-candidate.json','--validation-dir','output/sport-validation'],
 ['scripts/exposure_ledger.py','status','--ledger','data/exposure/exposure.sqlite3'],
 ['scripts/exposure_ledger.py','snapshot','--ledger','data/exposure/exposure.sqlite3','--output','output/exposure-snapshot.json'],
 ['scripts/exposure_ledger.py','verify-snapshot','--snapshot','output/exposure-snapshot.json'],
 ['scripts/validate_live_activation.py','--mode','hermetic','--require-funded','--output','output/activation-hermetic.json'],
 ['scripts/validate_live_activation.py','--mode','production','--database','data/prediction_evidence/evidence.sqlite3','--policy','data/policies/active_wager_policy.json','--ledger','data/exposure/exposure.sqlite3','--output','output/activation-production.json'],
 ['scripts/validate_live_activation.py','--mode','replay','--database','data/prediction_evidence/evidence.sqlite3','--output','output/activation-replay.json'],
 ['scripts/validate_live_activation.py','--mode','parlay','--fixtures','tests/fixtures/activation','--output','output/activation-parlay.json'],
]
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--full-test-result',default='test-results/activation-final.txt');parser.add_argument('--acceptance-result',default='test-results/activation-focused.txt');args=parser.parse_args()
 out=ROOT/'output';out.mkdir(exist_ok=True);records=[]
 for command in commands:
  run=subprocess.run([sys.executable,*command],cwd=ROOT,text=True,capture_output=True,encoding='utf-8')
  records.append({'command':'python '+' '.join(command),'exit_code':run.returncode,'output':run.stdout+run.stderr})
  print(command[0],run.returncode)
 proof={'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'commands':records,'ENGINEERING_NONZERO_PATH_VERIFIED':json.loads((out/'activation-hermetic.json').read_text()).get('ENGINEERING_NONZERO_PATH_VERIFIED',False) if (out/'activation-hermetic.json').exists() else False,'REAL_SPORT_AUTHORITY_ACTIVE':False,'AUTOMATED_BET_PLACEMENT':False,'external_configuration_present':{k:bool(os.environ.get(k)) for k in ('PARLAYPICKER_DRIVE_FOLDER_ID','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT','PARLAYPICKER_WAGER_POLICY_PATH','PARLAYPICKER_EXPOSURE_LEDGER')}}
 if (out/'activation-production.json').exists():proof.update(json.loads((out/'activation-production.json').read_text()))
 def read_json(path):return json.loads(path.read_text()) if path.exists() else None
 proof['candidate_policy_hash']=(read_json(out/'wager-policy-candidate.json') or {}).get('policy_hash')
 proof['active_policy_hash']=(read_json(ROOT/'data/policies/active_wager_policy.json') or {}).get('policy_hash')
 proof['strict_validation_hashes']={f.stem:json.loads(f.read_text()).get('validation_hash') for f in (out/'sport-validation').glob('*.json')}
 exposure=read_json(out/'exposure-snapshot.json') or {}
 proof['exposure_snapshot_hash']=exposure.get('snapshot_hash');proof['exposure_as_of']=exposure.get('as_of')
 proof['full_test_result']=(ROOT/args.full_test_result).read_text(encoding='utf-8').splitlines()[-1]
 proof['activation_acceptance_result']=(ROOT/args.acceptance_result).read_text(encoding='utf-8').splitlines()[-1]
 proof['real_qualified_parlay_count']=0 if proof.get('FUNDED_CANDIDATES')==0 else None
 proof['real_actual_price_verified_actionable_parlay_count']=0 if proof.get('FUNDED_CANDIDATES')==0 else None
 proof['synthetic_parlay_proof']=read_json(out/'activation-parlay.json')

 (out/'activation-proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
 (out/'activation-proof.md').write_text('# Activation proof\n\nEngineering path and real authority are separate.\n\n```json\n'+json.dumps(proof,indent=2)+'\n```\n',encoding='utf-8')
if __name__=='__main__':main()
