"""One manual reviewed operation. No scheduling, retries or persistent key.

The operation manifest is data, never executable code. Native custody/configuration
and transport are reused. Every failed launch keeps its exclusive claim.
"""
import argparse
from datetime import datetime,timedelta,timezone
import getpass
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
VERSION='ncaaf-reviewed-target-operation-v1'


def require(ok,code):
    if not ok:raise ValueError(code)


def utc():return datetime.now(timezone.utc).isoformat()


def unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'NCAAF_TARGET_OPERATOR_MANIFEST');result[key]=value
    return result


def envelope(value):
    require(isinstance(value,dict) and set(value)=={'payload','sha256'} and isinstance(value['payload'],dict),
        'NCAAF_TARGET_OPERATOR_MANIFEST')
    raw=json.dumps(value['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(hashlib.sha256(raw).hexdigest()==value['sha256'],'NCAAF_TARGET_OPERATOR_MANIFEST')
    return value['payload']


def verify_checkout(expected,*,root=ROOT):
    """Live full-tree verification, independent of application validators/caches."""
    try:
        def git(*args):return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL)
        require(isinstance(expected,str) and len(expected)==40 and all(c in '0123456789abcdef' for c in expected),
            'NCAAF_TARGET_RUNTIME_CONFLICT')
        require(git('rev-parse','HEAD').decode().strip()==expected,'NCAAF_TARGET_RUNTIME_CONFLICT')
        for args in (('diff','--quiet',expected,'--'),('diff','--cached','--quiet',expected,'--')):
            subprocess.run(['git',*args],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
        for row in git('ls-tree','-rz',expected).split(b'\0'):
            if not row:continue
            meta,name=row.split(b'\t',1);mode,kind,oid=meta.split();relative=Path(name.decode())
            require(kind==b'blob' and not relative.is_absolute() and '..' not in relative.parts,'NCAAF_TARGET_RUNTIME_CONFLICT')
            raw=(root/relative).read_bytes()
            def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest().encode()
            require(blob(raw)==oid or blob(raw.replace(b'\r\n',b'\n'))==oid,'NCAAF_TARGET_RUNTIME_CONFLICT')
    except (OSError,ValueError,TypeError,subprocess.SubprocessError):
        raise ValueError('NCAAF_TARGET_RUNTIME_CONFLICT') from None


def application():
    # Only after authenticated operation data and live source verification.
    global stage,target
    sys.path.insert(0,str(ROOT))
    from app_core import ncaaf_staged_acquisition as stage,ncaaf_reviewed_target as target
    return stage,target


def load(path,limit):
    with Path(path).open('rb') as f:raw=f.read(limit+1)
    require(0<len(raw)<=limit,'NCAAF_TARGET_OPERATOR_SIZE')
    def constant(value):raise ValueError('NCAAF_TARGET_OPERATOR_MANIFEST')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=constant),raw


def preflight(manifest,manifest_hash,configuration_hash,*,at):
    value,raw=load(manifest,128*1024)
    require(hashlib.sha256(raw).hexdigest()==manifest_hash,'NCAAF_TARGET_OPERATOR_MANIFEST')
    m=envelope(value)
    require(set(m)==set('version plan authorization configuration custody_root runtime setup_cutoff latest_start history prequote'.split())
        and m['version']==VERSION,'NCAAF_TARGET_OPERATOR_MANIFEST')
    require(isinstance(m['runtime'],dict),'NCAAF_TARGET_RUNTIME_RECEIPT')
    verify_checkout(m['runtime'].get('code_head'))
    stage,target=application()
    filtered='app_core/ncaaf_filtered_history.py' in m['runtime'].get('sources',{})
    fresh_runtime='app_core/ncaaf_fresh_history.py' in m['runtime'].get('sources',{})
    target.check_runtime_receipt(m['runtime'],filtered=filtered,fresh=fresh_runtime,continuation='app_core/ncaaf_history_continuation.py' in m['runtime'].get('sources',{}))
    # The control receipt binds the manifest subject without a circular future
    # configuration-file hash. The exact file hash remains a reviewed argument.
    identity=stage.model.digest({k:v for k,v in m.items() if k!='runtime'})
    require(m['runtime']==target.runtime_receipt(m['runtime']['code_head'],identity,filtered=filtered,fresh=fresh_runtime,continuation='app_core/ncaaf_history_continuation.py' in m['runtime'].get('sources',{})),'NCAAF_TARGET_RUNTIME_CONFLICT')
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    require(current==m['runtime']['code_head'],'NCAAF_TARGET_RUNTIME_CONFLICT')
    plan,_=load(m['plan'],stage.MAX_BUNDLE);p=stage.validate_plan(plan,executable=True)
    require(p['version'] in {target.HISTORY_PLAN,target.QUOTE_PLAN,'ncaaf-filtered-history-stage-plan-v4','ncaaf-filtered-final-quote-stage-plan-v4','ncaaf-fresh-filtered-history-stage-plan-v5','ncaaf-history-continuation-stage-plan-v6','ncaaf-fresh-filtered-final-quote-stage-plan-v5','ncaaf-continuation-final-quote-stage-plan-v6'},'NCAAF_TARGET_OPERATOR_VERSION')
    config,cr=load(m['configuration'],128*1024)
    require(hashlib.sha256(cr).hexdigest()==configuration_hash,'NCAAF_TARGET_OPERATOR_CONFIGURATION')
    c=stage.configuration(plan,cr,configuration_hash,at=at)['payload']
    a,_=load(m['authorization'],128*1024)
    require(a==c['authorization'] and c['runtime']==m['runtime'] and c['custody']['root']==m['custody_root'],
        'NCAAF_TARGET_OPERATOR_CONFIGURATION')
    now=target.instant(at);w=p['execution_window']
    if filtered:
        require(target.instant(c['review']['reviewed_at'])<target.instant(m['setup_cutoff'])
            and target.instant(w['not_before'])<=now<=target.instant(m['latest_start'])
            and target.instant(m['setup_cutoff'])+timedelta(minutes=5)<=target.instant(w['not_after'])
            and target.instant(m['latest_start'])+timedelta(seconds=120)<=target.instant(w['not_after']), 'NCAAF_TARGET_OPERATOR_WINDOW')
    else:require(target.instant(c['review']['reviewed_at'])<target.instant(m['setup_cutoff'])<target.instant(w['not_before'])
        <=now<=target.instant(m['latest_start'])
        and target.instant(m['latest_start'])+timedelta(seconds=60)<=target.instant(w['not_after']),
        'NCAAF_TARGET_OPERATOR_WINDOW')
    require((m['history'] is m['prequote'] is None) if stage.is_history(p) else all(isinstance(m[k],str) for k in ('history','prequote')),
        'NCAAF_TARGET_OPERATOR_PARENT')
    root=Path(m['custody_root']);require(root.is_absolute(),'NCAAF_STAGED_CUSTODY')
    return m,plan,cr,root


def execute(manifest,manifest_hash,configuration_hash):
    require(sys.stdin.isatty(),'NCAAF_STAGED_PRIVATE_TERMINAL_REQUIRED')
    at=utc();m,plan,raw,root=preflight(manifest,manifest_hash,configuration_hash,at=at)
    stage,target=application()
    native=runpy.run_path(str(ROOT/'scripts/ncaaf_staged_acquisition.py'),run_name='REVIEWED_TARGET_NATIVE_CLI')
    native['verify_runtime']()
    with stage.controlled(plan,raw,configuration_hash,root=root,at=at):
        require(not any((root/(plan['sha256']+s)).exists() for s in ('.attempts.jsonl','.stage.json')),
            'NCAAF_STAGED_ALREADY_SPENT')
        claim=root/(plan['sha256']+'.operator-claim.json')
        stage._exclusive(claim,stage.model.encode(dict(version=VERSION,manifest_file_sha256=manifest_hash,
            plan_sha256=plan['sha256'],configuration_file_sha256=configuration_hash,claimed_at=at)))
        result=dict(version=VERSION,plan_sha256=plan['sha256'],manifest_file_sha256=manifest_hash,
            configuration_file_sha256=configuration_hash,claim_sha256=hashlib.sha256(claim.read_bytes()).hexdigest(),
            status='STOPPED_NO_RETRY',reason='PRIVATE_ENTITLEMENT_INPUT',provider_attempts='UNKNOWN',retry=False)
        try:
            print('FIRST PROMPT: type INCLUDED, then Enter. NOT an API key. Confirm included entitlement, no paid overage and unchanged permissions.')
            answer=getpass.getpass('Entitlement (masked): ')
            if answer!='INCLUDED':
                result['reason']='ENTITLEMENT_LITERAL_REQUIRED'
                return 1
            answer=None
            result['entitlement_attestation_at']=stage.transport.utc()
            result['reason']='NATIVE_EXECUTION_FAILURE'
            print('SECOND PROMPT: enter the CFBD API key only at the separate native masked prompt (Odds key for a separately authorized quote).')
            argv=['execute',m['plan'],'--authorization',m['authorization'],'--controlled-bindings',m['configuration'],
                '--bindings-sha256',configuration_hash,'--custody-root',m['custody_root'],'--transport-diagnostics']
            if m['history'] is not None:argv+=['--history',m['history'],'--prequote',m['prequote']]
            code=native['main'](argv)
            path=root/(plan['sha256']+'.stage.json')
            if path.exists():
                bundle,br=load(path,stage.MAX_BUNDLE);b=stage.read_bundle(bundle,complete=False);stage.verify_journal(root,bundle)
                from app_core import ncaaf_transport_diagnostics as diagnostic
                _,dr=load(root/(plan['sha256']+'.transport-diagnostics.json'),32*1024)
                diagnostic.read(root,bundle,dr,hashlib.sha256(dr).hexdigest())
                result.update(status=b['status'],reason=b['reason'],provider_attempts=b['attempts'],
                    bundle_sha256=bundle['sha256'],bundle_file_sha256=hashlib.sha256(br).hexdigest(),journal_sha256=b['journal_sha256'],
                    diagnostics_file_sha256=hashlib.sha256(dr).hexdigest(),readback='PASS')
            else:
                result['reason']='NATIVE_STOPPED_BEFORE_BUNDLE'
            return code
        except (EOFError,KeyboardInterrupt):
            result['reason']='PRIVATE_INPUT_INTERRUPTED'
            raise
        except Exception:
            # Preserve failure and journal; never expose arbitrary exception text.
            result['reason']='NATIVE_EXECUTION_FAILURE'
            return 1
        finally:
            answer=None
            journal=root/(plan['sha256']+'.attempts.jsonl')
            if not journal.exists():result['provider_attempts']=0
            result['finished_at']=stage.transport.utc()
            stage._exclusive(root/(plan['sha256']+'.operator-report.json'),stage.model.encode(result))
            print(json.dumps(result,sort_keys=True))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['execute']);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--configuration-sha256',required=True)
    a=p.parse_args(argv)
    try:return execute(a.manifest,a.manifest_sha256,a.configuration_sha256)
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError):
        print(json.dumps(dict(status='STOPPED_NO_RETRY',reason='NCAAF_TARGET_OPERATOR_PREFLIGHT',retry=False)));return 1


if __name__=='__main__':raise SystemExit(main())
