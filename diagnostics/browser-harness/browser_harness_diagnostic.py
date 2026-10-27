"""Receipt first, sanitized failure retention, one qualification invocation."""
import os
import sys

workspace=os.environ['GITHUB_WORKSPACE']
out=os.path.join(workspace,'browser-evidence')
os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'startup.json'),'x',encoding='utf-8') as stream:
    stream.write('{"phase":"BOOTSTRAP","automatic_retries":0}\n')
phase='IMPORTING_QUALIFICATION'
try:
    import qualification
    status=qualification.run(workspace,out)
except BaseException as error:
    status=2
    phase=getattr(sys.modules.get('qualification'),'PHASE',phase)
    # No raw exception or traceback; even optional JSON import may have failed.
    safe=type(error).__name__
    with open(os.path.join(out,'startup-failure.json'),'w',encoding='utf-8') as stream:
        stream.write('{"phase":"'+phase+'","exception_class":"'+safe+'","raw_exception_retained":false,"qualification_passed":false}\n')
    print('Diagnostic failed: '+phase+' '+safe,flush=True)
sys.exit(status)
