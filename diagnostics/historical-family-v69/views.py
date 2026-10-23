"""Network-blocked setup/read/cleanup comparison; never full-body coverage."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from unittest.mock import patch
import argparse,hashlib,io,json,socket,sys,time,subprocess
A=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--subject',choices=['baseline','candidate'],required=True);args=p.parse_args()
C=args.checkout.resolve();sys.path[:0]=[str(C),str(C/'tests')]
manifest=json.loads((A/'manifest.json').read_bytes());subject=manifest['sources'][args.subject]
for name,digest in manifest['data_sha256'].items():assert hashlib.sha256((A/name).read_bytes()).hexdigest()==digest,('TOOL_CHANGED',name)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip()==subject['tested_head']
assert subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=C,text=True).strip()==subject['tree']
for name,row in json.loads((A/(args.subject+'-proof.json')).read_bytes()).items():
    with io.FileIO(C/name,'r')as stream:assert hashlib.sha256(stream.read()).hexdigest()==row['sha256'],('SOURCE_CHANGED',name)
def blocked(*a,**k):raise AssertionError('FAMILY_QUALIFICATION_EXTERNAL_APPLICATION_TRANSPORT_BLOCKED')
socket.create_connection=socket.getaddrinfo=blocked;socket.socket.connect=blocked
from tests import historical_scope_input_boundary as boundary
specs=json.loads((A/'context-specs.json').read_bytes());rows=[];negative=[]
binding=json.loads(boundary.physical_bytes(C/boundary.BINDING))['historical_input_boundary']
started=time.perf_counter();out=args.out
assert not out.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def save(complete=False):
    with io.open(out,'w',encoding='utf-8',newline='\n')as stream:
        json.dump(dict(schema='MATCHED_FAMILY_VIEWS_V61',recorded_at=datetime.now(timezone.utc).isoformat(),checkout=str(C),
            source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip(),
            complete=complete,contexts=rows,negative_controls=negative,wall_seconds=time.perf_counter()-started,
            interpretation='Paired Ubuntu context samples, not complete test-body or full-release coverage.',
            application_network='BLOCKED',provider_requests=0,release_coverage=False),stream,indent=2);stream.write('\n')
def request(spec):
    return SimpleNamespace(module=SimpleNamespace(__name__=Path(spec['path']).stem,__file__=str(C/spec['path'])),
        node=SimpleNamespace(originalname=spec['node']))
for pass_index,order in enumerate((specs,)):
    for spec in order:
        req=request(spec);before=time.perf_counter();plan=boundary.resolve_context(req);resolved=time.perf_counter()
        assert plan.role is boundary.Role.HISTORICAL
        methods=(Path.open,Path.read_bytes,Path.read_text,io.open);incoming=tuple(boundary._ACTIVE)
        fields={};paths=sorted({boundary.GUARD,'scripts/ci_test_file_costs_v1.json','.github/workflows/ci.yml',
                              *binding['consumers'][spec['path']].get('source_inputs',{})})
        with boundary.InputOwner(plan)as owner:
            boundary.apply_selected_inputs(plan,req,owner.patch);constructed=time.perf_counter()
            for relative in paths:
                path=C/relative;raw=path.read_bytes()
                with path.open('rb')as stream:opened=stream.read()
                text=path.read_text(encoding='utf-8').encode()
                assert opened==raw and text.replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n'),relative
                fields[relative]=dict(bytes=len(raw),sha256=sha(raw),git_sha256=sha(raw.replace(b'\r\n',b'\n')))
            read_done=time.perf_counter()
        ended=time.perf_counter()
        assert tuple(boundary._ACTIVE)==incoming and methods==(Path.open,Path.read_bytes,Path.read_text,io.open)
        rows.append(dict(path=spec['path'],node=spec['node'],pass_index=pass_index,generation=plan.generation,role=plan.role.value,
                         resolve_seconds=resolved-before,setup_seconds=constructed-resolved,read_seconds=read_done-constructed,
                         cleanup_seconds=ended-read_done,total_seconds=ended-before,view_hashes=fields,restored=True))
        save();print(json.dumps({'path':spec['path'],'pass':pass_index,'seconds':ended-before}),flush=True)
families={}
for spec in specs:families.setdefault(spec['shape']['generation'],spec)
for generation,spec in families.items():
    plan=boundary.resolve_context(request(spec));read=boundary.physical_bytes
    methods=(Path.open,Path.read_bytes,Path.read_text,io.open);incoming=tuple(boundary._ACTIVE)
    def changed(path):
        value=read(path)
        return value+b'\n ' if Path(path)==C/boundary.BINDING else value
    try:
        with patch.object(boundary,'physical_bytes',changed):boundary.authenticate_plan(plan)
    except ValueError as exc:
        assert exc.args and exc.args[0]=='HISTORICAL_BINDING_CHANGED'
        negative.append(dict(generation=generation,attack='WARM_BINDING_SUBSTITUTION',code=exc.args[0],rejected=True))
    else:raise AssertionError('WARM_BINDING_SUBSTITUTION_ACCEPTED')
    assert tuple(boundary._ACTIVE)==incoming and methods==(Path.open,Path.read_bytes,Path.read_text,io.open)
save(True)
print(json.dumps({'contexts':len(rows),'negative_controls':len(negative),'complete':True,'wall_seconds':time.perf_counter()-started}))
