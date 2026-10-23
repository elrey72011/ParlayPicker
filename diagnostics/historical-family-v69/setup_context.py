"""Authenticate shallow subject sources and explicitly declared Git objects."""
from pathlib import Path
import argparse,hashlib,io,json,platform,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--subject',choices=['baseline','candidate'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--no-fetch',action='store_true');a=p.parse_args()
D=Path(__file__).resolve().parent;m=json.loads((D/'manifest.json').read_bytes());s=m['sources'][a.subject];a.out.mkdir(exist_ok=True)
for name,digest in m['data_sha256'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,('TOOL_CHANGED',name)
def git(*args):return subprocess.check_output(['git',*args],cwd=a.root,text=True).strip()
assert git('rev-parse','HEAD')==s['tested_head'] and git('rev-parse','HEAD^{tree}')==s['tree']
assert [line[7:]for line in git('cat-file','-p','HEAD').splitlines()if line.startswith('parent ')]==s['parents']
assert git('rev-parse','--is-shallow-repository')=='true'
for obj in s['required_objects']:
    if subprocess.run(['git','cat-file','-e',obj],cwd=a.root,capture_output=True).returncode:
        assert not a.no_fetch,('REQUIRED_OBJECT_MISSING',obj)
        subprocess.run(['git','fetch','--depth','1','origin',obj],cwd=a.root,check=True)
    subprocess.run(['git','cat-file','-e',obj],cwd=a.root,check=True,capture_output=True)
proof=json.loads((D/(a.subject+'-proof.json')).read_bytes())
for name,row in proof.items():
    with io.FileIO(a.root/name,'r')as stream:assert hashlib.sha256(stream.read()).hexdigest()==row['sha256'],('PHYSICAL_SOURCE_CHANGED',name)
assert not git('status','--porcelain')
git_version=subprocess.check_output(['git','--version'],text=True).strip();assert git_version=='git version 2.55.0'
assert sys.version_info[:3]==(3,12,15)
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze']).decode()
def canonical(name):return name.lower().replace('_','-').replace('.','-')
versions={canonical(line.split('==')[0]):line.split('==')[1]for line in freeze.splitlines()if '=='in line}
for line in (D/'constraints.txt').read_text().splitlines():
    if '=='in line:
        name,version=line.split('==');assert versions.get(canonical(name))==version,('DEPENDENCY_CHANGED',name)
(a.out/(a.subject+'-environment.json')).write_text(json.dumps(dict(subject=a.subject,head=s['tested_head'],tree=s['tree'],parents=s['parents'],physical_sources=len(proof),python=sys.version,platform=platform.platform(),git=git_version,method_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest()),indent=2)+'\n')
(a.out/(a.subject+'-packages.txt')).write_text(freeze)
