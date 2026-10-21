"""Authenticate the exact shallow source before any application import."""
from pathlib import Path
import argparse,hashlib,io,json,platform,subprocess,sys
P=argparse.ArgumentParser();P.add_argument('root');P.add_argument('out');P.add_argument('--no-fetch',action='store_true');a=P.parse_args()
D=Path(__file__).resolve().parent;root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=True)
m=json.loads((D/'manifest.json').read_bytes())
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')==m['tested_merge'] and git('rev-parse','HEAD^{tree}')==m['source_tree']
assert [x[7:]for x in git('cat-file','-p','HEAD').splitlines()if x.startswith('parent ')]==m['ordered_parents']
for needed in m['required_immutable_git_objects']:
    if subprocess.run(['git','cat-file','-e',needed],cwd=root,capture_output=True).returncode:
        assert not a.no_fetch,('MISSING_REQUIRED_OBJECT',needed)
        subprocess.run(['git','fetch','--depth','1','origin',needed],cwd=root,check=True)
    assert subprocess.run(['git','cat-file','-e',needed],cwd=root,capture_output=True).returncode==0
for path,row in json.loads((D/'source-proof.json').read_bytes()).items():
    with io.FileIO(root/path,'r')as f:assert hashlib.sha256(f.read()).hexdigest()==row['sha256'],('SOURCE_CHANGED',path)
assert not git('status','--porcelain')
data=dict(head=m['tested_merge'],tree=m['source_tree'],parents=m['ordered_parents'],python=sys.version,platform=platform.platform(),shallow=git('rev-parse','--is-shallow-repository'),physical_files_verified=49,manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest())
(out/'environment.json').write_text(json.dumps(data,indent=2)+'\n')
(out/'packages.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
