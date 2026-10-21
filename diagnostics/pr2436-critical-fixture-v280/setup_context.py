from pathlib import Path
import hashlib,json,os,platform,subprocess,sys
D=Path(__file__).resolve().parent;root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
m=json.loads((D/'manifest.json').read_bytes())
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def parents():return [line.split()[1] for line in git('cat-file','-p','HEAD').splitlines() if line.startswith('parent ')]
assert git('rev-parse','HEAD')==m['tested_merge'] and git('rev-parse','HEAD^{tree}')==m['source_tree']
assert parents()==m['ordered_parents']
before=git('rev-parse','--is-shallow-repository')
for needed in m['required_immutable_git_objects']:
 if subprocess.run(['git','cat-file','-e',needed],cwd=root,capture_output=True).returncode:
  subprocess.run(['git','fetch','--depth','1','origin',needed],cwd=root,check=True)
 assert subprocess.run(['git','cat-file','-e',needed],cwd=root,capture_output=True).returncode==0
assert not git('status','--porcelain')
data=dict(python=sys.version,platform=platform.platform(),image={key:os.environ.get(key)for key in ('ImageOS','ImageVersion','RUNNER_OS','RUNNER_ARCH')},source_head=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),parents=parents(),shallow_before=before,shallow_after=git('rev-parse','--is-shallow-repository'),required_immutable_objects=m['required_immutable_git_objects'],checkout_note='Depth-one tested-merge checkout plus explicit depth-one immutable source-parent objects. No arbitrary refs, source mutation or release evidence.',manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest())
(out/'environment.json').write_text(json.dumps(data,indent=2)+'\n')
(out/'packages.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
