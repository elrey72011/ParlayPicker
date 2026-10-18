"""Public package versions and exact source identity only; no environment dump."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
lock=Path(__file__).with_name('diagnostic-constraints.txt').read_text().splitlines()
versions={name:importlib.metadata.version(name) for name,_ in (line.split('==') for line in lock)}
assert all(versions[name]==version for name,version in (line.split('==') for line in lock))
assert platform.python_version()=='3.12.15'
record=dict(python=platform.python_version(),pinned_public_distributions=versions,requirements_sha256=hashlib.sha256(Path('requirements.txt').read_bytes()).hexdigest(),
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),
    depth=subprocess.check_output(['git','rev-parse','--is-shallow-repository'],text=True).strip(),
    differences='Diagnostic observer outside checkout; checkout authentication is not persisted; Ubuntu24.04 and Python3.12.15 match the failed job. Default pytest temporary paths, same application tree,full collection and exact selected prefix; no application source is changed.')
args.output.mkdir(parents=True,exist_ok=True);(args.output/'pinned-context.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS exact Python and '+str(len(versions))+' public distribution pins')
