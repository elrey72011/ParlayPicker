from pathlib import Path
import hashlib,json,sys
p=Path(__file__).resolve().parent
raw=(p/'helper-manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==sys.argv[1]
m=json.loads(raw)
for name,value in m['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==value,name
print(json.dumps(dict(helper_manifest_sha256=sys.argv[1],files=len(m['files']),verified=True)))
