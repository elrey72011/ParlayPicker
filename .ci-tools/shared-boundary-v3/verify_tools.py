"""Authenticate prepared diagnostic tools before any application import."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent
raw=(R/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==sys.argv[1]
d=json.loads(raw)
for path,expected in d['files'].items():
 value=(R/path).read_bytes();assert hashlib.sha256(value).hexdigest()==expected, 'DIAGNOSTIC_TOOL_CHANGED'
print(json.dumps(dict(status='PASS_STDLIB_ONLY',files=len(d['files']),manifest_sha256=sys.argv[1],application_imports=0)))

