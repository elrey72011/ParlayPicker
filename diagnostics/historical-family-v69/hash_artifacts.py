from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);root.mkdir(exist_ok=True)
rows=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(root.rglob('*'))if p.is_file()and p.name!='COMPLETE-ARTIFACT-FILE-HASHES.json']
(root/'COMPLETE-ARTIFACT-FILE-HASHES.json').write_text(json.dumps(rows,indent=2)+'\n')
