"""Startup receipt precedes optional imports and package inputs."""
import os,sys
out=sys.argv[1]
os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'startup.json'),'x',encoding='utf-8') as stream:
    stream.write('{"phase":"STARTUP","automatic_retries":0}\n')
try:
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parent
    manifest=(root/'manifest.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest()==os.environ['CHROME_PACKAGE_MANIFEST_SHA256']
    for row in json.loads(manifest):
        path=root/row['path'];data=path.read_bytes()
        assert path.resolve().is_relative_to(root) and len(data)==row['bytes']
        assert hashlib.sha256(data).hexdigest()==row['sha256']
    from probe import run
    from transport_temp import short_transport_tmp
    transport=None
    try:
        with short_transport_tmp() as transport:
            code=run(Path(sys.argv[2]),Path(out),transport_tmp=transport)
    finally:
        if transport is not None:
            with open(os.path.join(out,'transport-temp-cleanup.json'),'w',encoding='utf-8') as stream:
                json.dump(dict(owned_root_removed=not transport.exists()),stream)
except BaseException as error:
    # No raw exception strings: dependencies/inputs can fail before the runner.
    import json
    with open(os.path.join(out,'startup-failure.json'),'w',encoding='utf-8') as stream:
        json.dump(dict(phase='PACKAGE_OR_RUNNER_FAILURE',exception_class=type(error).__name__),stream)
    code=2
sys.exit(code)
