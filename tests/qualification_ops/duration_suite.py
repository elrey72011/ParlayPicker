"""Additional offline full-envelope and fault-budget measurements."""
from pathlib import Path
import importlib.util
import json
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
i=importlib.util.spec_from_file_location('auth_tests',HERE/'auth_recovery_suite.py');a=importlib.util.module_from_spec(i);i.loader.exec_module(a)
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,required=True);arguments=parser.parse_args()
out=arguments.result
assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
records=[]
with tempfile.TemporaryDirectory(prefix='ParlayPicker-offline-duration-') as temp:
    root=Path(temp)
    fx=a.fixture(root/'full-envelope',40);fx.spec['max_new_objects_per_slice']=5;fx.save_spec()
    m=a.prior.load_driver(a.DRIVER);m.source_check(fx.spec);fx.initialize(m)
    backend=a.AuthWire(fx.files,fx.wire);clock=backend.clock
    verified=m.verified_record
    charged=set()
    def boundary(path):
        value=verified(path)
        key=str(path)
        if key.endswith('-result.json') and 'slice-' in key and key not in charged:
            charged.add(key);clock.advance(2999 if value['slice']==8 else 3000)
        return value
    with a.RealAuthRuntime(backend),patch.object(m,'time',SimpleNamespace(monotonic=clock.monotonic,sleep=clock.advance)),patch.object(m,'verified_record',boundary):
        m.capture_block(fx.spec)
        assert clock.seconds==23999
        meter=m.TransportBudget(fx.root,fx.spec,m.load_state(fx.root,fx.spec)['transport'])
        ctx=m._capture_auth_states[str(fx.root.resolve())]
        factory=m.guarded_session_factory(fx.spec,meter,set(),a.evidence_drive._authorized_session,ctx)
        sessions=[factory() for _ in range(4)]
        try:
            # Concurrent validity check at the end of the full capture envelope.
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda s:s.get('https://www.googleapis.com/drive/v3/files/fixture-folder').raise_for_status(),sessions))
        finally:
            for s in sessions:s.close()
        clock.advance(1);at_capture_end=backend.oauth_posts
        # Assembly + assessment + unused whole-operation allowance: zero HTTP.
        clock.advance(3000)
        assert clock.seconds==27000 and backend.oauth_posts==at_capture_end
        assert backend.oauth_posts==5
    records.append({'test':'full_24000_capture_27000_whole_envelope','status':'PASS','actual_oauth_posts':backend.oauth_posts,'slice_oauth_totals':[verified(fx.root/f'slice-{n:02d}-result.json')['transport']['oauth_attempts'] for n in range(1,9)],'actual_get_attempts':meter['drive_get_attempts'],'simulated_capture_seconds':24000,'simulated_whole_seconds':27000,'credential_constructions':backend.actual_session_constructions,'assumption':'3000s boundary intervals, 3600s tokens, final concurrent expiry probe at23999; no authentication needed in local-only phases','throughput_claim':False})
    fx=a.fixture(root/'fault-budget',160);m=a.prior.load_driver(a.DRIVER);m.source_check(fx.spec);fx.initialize(m)
    backend=a.AuthWire(fx.files,fx.wire);backend.media_seconds=3600
    with a.RealAuthRuntime(backend):
        try:m.capture(fx.spec,1);raise AssertionError('BUDGET_FAILURE_EXPECTED')
        except RuntimeError as exc:assert str(exc)=='OAUTH_REQUEST_LIMIT'
    meter=m.TransportBudget(fx.root,fx.spec,m.load_state(fx.root,fx.spec)['transport'])
    assert meter['oauth_attempts']==20 and backend.oauth_posts==20
    records.append({'test':'expiry_faults_stop_at_20_without_completion_claim','status':'PASS','actual_oauth_posts':20,'counted_oauth_attempts':20,'reason':'OAUTH_REQUEST_LIMIT','accepted_objects':sum(len(m.check_seal(fx.root,ref)['objects']) for ref in m.load_state(fx.root,fx.spec)['accepted_batches']),'actual_get_attempts':meter['drive_get_attempts'],'real_wait_for_expiry':False})
    # Only test-owned temporary files are removed by this context manager.
    assert root.resolve().parent==Path(tempfile.gettempdir()).resolve() and root.name.startswith('ParlayPicker-offline-duration-')
a.write(out,{'driver_sha256':a.sha(a.DRIVER),'test_sha256':a.sha(__file__),'success':True,'tests_run':2,'real_socket_attempts':a.REAL_SOCKET_ATTEMPTS,'results':records,'synthetic_only':True})
print(json.dumps(records,indent=2))
