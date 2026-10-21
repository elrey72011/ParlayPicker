"""Authenticated routing for synthetic historical consumers; no runtime authority."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib
import io
import json
import sys
import types
import base64
import zlib
import ast

ROOT = Path(__file__).resolve().parents[1]
BINDING = 'docs/paid-launch/ci-fixture-stability-bindings-v1.json'
GUARD = 'scripts/check_launch_change_scope.py'

class Role(Enum):
    LIVE = 'LIVE'
    NATIVE = 'NATIVE'
    HISTORICAL = 'HISTORICAL'

@dataclass(frozen=True)
class InputContextPlan:
    role: Role
    consumer: str
    source_sha256: str
    node: str
    phase: str
    fixture: str
    generation: str
    binding_sha256: str
    fixture_definition: str = ''
    fixture_consumer: str = ''
    fixture_consumer_sha256: str = ''
    generated_contract: str = ''
    generated_path: str = ''
    generated_parent: str = ''


def physical_bytes(path):
    # Path.read_bytes may delegate to a patched Path.open. Neither is authority.
    with io.FileIO(path, 'r') as stream:
        value = stream.read(8 * 1024 * 1024 + 1)
    if len(value) > 8 * 1024 * 1024:
        raise ValueError('HISTORICAL_PHYSICAL_INPUT_LIMIT')
    return value


def generation_input(plan, relative, supplied=None):
    """Physical source -> exact declared generation, each edge authenticated."""
    authenticate_plan(plan)
    binding = json.loads(physical_bytes(ROOT / BINDING))['historical_input_boundary']
    target = binding['consumers'][plan.consumer].get('source_inputs', {}).get(relative)
    if target is None:
        return None
    source = physical_bytes(ROOT / relative).replace(b'\r\n', b'\n')
    from scripts.ncaaf_reviewed_target_scope import previous_source
    permitted = {sha(source)}
    for path, expected in binding['source_generations'].items():
        raw = physical_bytes(ROOT / path)
        if sha(raw) != expected:
            raise ValueError('HISTORICAL_GENERATION_BINDING_CHANGED')
        document = json.loads(raw)
        for record in (document, document.get('historical_future', {})):
            row = record.get('source_reconstructions', {}).get(relative)
            if row:
                permitted.update((row['before_sha256'], row['after_sha256']))
    if supplied is not None and sha(supplied) not in permitted:
        return supplied  # Keep unrelated tampering visible to the original reader.
    for _ in range(16):
        if sha(source) == target:
            return source
        prior = source
        for path in binding['source_generations']:
            document = json.loads(physical_bytes(ROOT / path))
            for record in (document, document.get('historical_future', {})):
                if 'source_reconstructions' in record:
                    source = previous_source(relative, source, record)
        if source == prior:
            break
    raise ValueError('HISTORICAL_INPUT_GENERATION_CHANGED')


def sha(value):
    return hashlib.sha256(value.replace(b'\r\n', b'\n')).hexdigest()


def historical_helper_input(plan, relative, supplied):
    """An edited helper remains an authenticated input to older readers."""
    authenticate_plan(plan)
    record=json.loads(physical_bytes(ROOT/BINDING))['historical_input_boundary'].get('helper_preimages',{}).get(relative)
    if record is None or sha(supplied)!=record['after_sha256']:
        return supplied
    decoder=zlib.decompressobj()
    value=decoder.decompress(base64.b85decode(record['compressed']),record['bytes']+1)
    blob=hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()
    if (not decoder.eof or decoder.unused_data or len(value)!=record['bytes']
        or len(value)>8*1024*1024 or sha(value)!=record['before_sha256']
        or blob!=record['before_git_blob']):
        raise ValueError('HISTORICAL_HELPER_PREIMAGE_CHANGED')
    return value


def native_context(name, node):
    if name == 'test_ci_fixture_stability_scope':
        return node not in {
            'test_single_advisory_amendment_preserves_all_other_rows_and_metadata',
            'test_second_advisory_amendment_retains_exact_single_entry_predecessor',
            'test_second_advisory_preimage_is_narrow_authenticated_and_restored',
            'test_complete_profile_refresh_preserves_all_prior_rows_and_measurements',
            'test_complete_profile_views_are_exact_reject_alteration_and_restore_current_reads'}
    return (name, node) in {
        ('test_ci_five_shards', 'test_five_shard_gate_uploads_and_reconciles_all_five'),
        ('test_ci_six_shards', 'test_six_way_current_gate_preserves_limits_and_all_artifacts'),
        ('test_ci_test_shards', 'test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate'),
        ('test_ncaaf_staged_ci_capacity', 'test_current_nine_way_configuration_keeps_original_limits'),
        ('test_ncaaf_transport_scope', 'test_external_fixture_children_keep_their_actual_live_context'),
        ('test_ci_scheduling_fixture_scope', 'test_unchanged_native_context_assertion_receives_no_projection')}


def fixture_alias_identity(binding,constructor,definition,alias,scope,consumer):
    rows=binding['historical_input_boundary'].get('fixture_aliases',[])
    if len(rows)>32:raise ValueError('HISTORICAL_FIXTURE_ALIAS_LIMIT')
    matches=[r for r in rows if (r['constructor'],r['fixture'],r['alias'],r['scope'],r['consumer'])
             ==(constructor,definition,alias,scope,consumer)]
    if len(matches)!=1:raise ValueError('HISTORICAL_FIXTURE_ALIAS_NOT_AUTHENTICATED')
    record=matches[0];source=physical_bytes(ROOT/consumer)
    if sha(source)!=record['consumer_sha256']:
        raise ValueError('HISTORICAL_FIXTURE_ALIAS_SOURCE_CHANGED')
    # Authenticate the actual import site, not a mutable alias object's identity.
    found=any(isinstance(n,ast.ImportFrom) and n.lineno==record['import_line']
        and n.module is not None and 'tests/'+n.module.removeprefix('tests.').replace('.','/')+'.py'==constructor
        and any(a.name==definition and a.asname==alias for a in n.names)
        for n in ast.walk(ast.parse(source)))
    if not found:raise ValueError('HISTORICAL_FIXTURE_ALIAS_NOT_AUTHENTICATED')
    return record['consumer_sha256']


def _legacy_request_context(request, caller, api):
    # A legacy descriptor is data; authority is the physical API/call site.
    try:
        api_path=Path(api.f_code.co_filename).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    raw=physical_bytes(ROOT/BINDING)
    data=json.loads(raw)
    rows=data.get('historical_input_boundary',{}).get('legacy_request_calls',[])
    if not isinstance(rows,list) or len(rows)>32:
        raise ValueError('HISTORICAL_API_CALLSITE_LIMIT')
    candidates=[r for r in rows if isinstance(r,dict)
        and r.get('api_path')==api_path and r.get('api_function')==api.f_code.co_name]
    if not candidates:
        return None
    scope,checked=canonical_guard()._ci_fixture_module()
    if scope.canonical(checked)!=raw.replace(b'\r\n',b'\n'):
        raise ValueError('HISTORICAL_BINDING_CHANGED')
    consumers=checked['historical_input_boundary']['consumers']
    for row in candidates:
        if set(row)!=set('api_path api_function api_first_line caller_path caller_function caller_first_line call_line targets'.split()):
            raise ValueError('HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED')
        api_record=consumers.get(row['api_path']) or checked['historical_input_boundary'].get('helper_preimages',{}).get(row['api_path'])
        if (api_record is None
            or sha(physical_bytes(ROOT/row['api_path']))!=api_record.get('sha256',api_record.get('after_sha256'))):
            raise ValueError('HISTORICAL_API_SOURCE_CHANGED')
        if (row['caller_path'] not in consumers
            or sha(physical_bytes(ROOT/row['caller_path']))!=consumers[row['caller_path']]['sha256']):
            raise ValueError('HISTORICAL_API_CONSUMER_SOURCE_CHANGED')
        if not isinstance(row['targets'],dict) or not 0<len(row['targets'])<=8:
            raise ValueError('HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED')
        if not all(isinstance(k,str) and 0<len(k)<=256 and v in consumers for k,v in row['targets'].items()):
            raise ValueError('HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED')
    try:
        caller_path=Path(caller.f_code.co_filename).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError('HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED') from None
    matches=[r for r in candidates if (r['caller_path'],r['caller_function'],r['caller_first_line'],r['call_line'],r['api_first_line'])
        ==(caller_path,caller.f_code.co_name,caller.f_code.co_firstlineno,caller.f_lineno,api.f_code.co_firstlineno)]
    if len(matches)!=1:
        raise ValueError('HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED')
    row=matches[0]
    source=physical_bytes(ROOT/caller_path)
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
        and n.name==row['caller_function']
        and min([n.lineno]+[d.lineno for d in n.decorator_list])==row['caller_first_line']]
    if len(functions)!=1 or not any(isinstance(n,ast.Call) and n.lineno==row['call_line']
        and isinstance(n.func,ast.Attribute) and n.func.attr==row['api_function'] for n in ast.walk(functions[0])):
        raise ValueError('HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED')
    descriptor=getattr(request.module,'__name__',None)
    target=row['targets'].get(descriptor)
    if target is None:
        raise ValueError('HISTORICAL_API_CALLSITE_NOT_AUTHENTICATED')
    proxy=types.SimpleNamespace(module=types.SimpleNamespace(__name__=descriptor,__file__=str(ROOT/target)),
        node=getattr(request,'node',None),function=getattr(request,'function',None))
    return resolve_context(proxy,historical=True)



def capture_generated_parent(config):
    # An inherited node is only an origin hint. All physical sources, exact
    # literal programs, fixture site and binding are authenticated before use.
    import os
    marker=os.environ.get('PYTEST_CURRENT_TEST','')
    rows=json.loads(physical_bytes(ROOT/BINDING)).get('historical_input_boundary',{}).get('generated_fixture_contracts',[])
    if not isinstance(rows,list) or len(rows)>8:
        raise ValueError('HISTORICAL_GENERATED_CONTRACT_LIMIT')
    matches=[r['parent']+'::'+r['function'] for r in rows
             if marker==r['parent']+'::'+r['function']+' (call)']
    config._historical_generated_parent=matches[0] if len(matches)==1 else ''


def _generated_record(data,contract,path,parent_node):
    # Registry paths are usable only after the current guard authenticates the
    # complete physical binding. Source labels or an inherited node are no grant.
    scope,checked=canonical_guard()._ci_fixture_module()
    if scope.canonical(checked)!=physical_bytes(ROOT/BINDING).replace(b'\r\n',b'\n') or checked!=data:
        raise ValueError('HISTORICAL_BINDING_CHANGED')
    rows=data.get('historical_input_boundary',{}).get('generated_fixture_contracts',[])
    if not isinstance(rows,list) or len(rows)>8:
        raise ValueError('HISTORICAL_GENERATED_CONTRACT_LIMIT')
    matches=[r for r in rows if isinstance(r,dict) and r.get('id')==contract]
    if len(matches)!=1:raise ValueError('HISTORICAL_GENERATED_NOT_AUTHENTICATED')
    row=matches[0]
    if set(row)!=set('id parent parent_sha256 function line directory conftest_sha256 programs fixture generation input_sha256'.split()):
        raise ValueError('HISTORICAL_GENERATED_NOT_AUTHENTICATED')
    if parent_node!=row['parent']+'::'+row['function']:
        raise ValueError('HISTORICAL_GENERATED_ORIGIN_CHANGED')
    parent_record=data['historical_input_boundary']['consumers'].get(row['parent'])
    if parent_record is None or parent_record['sha256']!=row['parent_sha256']:
        raise ValueError('HISTORICAL_GENERATED_PARENT_CHANGED')
    raw=physical_bytes(ROOT/row['parent'])
    if sha(raw)!=row['parent_sha256']:
        raise ValueError('HISTORICAL_GENERATED_PARENT_CHANGED')
    functions=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)
               and n.name==row['function'] and n.lineno==row['line']]
    if len(functions)!=1 or not isinstance(row['programs'],dict) or len(row['programs'])!=2:
        raise ValueError('HISTORICAL_GENERATED_CALLSITE_CHANGED')
    import textwrap
    literals={}
    for call in ast.walk(functions[0]):
        if not(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='write_text'):continue
        target=call.func.value
        if not(isinstance(target,ast.BinOp) and isinstance(target.op,ast.Div) and isinstance(target.right,ast.Constant)):continue
        value=call.args[0]
        if not(isinstance(value,ast.Call) and ast.unparse(value.func)=='textwrap.dedent'):continue
        program=textwrap.dedent(ast.literal_eval(value.args[0])).encode()
        literals[target.right.value]={'sha256':sha(program),'write_line':call.lineno}
    if literals!=row['programs']:
        raise ValueError('HISTORICAL_GENERATED_CALLSITE_CHANGED')
    path=Path(path).resolve();directory=path.parent
    if directory.name!=row['directory'] or path.name not in row['programs']:
        raise ValueError('HISTORICAL_GENERATED_ORIGIN_CHANGED')
    for name,expected in row['programs'].items():
        if Path(name).name!=name or set(expected)!={'sha256','write_line'} or sha(physical_bytes(directory/name))!=expected['sha256']:
            raise ValueError('HISTORICAL_GENERATED_PROGRAM_CHANGED')
    if (sha(physical_bytes(ROOT/'tests/conftest.py'))!=row['conftest_sha256']
        or sha(physical_bytes(directory/'conftest.py'))!=row['conftest_sha256']):
        raise ValueError('HISTORICAL_GENERATED_CONFTEST_CHANGED')
    site=row['fixture']
    if set(site)!={'program','name','scope','line'} or site['scope']!='session' or site['program'] not in row['programs']:
        raise ValueError('HISTORICAL_GENERATED_FIXTURE_CHANGED')
    definitions=[n for n in ast.parse(physical_bytes(directory/site['program'])).body
                 if isinstance(n,ast.FunctionDef) and n.name==site['name']]
    if len(definitions)!=1 or min([definitions[0].lineno]+[d.lineno for d in definitions[0].decorator_list])!=site['line']:
        raise ValueError('HISTORICAL_GENERATED_FIXTURE_CHANGED')
    declared=json.loads(physical_bytes(ROOT/'docs/paid-launch/ncaaf-future-spread-binding-v1.json'))
    if row['input_sha256']!=declared['previous_guard_sha256'] or row['generation']!='ncaaf-future-source':
        raise ValueError('HISTORICAL_GENERATED_GENERATION_CHANGED')
    return row


def _generated_plan(request,path,phase,fixture,historical):
    origin=getattr(getattr(request,'config',None),'_historical_generated_parent','')
    raw=physical_bytes(ROOT/BINDING);data=json.loads(raw)
    rows=data.get('historical_input_boundary',{}).get('generated_fixture_contracts',[])
    if not isinstance(rows,list) or len(rows)>8:raise ValueError('HISTORICAL_GENERATED_CONTRACT_LIMIT')
    candidates=[r for r in rows if isinstance(r,dict) and origin==r.get('parent','')+'::'+r.get('function','')
                and path.parent.name==r.get('directory') and path.name in r.get('programs',{})]
    if not candidates:return None
    if len(candidates)!=1:raise ValueError('HISTORICAL_GENERATED_NOT_AUTHENTICATED')
    row=_generated_record(data,candidates[0]['id'],path,origin);site=row['fixture']
    definition=''
    if phase=='session':
        function=getattr(request,'function',None);code=getattr(function,'__code__',None)
        if (path.name!=site['program'] or fixture!=site['name'] or code is None
            or Path(code.co_filename).resolve()!=path or code.co_name!=site['name'] or code.co_firstlineno!=site['line']):
            raise ValueError('HISTORICAL_GENERATED_FIXTURE_CHANGED')
        definition=site['name'];role=Role.HISTORICAL;generation=row['generation'];node=definition
    elif phase=='call' and not fixture:
        role=Role.NATIVE;generation='current'
        node=(getattr(getattr(request,'node',None),'originalname',None)
              or getattr(getattr(request,'node',None),'name','').split('[',1)[0]
              or getattr(getattr(request,'function',None),'__name__',''))
        tests={n.name for n in ast.parse(physical_bytes(path)).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
        if node not in tests:raise ValueError('HISTORICAL_GENERATED_CALLSITE_CHANGED')
    else:raise ValueError('HISTORICAL_GENERATED_FIXTURE_CHANGED')
    if historical and role is not Role.HISTORICAL:raise ValueError('HISTORICAL_CONSUMER_NOT_AUTHENTICATED')
    plan=InputContextPlan(role,row['parent'],sha(physical_bytes(path)),node,phase,fixture,generation,sha(raw),definition,
                          generated_contract=row['id'],generated_path=str(path),generated_parent=origin)
    authenticate_plan(plan)
    return plan


def _authenticate_generated(plan,data):
    row=_generated_record(data,plan.generated_contract,plan.generated_path,plan.generated_parent)
    site=row['fixture'];path=Path(plan.generated_path)
    if plan.consumer!=row['parent'] or plan.source_sha256!=sha(physical_bytes(path)):
        raise ValueError('HISTORICAL_GENERATED_PROGRAM_CHANGED')
    if plan.fixture_consumer or plan.fixture_consumer_sha256:
        raise ValueError('HISTORICAL_GENERATED_FIXTURE_CHANGED')
    if plan.phase=='session':
        if (path.name!=site['program'] or plan.fixture!=site['name'] or plan.fixture_definition!=site['name']
            or plan.node!=site['name'] or plan.role is not Role.HISTORICAL or plan.generation!=row['generation']):
            raise ValueError('HISTORICAL_GENERATED_FIXTURE_CHANGED')
    elif plan.phase=='call':
        tests={n.name for n in ast.parse(physical_bytes(path)).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
        if plan.role is not Role.NATIVE or plan.generation!='current' or plan.fixture or plan.fixture_definition or plan.node not in tests:
            raise ValueError('HISTORICAL_GENERATED_CALLSITE_CHANGED')
    else:raise ValueError('HISTORICAL_GENERATED_FIXTURE_CHANGED')
    return row


def _install_generated_source_input(plan,patch,live):
    # Only an immutable input snapshot is historical. The actual guard namespace
    # remains native; neither a module lease nor predecessor cache is created.
    data=json.loads(physical_bytes(ROOT/BINDING));row=_authenticate_generated(plan,data)
    scope,binding=live._ci_fixture_module()
    physical=physical_bytes(ROOT/GUARD).replace(b'\r\n',b'\n')
    value=live._ncaaf_future_base_source(scope.previous_guard(physical,binding))
    if sha(value)!=row['input_sha256']:raise ValueError('HISTORICAL_GENERATED_GENERATION_CHANGED')
    opened=Path.open;target=(ROOT/GUARD).resolve()
    def exact_input(path,mode='r',*args,**kwargs):
        if mode!='rb' or path.resolve()!=target:return opened(path,mode,*args,**kwargs)
        with opened(path,'rb') as stream:supplied=stream.read()
        if sha(supplied)==sha(physical):return io.BytesIO(value)
        if sha(supplied)==row['input_sha256']:return io.BytesIO(supplied)
        raise ValueError('HISTORICAL_GENERATED_SOURCE_INPUT_CHANGED')
    patch.setattr(Path,'open',exact_input)


def resolve_context(request, *, phase='call', fixture='', historical=False):
    module = request.module
    try:
        path = Path(module.__file__).resolve()
        relative = path.relative_to(ROOT).as_posix()
    except (ValueError, AttributeError):
        if getattr(module,'__file__',None) is not None:
            generated=_generated_plan(request,Path(module.__file__).resolve(),phase,fixture,historical)
            if generated is not None:return generated
        if getattr(module,'__file__',None) is None and phase=='call' and not fixture:
            api=sys._getframe(1)
            legacy=_legacy_request_context(request,api.f_back,api)
            if legacy is not None:return legacy
        if historical:
            raise ValueError('HISTORICAL_CONSUMER_NOT_AUTHENTICATED')
        return InputContextPlan(Role.LIVE, '', '', '', phase, fixture, 'current', '')
    node = (getattr(getattr(request, 'node', None), 'originalname', None)
            or getattr(getattr(request, 'node', None), 'name', '').split('[', 1)[0]
            or getattr(getattr(request, 'function', None), '__name__', ''))
    # Registry is inert data. No predecessor import, reconstruction or cache is
    # consulted to decide that an unregistered (including paid) consumer is live.
    binding_raw = physical_bytes(ROOT / BINDING)
    binding = json.loads(binding_raw)
    record = binding.get('historical_input_boundary', {}).get('consumers', {}).get(relative)
    if record is None:
        if historical:
            raise ValueError('HISTORICAL_CONSUMER_NOT_AUTHENTICATED')
        return InputContextPlan(Role.LIVE, relative, sha(physical_bytes(path)), node, phase, fixture, 'current', sha(binding_raw))
    if sha(physical_bytes(path)) != record['sha256']:
        raise ValueError('HISTORICAL_CONSUMER_SOURCE_CHANGED')
    definition=caller=caller_sha=''
    if phase in {'session', 'module'}:
        function = getattr(request, 'function', None)
        definition=getattr(function,'__name__','')
        site = record.get('fixtures', {}).get(definition)
        code = getattr(function, '__code__', None)
        if (site is None or site['scope'] != phase or code is None
            or Path(code.co_filename).resolve() != path
            or code.co_firstlineno != site['line']):
            raise ValueError('HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED')
        if fixture!=definition:
            supplied=getattr(request,'fixture_consumer_path',None)
            consumer_function=getattr(request,'fixture_consumer_function',None)
            consumer_code=getattr(consumer_function,'__code__',None)
            if supplied is None or consumer_code is None or Path(consumer_code.co_filename).resolve()!=Path(supplied).resolve():
                raise ValueError('HISTORICAL_FIXTURE_ALIAS_NOT_AUTHENTICATED')
            try:caller=Path(supplied).resolve().relative_to(ROOT).as_posix()
            except ValueError:raise ValueError('HISTORICAL_FIXTURE_ALIAS_NOT_AUTHENTICATED') from None
            caller_sha=fixture_alias_identity(binding,relative,definition,fixture,phase,caller)
    name = path.stem
    role = Role.NATIVE if native_context(name, node) else Role.HISTORICAL
    generation = record['generation'] if role is Role.HISTORICAL else 'current'
    plan = InputContextPlan(role, relative, record['sha256'], node, phase, fixture, generation, sha(binding_raw),definition,caller,caller_sha)
    if role is Role.HISTORICAL:
        selected_reader_generation(binding, plan)
    authenticate_plan(plan)
    return plan


def authenticate_plan(plan):
    binding_raw=physical_bytes(ROOT/BINDING)
    if sha(binding_raw) != plan.binding_sha256:
        raise ValueError('HISTORICAL_BINDING_CHANGED')
    data=json.loads(binding_raw)
    if plan.generated_contract:
        _authenticate_generated(plan,data)
    else:
        if plan.generated_path or plan.generated_parent:raise ValueError('HISTORICAL_GENERATED_NOT_AUTHENTICATED')
        record = data.get('historical_input_boundary', {}).get('consumers', {}).get(plan.consumer)
        if record is None:
            raise ValueError('HISTORICAL_CONSUMER_NOT_AUTHENTICATED')
        if plan.source_sha256 != record['sha256'] or sha(physical_bytes(ROOT/plan.consumer))!=record['sha256']:
            raise ValueError('HISTORICAL_CONSUMER_SOURCE_CHANGED')
        if plan.phase not in {'call', 'session', 'module'} or (plan.phase == 'call' and plan.fixture):
            raise ValueError('HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED')
        expected_role = Role.NATIVE if native_context(Path(plan.consumer).stem, plan.node) else Role.HISTORICAL
        expected_generation = record['generation'] if expected_role is Role.HISTORICAL else 'current'
        if plan.role is not expected_role or plan.generation != expected_generation:
            raise ValueError('HISTORICAL_CONTEXT_GENERATION_CHANGED')
        if plan.phase in {'module','session'}:
            if record.get('fixtures',{}).get(plan.fixture_definition,{}).get('scope')!=plan.phase:
                raise ValueError('HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED')
            if plan.fixture!=plan.fixture_definition:
                expected=fixture_alias_identity(data,plan.consumer,plan.fixture_definition,plan.fixture,plan.phase,plan.fixture_consumer)
                if expected!=plan.fixture_consumer_sha256:raise ValueError('HISTORICAL_FIXTURE_ALIAS_SOURCE_CHANGED')
            elif plan.fixture_consumer or plan.fixture_consumer_sha256:
                raise ValueError('HISTORICAL_FIXTURE_ALIAS_NOT_AUTHENTICATED')
        elif plan.fixture_definition or plan.fixture_consumer or plan.fixture_consumer_sha256:
            raise ValueError('HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED')
    # Authentication reads real guard/binding/scope bytes, never active views.
    live = canonical_guard()
    loader = getattr(live, '_ci_fixture_module', None)
    if loader is None:
        raise ValueError('HISTORICAL_CURRENT_GUARD_UNAVAILABLE')
    scope, binding = loader()
    if sha(scope.canonical(binding)) != plan.binding_sha256:
        raise ValueError('HISTORICAL_BINDING_CHANGED')
    if plan.role is Role.HISTORICAL and not plan.generated_contract:
        if binding.get('private_delivery_repair') != scope.PRIVATE_DELIVERY_REPAIR:
            raise ValueError('HISTORICAL_READER_BINDING_CHANGED')
        selected_reader_generation(binding, plan)
    for relative, record in binding['historical_input_boundary'].get('helper_preimages',{}).items():
        if sha(physical_bytes(ROOT/relative))!=record['after_sha256']:
            raise ValueError('HISTORICAL_HELPER_SOURCE_CHANGED')
    authenticated_predecessor(scope, binding)
    return live


_ACTIVE = []
_LIVE_GUARD = None
_ABSENT = object()
_PREDECESSOR_IMAGE = None

# Exact generations used by the released guard's five import-time readers.
# These are byte transformations, never cached validation or module objects.
_BOOTSTRAP_SOURCE = '3720e55ec37baae7dcf53d5eb3b3436fcc69ae41df820a110900965945adb306'
_BOOTSTRAP_IMAGES = None
_BOOTSTRAP_PLAN = None
_BOOTSTRAP_NATIVE_COMPILE = compile
_BOOTSTRAP_LIMIT = 32 * 1024 * 1024
_BOOTSTRAP_GENERATIONS = (
    ('_ncaaf_custody_predecessor', '_ncaaf_custody_parent_guard_source', '_ncaaf_pilot_parent_guard_source',
     '35ac29313e9509efd68894a86e4a98b109a58459605fe589cc6f86159e0dbc64', '0772744a3b07e7a3aa67d889efff45743e4562e4fe8b781e2d2b3d5111424d9d', 1),
    ('_ncaaf_pilot_predecessor', '_ncaaf_pilot_parent_guard_source', '_discovery_parent_guard_source',
     '6e82f1a4e0e14a34d62b3ead8eba8f44f382e40093dcfec447029d22027a219c', '35ac29313e9509efd68894a86e4a98b109a58459605fe589cc6f86159e0dbc64', 1),
    ('_discovery_predecessor', '_discovery_parent_guard_source', '_cloud_peel_exact_guard',
     '8df5b4b619e220d2884f8f395562ab356d41e247e3729a72774d4e0b79cc4d00', '6e82f1a4e0e14a34d62b3ead8eba8f44f382e40093dcfec447029d22027a219c', 2),
    ('_cloud_predecessor', '_cloud_peel_exact_guard', '_owner_base_source',
     '1639ccad41c0943799b30aaf6f8c0f08c08aaef96293408d147a919537a6406b', '8df5b4b619e220d2884f8f395562ab356d41e247e3729a72774d4e0b79cc4d00', 1),
    ('_owner_predecessor', '_owner_base_source', '_nfl_owner_base_source',
     '0ab2edd3f3a246fde7aa25c81a6e00bceb4a41621f2fb2ea276392725e29cb6b', '1639ccad41c0943799b30aaf6f8c0f08c08aaef96293408d147a919537a6406b', 1),
)


_RELEASED_READER_GENERATION = 'released-future-v1'
_LEGACY_READER_SOURCES = {
    'legacy-drive-after-v1': {
        'version': 'authenticated-historical-reader-source-v1',
        'source_generation': 'legacy-drive-after-v1',
        'sha256': '2116a73036a6243da92120a29c5a4dcf242d21fa735e31beeeceedd0174ef386',
        'git_blob': '30d04ccf7764a9f07bf9b8c98d3dd8a6556976b4',
        'bytes': 170111,
        'source_provenance': {
            'parent_source_sha256': _BOOTSTRAP_SOURCE,
            'original_binding': 'ESTIMATE_BINDINGS.previous_guard_sha256',
            'original_reader': '_estimate_previous_guard_source',
            'expected_hash_preserved': True,
        },
    },
}


def selected_reader_generation(binding, plan=None):
    """Select inert, exact reader authority before a view or constructor exists."""
    if plan is None:
        if not _ACTIVE:
            raise ValueError('HISTORICAL_CONTEXT_NOT_OWNED')
        plan = _ACTIVE[-1].plan
    if plan.role is not Role.HISTORICAL or plan.generated_contract:
        raise ValueError('HISTORICAL_READER_CONTEXT_CHANGED')
    record = binding['historical_input_boundary']['consumers'].get(plan.consumer)
    declared = binding['private_delivery_repair'].get('reader_generation_contracts', {})
    key = record.get('guard_source_generation') if record is not None else None
    if (key not in {_RELEASED_READER_GENERATION, *_LEGACY_READER_SOURCES}
            or declared.get(plan.consumer) != key):
        raise ValueError('HISTORICAL_READER_GENERATION_CHANGED')
    return key


def decode_reader_source(row):
    """Decode a complete code preimage; this confers no source-use authority."""
    if not isinstance(row, dict):
        raise ValueError('HISTORICAL_READER_SOURCE_CHANGED')
    key = row.get('source_generation')
    expected = _LEGACY_READER_SOURCES.get(key)
    if (expected is None or set(row) != set(expected) | {'compressed_b85'}
            or {k: v for k, v in row.items() if k != 'compressed_b85'} != expected
            or not isinstance(row['compressed_b85'], str)
            or len(row['compressed_b85']) > 1024 * 1024):
        raise ValueError('HISTORICAL_READER_SOURCE_CHANGED')
    try:
        decoder = zlib.decompressobj()
        value = decoder.decompress(base64.b85decode(row['compressed_b85']), expected['bytes'] + 1)
    except (ValueError, zlib.error):
        raise ValueError('HISTORICAL_READER_SOURCE_CHANGED') from None
    blob = hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()
    if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
            or len(value) != expected['bytes'] or sha(value) != expected['sha256']
            or blob != expected['git_blob']):
        raise ValueError('HISTORICAL_READER_SOURCE_CHANGED')
    return value


def historical_reader_source(live, binding, key):
    if not _ACTIVE or selected_reader_generation(binding) != key:
        raise ValueError('HISTORICAL_CONTEXT_NOT_OWNED')
    plan = _ACTIVE[-1].plan
    authenticate_plan(plan)
    scope, current = live._ci_fixture_module()
    if scope.canonical(current) != scope.canonical(binding):
        raise ValueError('HISTORICAL_READER_BINDING_CHANGED')
    rows = current.get('historical_reader_sources', {})
    if set(rows) != set(_LEGACY_READER_SOURCES) or key not in rows:
        raise ValueError('HISTORICAL_READER_SOURCE_CHANGED')
    return decode_reader_source(rows[key])


def _run_bootstrap_assignment(module, code, reader, source, image, expected):
    """Only this assignment owns a generation adapter; definitions run normally."""
    original = getattr(module, reader)
    def normalized(value, *args, **kwargs):
        selected = value == source
        result = original(image if selected else value, *args, **kwargs)
        if selected and (type(result) is not bytes or sha(result) != expected):
            raise ValueError('HISTORICAL_BOOTSTRAP_OUTPUT_CHANGED')
        return result
    setattr(module, reader, normalized)
    try:
        exec(code, module.__dict__)
    finally:
        # No definitions execute in this interval. Unexpected replacement is a
        # conflict, not permission to overwrite somebody else's patch.
        owned = getattr(module, reader) is normalized
        if owned:
            setattr(module, reader, original)
        else:
            raise ValueError('HISTORICAL_BOOTSTRAP_READER_OWNER_CHANGED')


def _bootstrap_immutable_size(value):
    """Bound the complete retained byte/code graph; never retain a namespace."""
    pending=[value];seen=set();size=0
    while pending:
        item=pending.pop()
        if id(item) in seen:continue
        seen.add(id(item));size+=sys.getsizeof(item)
        if size>_BOOTSTRAP_LIMIT or len(seen)>128000:return _BOOTSTRAP_LIMIT+1
        if isinstance(item,types.CodeType):
            pending.extend((item.co_code,item.co_consts,item.co_names,item.co_varnames,
                            item.co_filename,item.co_linetable,item.co_exceptiontable,
                            item.co_freevars,item.co_cellvars))
        elif type(item) in (tuple,frozenset):pending.extend(item)
        elif item is _BOOTSTRAP_NATIVE_COMPILE:pass
        elif type(item) not in (str,bytes,int,float,complex,bool,type(None),type(Ellipsis)):
            return _BOOTSTRAP_LIMIT+1
    return size


def _bootstrap_code_digest(value):
    """Content digest independent of marshal reference counts/interning."""
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,types.CodeType):
            digest.update(b'C')
            visit(tuple(getattr(item,name) for name in (
                'co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals',
                'co_stacksize','co_flags','co_code','co_consts','co_names','co_varnames',
                'co_filename','co_name','co_qualname','co_firstlineno','co_linetable',
                'co_exceptiontable','co_freevars','co_cellvars')))
        elif type(item)is tuple:
            digest.update(b'T'+str(len(item)).encode()+b':')
            for part in item:visit(part)
        elif type(item)is frozenset:
            visit(('frozenset',tuple(sorted(_bootstrap_code_digest(part)for part in item))))
        elif type(item)in(str,bytes,int,float,complex,bool,type(None),type(Ellipsis)):
            raw=item if type(item)is bytes else repr(item).encode()
            digest.update(type(item).__name__.encode()+b':'+str(len(raw)).encode()+b':'+raw)
        else:raise ValueError('HISTORICAL_BOOTSTRAP_PLAN_TYPE_CHANGED')
    visit(value)
    return digest.hexdigest()


def execute_released_bootstrap(source, module, live, binding):
    """Authenticate physical inputs, then construct fresh historical namespaces.

    The old constructor read a later 3.4-MB generation at every bootstrap and
    repeatedly peeled unrelated successor layers. Give each original reader its
    exact input generation during its assignment, with unchanged reconstruction
    and output assertions. No mutable historical namespace is reused.
    """
    global _BOOTSTRAP_IMAGES,_BOOTSTRAP_PLAN
    if type(source) is not bytes or sha(source) != _BOOTSTRAP_SOURCE:
        raise ValueError('HISTORICAL_BOOTSTRAP_GENERATION_CHANGED')
    scope, current = live._ci_fixture_module()
    if scope.canonical(current) != scope.canonical(binding):
        raise ValueError('HISTORICAL_BOOTSTRAP_BINDING_CHANGED')
    declaration=current['private_delivery_repair']
    if (declaration['bootstrap_source_sha256']!=_BOOTSTRAP_SOURCE
        or tuple(tuple(row)for row in declaration['bootstrap_generations'])!=_BOOTSTRAP_GENERATIONS):
        raise ValueError('HISTORICAL_BOOTSTRAP_GENERATION_CHANGED')
    physical = physical_bytes(ROOT / GUARD).replace(b'\r\n', b'\n')
    scope.previous_guard(physical, current)  # Reauthenticate even on a warm hit.
    helper = physical_bytes(ROOT / 'tests/ncaaf_future_scope_inputs.py').replace(b'\r\n', b'\n')
    if scope.blob(helper) != current['reviewed_blobs']['tests/ncaaf_future_scope_inputs.py']:
        raise ValueError('HISTORICAL_BOOTSTRAP_HELPER_CHANGED')
    boundary = physical_bytes(ROOT / 'tests/historical_scope_input_boundary.py').replace(b'\r\n', b'\n')
    if scope.blob(boundary) != current['reviewed_blobs']['tests/historical_scope_input_boundary.py']:
        raise ValueError('HISTORICAL_BOOTSTRAP_HELPER_CHANGED')
    key = (physical, scope.canonical(current), source, helper,
           physical_bytes(ROOT / 'scripts/ci_fixture_stability_scope.py').replace(b'\r\n', b'\n'),
           boundary)
    plan_suffix=(module.__file__.encode(),sys.implementation.cache_tag.encode(),
                 sys.version.encode(),str(sys.flags.optimize).encode())
    plan_key=key+plan_suffix
    if (_BOOTSTRAP_PLAN is not None and (_BOOTSTRAP_PLAN[0]!=plan_key
        or _BOOTSTRAP_PLAN[1] is not compile)):
        _BOOTSTRAP_PLAN=None
    previous = _BOOTSTRAP_IMAGES
    if previous is not None and previous[0] == key:
        # Authentication and full-byte comparison already ran. Retain one
        # immutable copy of identical inputs, not duplicate warm-read buffers.
        key=previous[0]
        images = previous[1]
    else:
        images = tuple(getattr(live, row[2])(source) for row in _BOOTSTRAP_GENERATIONS)
    plan_key=key+plan_suffix
    if (type(images) is not tuple or len(images) != len(_BOOTSTRAP_GENERATIONS)
        or any(type(image) is not bytes or sha(image) != row[3]
               for row, image in zip(_BOOTSTRAP_GENERATIONS, images))):
        raise ValueError('HISTORICAL_BOOTSTRAP_IMAGE_CHANGED')
    image_size=_bootstrap_immutable_size((key,images))
    if image_size <= _BOOTSTRAP_LIMIT:
        _BOOTSTRAP_IMAGES = (key, images)
    else:
        _BOOTSTRAP_IMAGES=None;_BOOTSTRAP_PLAN=None
    cached=_BOOTSTRAP_PLAN
    if cached is not None:
        chunks,tail=cached[2:4]
        actual_size=_bootstrap_immutable_size((_BOOTSTRAP_IMAGES,cached))
        actual_digest=_bootstrap_code_digest((chunks,tail))
        if actual_size>_BOOTSTRAP_LIMIT or actual_digest!=cached[4]:
            raise ValueError('HISTORICAL_BOOTSTRAP_PLAN_CHANGED',actual_size,_BOOTSTRAP_LIMIT,cached[4],actual_digest)
        for prefix,assignment,index,row in chunks:
            exec(prefix,module.__dict__)
            _run_bootstrap_assignment(module,assignment,row[1],source,images[index],row[4])
        exec(tail,module.__dict__)
        return
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    rows = {row[0]: (index, row) for index, row in enumerate(_BOOTSTRAP_GENERATIONS)}
    steps = []
    counts = {name: 0 for name in rows}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id in rows):
            name = node.value.func.id
            counts[name] += 1
            steps.append((offsets[node.lineno-1], offsets[node.end_lineno], *rows[name]))
    if any(counts[row[0]] != row[5] for row in _BOOTSTRAP_GENERATIONS):
        raise ValueError('HISTORICAL_BOOTSTRAP_SEQUENCE_CHANGED')
    # Compile all unchanged statements first. Padding preserves function line
    # identities; no frozen source text, assertion or future clock is rewritten.
    chunks = []
    cursor = 0
    for start, end, index, row in steps:
        prefix = b'\n' * source[:cursor].count(b'\n') + source[cursor:start]
        assignment = b'\n' * source[:start].count(b'\n') + source[start:end]
        chunks.append((compile(prefix, module.__file__, 'exec'),
                       compile(assignment, module.__file__, 'exec'), index, row))
        cursor = end
    tail = compile(b'\n' * source[:cursor].count(b'\n') + source[cursor:], module.__file__, 'exec')
    chunks=tuple(chunks)
    digest=_bootstrap_code_digest((chunks,tail))
    entry=(plan_key,compile,chunks,tail,digest,0)
    size=_bootstrap_immutable_size((_BOOTSTRAP_IMAGES,entry))+128
    if (size<=_BOOTSTRAP_LIMIT and compile is _BOOTSTRAP_NATIVE_COMPILE
        and isinstance(compile,types.BuiltinFunctionType)):
        # Code objects and their constants are immutable. Every use still
        # authenticates physical inputs and executes into new dictionaries.
        _BOOTSTRAP_PLAN=(plan_key,compile,chunks,tail,digest,size)
    for prefix, assignment, index, row in chunks:
        exec(prefix, module.__dict__)
        _run_bootstrap_assignment(module, assignment, row[1], source, images[index], row[4])
    exec(tail, module.__dict__)


def authenticated_predecessor(scope, binding):
    """Reuse one immutable transformation, never a validation or module verdict.

    The caller obtains a freshly authenticated scope/binding. Both physical
    inputs and the complete generation descriptor key the bytes. previous_guard
    still validates the current successor, binding and returned image each time.
    """
    global _PREDECESSOR_IMAGE
    source = physical_bytes(ROOT / GUARD).replace(b'\r\n', b'\n')
    key = (source, scope.canonical(binding),
           physical_bytes(ROOT / 'scripts/ci_fixture_stability_scope.py').replace(b'\r\n', b'\n'),
           binding['predecessor_guard']['sha256'])
    previous = _PREDECESSOR_IMAGE
    image = scope.previous_guard(source, binding,
        predecessor=previous[1] if previous is not None and previous[0] == key else None)
    _PREDECESSOR_IMAGE = (key, image)
    return image


def canonical_guard():
    global _LIVE_GUARD
    if _LIVE_GUARD is None:
        from scripts import check_launch_change_scope
        if not hasattr(check_launch_change_scope, '_ci_fixture_module'):
            raise ValueError('HISTORICAL_CURRENT_GUARD_UNAVAILABLE')
        _LIVE_GUARD = check_launch_change_scope
    return _LIVE_GUARD


def require_historical_fixture(name):
    # An explicit legacy request needs a physical, source-bound call contract.
    if not _ACTIVE or _ACTIVE[-1].plan.role is not Role.HISTORICAL:
        raise ValueError('HISTORICAL_FIXTURE_CONTEXT_REQUIRED')
    owner = _ACTIVE[-1]
    authenticate_plan(owner.plan)
    binding = json.loads(physical_bytes(ROOT / BINDING))
    consumer = binding['historical_input_boundary']['consumers'][owner.plan.consumer]
    if consumer.get('requested_historical_fixtures', {}).get(name) != 'ncaaf-future':
        raise ValueError('HISTORICAL_FIXTURE_REQUEST_NOT_AUTHENTICATED')
    module_name='exact_ncaaf_future_released_scope'
    entry=owner.items.get((id(sys.modules),module_name))
    if entry is None:
        raise ValueError('HISTORICAL_FIXTURE_MODULE_UNAVAILABLE')
    module=entry[3]
    if sys.modules.get(module_name) is not module:
        raise ValueError('HISTORICAL_FIXTURE_MODULE_CONFLICT')
    return module


def owned_api_patch(patch, plan):
    # Legacy API callers retain their explicit undo lifetime. The same owner
    # tracks these views so an inner native context can suspend them safely.
    authenticate_plan(plan)
    if _ACTIVE and patch is _ACTIVE[-1].patch:
        return patch
    class LegacyPatch:
        def __init__(self):
            self.records=[]
            self.owner=None
            self.original_undo=None
            self.saved_undo=_ABSENT
        def begin(self):
            if self.owner is not None:
                if not _ACTIVE or _ACTIVE[-1] is not self.owner:
                    raise ValueError('HISTORICAL_API_OWNER_CONFLICT')
                return
            if not _ACTIVE or _ACTIVE[-1].plan.role is Role.LIVE:
                raise ValueError('HISTORICAL_API_OWNER_REQUIRED')
            owner=_ACTIVE[-1]
            authenticate_plan(owner.plan)
            authenticate_plan(plan)
            original=patch.undo
            self.saved_undo=vars(patch).get('undo',_ABSENT)
            self.original_undo=original
            self.owner=owner
            patch.undo=self.undo
        def setattr(self, target, name, value, raising=True):
            self.begin()
            if len(self.records)>=128:
                raise ValueError('HISTORICAL_API_VIEW_LIMIT')
            key=(id(target),name)
            old=self.owner.attributes.get(key)
            old=list(old) if old is not None else None
            previous=getattr(target,name)
            self.owner.patch.setattr(target,name,value,raising=raising)
            self.records.append(('attribute',target,name,previous,value,key,old))
        def setitem(self, mapping, name, value):
            self.begin()
            if len(self.records)>=128:
                raise ValueError('HISTORICAL_API_VIEW_LIMIT')
            key=(id(mapping),name)
            old=self.owner.items.get(key)
            old=list(old) if old is not None else None
            previous=mapping.get(name,_ABSENT)
            self.owner.patch.setitem(mapping,name,value)
            self.records.append(('item',mapping,name,previous,value,key,old))
        def undo(self):
            try:
                suspended=self.owner in _ACTIVE and _ACTIVE[-1] is not self.owner
                for kind,target,name,previous,installed,key,old in reversed(self.records):
                    if kind=='attribute':
                        if not suspended and getattr(target,name) is installed:
                            setattr(target,name,previous)
                        entries=self.owner.attributes
                        snapshots='suspended'
                    else:
                        if not suspended and target.get(name,_ABSENT) is installed:
                            if previous is _ABSENT:target.pop(name,None)
                            else:target[name]=previous
                        entries=self.owner.items
                        snapshots='suspended_items'
                    if suspended:
                        saved=[]
                        for row in getattr(self.owner,snapshots,[]):
                            obj,field,original,value=row
                            if obj is target and field==name and value is installed:
                                if old is not None:saved.append((obj,field,original,old[3]))
                            else:saved.append(row)
                        setattr(self.owner,snapshots,saved)
                    current=entries.get(key)
                    if current is not None and current[3] is installed:
                        if old is None:entries.pop(key,None)
                        else:entries[key]=old
                self.records.clear()
            finally:
                if self.saved_undo is _ABSENT:vars(patch).pop('undo',None)
                else:patch.undo=self.saved_undo
                self.original_undo()
    return LegacyPatch()


def owned_fixture_patch(patch):
    # Explicit fixtures use the same ledger as the selected physical consumer.
    if not _ACTIVE or _ACTIVE[-1].plan.role is not Role.HISTORICAL:
        raise ValueError('HISTORICAL_FIXTURE_CONTEXT_REQUIRED')
    owner=_ACTIVE[-1]
    authenticate_plan(owner.plan)
    caller=sys._getframe(1)
    try:relative=Path(caller.f_code.co_filename).resolve().relative_to(ROOT).as_posix()
    except ValueError:raise ValueError('HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED') from None
    data=json.loads(physical_bytes(ROOT/BINDING))['historical_input_boundary']
    allowed=data.get('owned_explicit_fixtures',{}).get(relative,[])
    record=data['consumers'].get(relative) or data.get('helper_preimages',{}).get(relative)
    source=physical_bytes(ROOT/relative) if record else b''
    if (record is None or sha(source)!=record.get('sha256',record.get('after_sha256'))
        or caller.f_code.co_name not in allowed):
        raise ValueError('HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED')
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
        and n.name==caller.f_code.co_name
        and min([n.lineno]+[d.lineno for d in n.decorator_list])==caller.f_code.co_firstlineno]
    if len(functions)!=1 or not any(isinstance(d,ast.Attribute) and d.attr=='fixture'
        or isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='fixture'
        for d in functions[0].decorator_list):
        raise ValueError('HISTORICAL_FIXTURE_CALLSITE_NOT_AUTHENTICATED')
    return owned_api_patch(patch,owner.plan)


def register_owned_module(module):
    # Legacy helper callers observe their constructed module until their exact
    # historical context exits. No persistent module or root-only cache remains.
    if _ACTIVE and _ACTIVE[-1].plan.role is Role.HISTORICAL:
        owner=_ACTIVE[-1]
        authenticate_plan(owner.plan)
        owner.patch.setitem(sys.modules,module.__name__,module)

class InputOwner:
    """One lifetime owns all migrated views and aliases, with nested suspension."""
    def __init__(self, plan):
        self.plan = plan
        self.patch = None
        self.parent = None
        self.saved = None

    def __enter__(self):
        import pytest
        self.parent = _ACTIVE[-1] if _ACTIVE else None
        self.saved = (Path.read_bytes, Path.open)
        if self.parent is not None:
            # Undo only owned attributes temporarily; unrelated patches survive.
            self.parent.suspend()
        owner = self
        class OwnedPatch(pytest.MonkeyPatch):
            def setattr(self, target, name, value, raising=True):
                key = (id(target), name)
                if key not in owner.attributes:
                    owner.attributes[key] = [target, name, getattr(target, name, _ABSENT), value]
                else:
                    owner.attributes[key][3] = value
                return super().setattr(target, name, value, raising=raising)
            def setitem(self, mapping, name, value):
                key = (id(mapping), name)
                if key not in owner.items:
                    owner.items[key] = [mapping, name, mapping.get(name, _ABSENT), value]
                else:
                    owner.items[key][3] = value
                return super().setitem(mapping, name, value)
        self.attributes = {}
        self.items = {}
        self.patch = OwnedPatch()
        _ACTIVE.append(self)
        return self

    def suspend(self):
        self.suspended = []
        for obj, name, original, installed in self.attributes.values():
            if getattr(obj, name, _ABSENT) is installed:
                self.suspended.append((obj, name, original, installed))
                if original is _ABSENT:delattr(obj, name)
                else:setattr(obj, name, original)
        self.suspended_items = []
        for mapping, name, original, installed in self.items.values():
            if mapping.get(name, _ABSENT) is installed:
                self.suspended_items.append((mapping, name, original, installed))
                if original is _ABSENT:
                    mapping.pop(name, None)
                else:
                    mapping[name] = original

    def resume(self):
        for mapping, name, original, value in reversed(self.suspended_items):
            if mapping.get(name, _ABSENT) is original:
                mapping[name] = value
        for obj, name, original, value in reversed(self.suspended):
            if getattr(obj, name, _ABSENT) is original:
                setattr(obj, name, value)

    def __exit__(self, *exc):
        try:
            # A later unrelated patch is not owned by this context. Restore only
            # values that still match our installation, including module entries.
            for mapping, name, original, installed in reversed(list(self.items.values())):
                if mapping.get(name, _ABSENT) is installed:
                    if original is _ABSENT:
                        mapping.pop(name, None)
                    else:
                        mapping[name] = original
            for obj, name, original, installed in reversed(list(self.attributes.values())):
                if getattr(obj, name, _ABSENT) is installed:
                    if original is _ABSENT:delattr(obj, name)
                    else:setattr(obj, name, original)
        finally:
            assert _ACTIVE.pop() is self
            if self.parent is not None:
                self.parent.resume()



def capacity_input_records(plan):
    """Authenticate an explicit input operation before inspecting its generation."""
    authenticate_plan(plan)
    binding_raw=physical_bytes(ROOT/BINDING)
    if sha(binding_raw)!=plan.binding_sha256:
        raise ValueError('HISTORICAL_BINDING_CHANGED')
    binding=json.loads(binding_raw)
    relative='tests/ci_fixture_stability_inputs.py';raw=physical_bytes(ROOT/relative)
    identity=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if identity!=binding['reviewed_blobs'][relative]:
        raise ValueError('CI_CAPACITY_INPUT_HELPER_CHANGED')
    from tests import ci_fixture_stability_inputs as inputs
    records=json.loads(json.dumps(inputs.CAPACITY_SOURCE_RECORDS))
    encoded=(json.dumps(records,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if (sha(encoded)!=binding['measured_capacity']['source_records_sha256']
            or set(records)!=set(binding['measured_capacity']['source_paths'])):
        raise ValueError('CI_CAPACITY_INPUT_GENERATION_CHANGED')
    successors=json.loads(json.dumps(inputs.CAPACITY_SUCCESSOR_RECORDS))
    encoded=(json.dumps(successors,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    meta=binding['measured_capacity']
    if (sha(encoded)!=meta['successor_records_sha256']
            or set(successors)!=set(meta['successor_paths'])
            or set(successors)!=set(records)-{inputs.PROFILE}):
        raise ValueError('CI_CAPACITY_INPUT_GENERATION_CHANGED')
    for relative,record in successors.items():
        if record['before_sha256']!=records[relative]['after_sha256']:
            raise ValueError('CI_CAPACITY_INPUT_GENERATION_CHANGED')
        records[relative]['successor']=record
    return records

def install_capacity_input_view(plan,patch):
    if plan.role is not Role.HISTORICAL:return False
    patch=owned_api_patch(patch,plan)
    records=capacity_input_records(plan)
    native_open=Path.open
    def transition(row,raw):
        digest=sha(raw.replace(b'\r\n',b'\n'))
        if digest==row['before_sha256']:return raw
        if digest!=row['after_sha256']:raise ValueError('CI_CAPACITY_INPUT_GENERATION_CHANGED')
        z=zlib.decompressobj();value=z.decompress(base64.b85decode(row['compressed']),8*1024*1024+1)
        identity=hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()
        if not (z.eof and not z.unused_data and len(value)==row['bytes']<=8*1024*1024
                and sha(value)==row['before_sha256'] and identity==row['before_git_blob']):
            raise ValueError('CI_CAPACITY_AUTHENTICATED_PREDECESSOR_CHANGED')
        return value
    def image(relative,raw):
        row=records[relative];successor=row.get('successor')
        # Physical source must be the current authenticated generation. Only
        # the selected immutable edge(s) may produce a historical view.
        expected=successor['after_sha256'] if successor is not None else row['after_sha256']
        if sha(raw.replace(b'\r\n',b'\n'))!=expected:
            raise ValueError('CI_CAPACITY_INPUT_GENERATION_CHANGED')
        if successor is not None:raw=transition(successor,raw)
        return transition(row,raw)
    def opened(path,mode='r',*args,**kwargs):
        try:relative=path.resolve().relative_to(ROOT).as_posix()
        except ValueError:relative=None
        if relative not in records or mode not in ('r','rb'):return native_open(path,mode,*args,**kwargs)
        value=image(relative,physical_bytes(path))
        if mode=='rb':return io.BytesIO(value)
        import locale
        encoding=kwargs.get('encoding') or 'utf-8'
        if encoding=='locale':encoding=locale.getencoding()
        return io.StringIO(value.decode(encoding))
    patch.setattr(Path,'open',opened)
    return True


def install_transitive_module_inputs(plan, patch):
    """A declared generation owns fresh readers, independent of warm imports."""
    if plan.role is not Role.HISTORICAL:return
    authenticate_plan(plan)
    record=json.loads(physical_bytes(ROOT/BINDING))['historical_input_boundary']['consumers'][plan.consumer]
    rows=record.get('transitive_modules',{})
    if not isinstance(rows,dict) or len(rows)>32:raise ValueError('HISTORICAL_TRANSITIVE_LIMIT')
    if not rows:return
    if not _ACTIVE or _ACTIVE[-1].plan!=plan or patch is not _ACTIVE[-1].patch:
        raise ValueError('HISTORICAL_TRANSITIVE_OWNER_REQUIRED')
    import importlib
    package=importlib.import_module('tests')
    pending=[]
    for name,row in rows.items():
        if (set(row)!={'path','physical_sha256','input_sha256','generation'}
            or not name.startswith('tests.test_') or name.count('.')!=1
            or row['path']!=name.replace('.','/')+'.py' or row['generation']!=plan.generation):
            raise ValueError('HISTORICAL_TRANSITIVE_CONTRACT_CHANGED')
        path=ROOT/row['path']
        if sha(physical_bytes(path))!=row['physical_sha256']:
            raise ValueError('HISTORICAL_TRANSITIVE_SOURCE_CHANGED')
        value=path.read_bytes().replace(b'\r\n',b'\n')
        with path.open('rb') as stream:opened=stream.read().replace(b'\r\n',b'\n')
        if sha(value)!=row['input_sha256'] or opened!=value:
            raise ValueError('HISTORICAL_TRANSITIVE_VIEW_CHANGED')
        module=types.ModuleType(name);module.__file__=str(path);module.__package__='tests'
        pending.append((name,module,value))
    # Own package attributes and import entries before execution. No mutable
    # namespace is cached or borrowed from another generation.
    for name,module,value in pending:
        patch.setitem(sys.modules,name,module)
        patch.setattr(package,name.split('.')[1],module,raising=False)
    for name,module,value in pending:
        exec(compile(value,module.__file__,'exec'),module.__dict__)

def apply_selected_inputs(plan, request, patch):
    if plan.role is not Role.HISTORICAL:
        return False
    live = authenticate_plan(plan)
    if plan.generated_contract:
        _install_generated_source_input(plan,patch,live)
        return True
    install_capacity_input_view(plan,patch)
    records=json.loads(physical_bytes(ROOT/BINDING))['historical_input_boundary'].get('helper_preimages',{})
    if records:
        read=Path.read_bytes
        native_open=Path.open
        def exact_helper_open(path,mode='r',*args,**kwargs):
            try:relative=path.resolve().relative_to(ROOT).as_posix()
            except ValueError:relative=None
            if relative not in records or mode not in ('r','rb'):
                return native_open(path,mode,*args,**kwargs)
            with native_open(path,'rb') as stream:value=stream.read()
            value=historical_helper_input(plan,relative,value)
            if mode=='rb':return io.BytesIO(value)
            import locale
            encoding=kwargs.get('encoding') or 'utf-8'
            if encoding=='locale':encoding=locale.getencoding()
            return io.StringIO(value.decode(encoding))
        def exact_helper(path):
            value=read(path)
            try:relative=path.resolve().relative_to(ROOT).as_posix()
            except ValueError:return value
            return historical_helper_input(plan,relative,value) if relative in records else value
        patch.setattr(Path,'open',exact_helper_open)
        patch.setattr(Path,'read_bytes',exact_helper)
    from tests.ci_fixture_stability_inputs import _install_selected_predecessor_sources
    from tests.ncaaf_staged_capacity_fixtures import _install_selected_profile_inputs
    from tests.ncaaf_transport_scope_inputs import _install_selected_transport_inputs
    from tests.ncaaf_future_scope_inputs import _install_selected_future_inputs
    _install_selected_predecessor_sources(request, patch)
    _install_selected_profile_inputs(request, patch)
    _install_selected_transport_inputs(request, patch, plan)
    _install_selected_future_inputs(request, patch, live, plan)
    install_transitive_module_inputs(plan,patch)
    return True
