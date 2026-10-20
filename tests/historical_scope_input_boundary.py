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
            'test_second_advisory_preimage_is_narrow_authenticated_and_restored'}
    return (name, node) in {
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
    for relative, record in binding['historical_input_boundary'].get('helper_preimages',{}).items():
        if sha(physical_bytes(ROOT/relative))!=record['after_sha256']:
            raise ValueError('HISTORICAL_HELPER_SOURCE_CHANGED')
    scope.previous_guard(physical_bytes(ROOT / GUARD).replace(b'\r\n', b'\n'), binding)
    return live


_ACTIVE = []
_LIVE_GUARD = None
_ABSENT = object()


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
                    owner.attributes[key] = [target, name, getattr(target, name), value]
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
            if getattr(obj, name) is installed:
                self.suspended.append((obj, name, original, installed))
                setattr(obj, name, original)
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
            if getattr(obj, name) is original:
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
                if getattr(obj, name) is installed:
                    setattr(obj, name, original)
        finally:
            assert _ACTIVE.pop() is self
            if self.parent is not None:
                self.parent.resume()


def apply_selected_inputs(plan, request, patch):
    if plan.role is not Role.HISTORICAL:
        return False
    live = authenticate_plan(plan)
    if plan.generated_contract:
        _install_generated_source_input(plan,patch,live)
        return True
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
    return True
