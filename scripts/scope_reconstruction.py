"""Bounded memo of pure, exact byte reconstruction; never validation results."""
from collections import OrderedDict
from functools import wraps
import hashlib
import types

NAMES = ('_nfl_odds_timing_base_source', '_nfl_odds_base_source')
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_RETAINED_BYTES = 32 * 1024 * 1024
MAX_ENTRIES = 8
MAX_CONTEXT_FIELDS = 512


def _freeze(value, budget=None):
    if budget is None: budget=[256*1024,4096,64]
    budget[1]-=1
    if budget[1]<0: raise ValueError('RECONSTRUCTION_CONTEXT_LIMIT')
    if isinstance(value,(str,bytes)):
        budget[0]-=len(value)
        if budget[0]<0: raise ValueError('RECONSTRUCTION_CONTEXT_LIMIT')
        return value
    if isinstance(value,(int,float,bool,type(None))): return value
    if isinstance(value,dict):
        budget[2]-=1
        if budget[2]<0: raise ValueError('RECONSTRUCTION_CONTEXT_LIMIT')
        try: return tuple(sorted((_freeze(k,budget),_freeze(v,budget)) for k,v in value.items()))
        finally: budget[2]+=1
    if isinstance(value,(list,tuple)):
        budget[2]-=1
        if budget[2]<0: raise ValueError('RECONSTRUCTION_CONTEXT_LIMIT')
        try: return tuple(_freeze(v,budget) for v in value)
        finally: budget[2]+=1
    raise ValueError('RECONSTRUCTION_MUTABLE_CONTEXT')


class ReconstructionMemo:
    def __init__(self):
        self.entries = OrderedDict(); self.retained_bytes = 0
        self.hits = 0; self.misses = 0

    def call(self, method, source, context):
        if not isinstance(source, bytes) or len(source) > MAX_SOURCE_BYTES:
            return method(source)
        key = (method, context, source)
        if key in self.entries:
            self.hits += 1; self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses += 1
        result = method(source)
        if not isinstance(result, bytes) or len(result) > MAX_SOURCE_BYTES:
            return result
        cost = len(source) + len(result)
        while self.entries and (len(self.entries) >= MAX_ENTRIES or
                               self.retained_bytes + cost > MAX_RETAINED_BYTES):
            _, (_, removed) = self.entries.popitem(last=False)
            self.retained_bytes -= removed
        if cost <= MAX_RETAINED_BYTES:
            self.entries[key] = (result, cost); self.retained_bytes += cost
        return result


def _context_fields(method):
    """Follow captured/delegated source adapters; refuse mutable/I/O contexts."""
    pending=[method]; seen=set(); fields=[]; keys=set()
    while pending:
        fn=pending.pop()
        if id(fn) in seen: continue
        seen.add(id(fn))
        unwrapped=getattr(fn, '__wrapped__', None)
        if unwrapped is not None:
            pending.append(unwrapped);fields.append((fn,'__wrapped__',unwrapped,True))
        if not isinstance(fn, types.FunctionType):
            if unwrapped is None: return None
            continue
        fields.append((fn,'__code__',fn.__code__,True))
        for attr in ('__defaults__','__kwdefaults__'):
            value=getattr(fn,attr)
            try:_freeze(value)
            except (ValueError,TypeError):return None
            fields.append((fn,attr,value,False))
        for cell in fn.__closure__ or ():
            value=cell.cell_contents
            if callable(value):
                pending.append(value);fields.append((cell,'cell_contents',value,True))
            elif not isinstance(value, (str, bytes, int, float, bool, type(None))):
                return None
            else:fields.append((cell,'cell_contents',value,False))
        if len(fields)>MAX_CONTEXT_FIELDS:return None
        for name in fn.__code__.co_names:
            namespace=fn.__globals__
            if name in {'ROOT','Path','git','git_bytes','exists_at','open','eval','exec','compile','__import__'}: return None
            if name not in namespace: continue
            key=(id(namespace),name)
            if key in keys: continue
            keys.add(key); value=namespace[name]
            if isinstance(value, types.FunctionType) or callable(getattr(value,'__wrapped__',None)):
                pending.append(value); fields.append((namespace,name,value,True))
            elif isinstance(value,(str,bytes,int,float,bool,type(None))):
                fields.append((namespace,name,value,False))
            elif isinstance(value,(dict,list,tuple)):
                try: _freeze(value)
                except (ValueError,TypeError): return None
                fields.append((namespace,name,value,False))
            elif isinstance(value,types.ModuleType) and value.__name__=='hashlib':
                fields.append((vars(value),'sha256',value.sha256,True))
            elif isinstance(value,types.ModuleType) and value.__name__=='json':
                fields.extend((vars(value),part,getattr(value,part),True) for part in ('loads','dumps'))
            elif isinstance(value,type) and issubclass(value,Exception):
                fields.append((namespace,name,value,True))
            else: return None
            if len(fields)>MAX_CONTEXT_FIELDS: return None
    return fields


def install(g):
    if getattr(g, '_scope_pure_reconstruction_installed', False): return
    candidates=[]
    for name in NAMES:
        method=getattr(g,name,None)
        if not callable(method): continue
        fields=_context_fields(method)
        if fields is None: continue
        candidates.append((name,method,fields))
    replacements={}; finalizers=[]
    for name,method,fields in candidates:
        def wrap(method, fields):
            memo=ReconstructionMemo()
            state=[]
            def finalize():
            # Compile only fixed field access from this reviewed graph. The
            # getter snapshots live metadata on EVERY call; it caches no facts.
                environment={'_id':id};identity_reads=[];identity_expected=[];container_reads=[]
                for index,(namespace,key,original,identity) in enumerate(fields):
                    if isinstance(namespace,dict) and identity and namespace.get(key) is replacements.get(original):
                        original=replacements[original]
                        fields[index]=(namespace,key,original,identity)
                    token='_field_'+str(index);environment[token]=namespace
                    expression=token+'.get('+repr(key)+')' if isinstance(namespace,dict) else token+'.'+key
                    if identity or type(original) in (str,bytes,int,float,bool,type(None)):
                        identity_reads.append('_id('+expression+')');identity_expected.append(id(original))
                    else:container_reads.append(expression)
                identities='('+','.join(identity_reads)+',)' if identity_reads else '()'
                containers='('+','.join(container_reads)+',)' if container_reads else '()'
                text='def snapshot():\n    return '+identities+', '+containers+'\n'
                exec(compile(text,'<reviewed-reconstruction-context>','exec'),environment)
                state.extend((environment['snapshot'],tuple(identity_expected),fields))
            @wraps(method)
            def wrapped(source):
                try:
                    identities,containers=state[0]()
                    if identities!=state[1]:return method(source)
                    values=tuple(_freeze(value) for value in containers)
                except (AttributeError,ValueError,TypeError):return method(source)
                context=(getattr(g,'__file__',None),values)
                return memo.call(method,source,context)
            wrapped.reconstruction_memo=memo
            wrapped.reconstruction_context_fields=len(fields)
            return wrapped,finalize
        wrapped,finalize=wrap(method,fields)
        setattr(g,name,wrapped);finalizers.append(finalize)
        replacements[method]=getattr(g,name)
    # Only these deliberate replacements are allowed. Other delegate changes
    # bypass the memo and reexecute the original source reconstruction.
    for finalize in finalizers:finalize()
    g._scope_pure_reconstruction_installed=True
