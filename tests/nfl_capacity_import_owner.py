"""Own the exact predecessor fixture's transitive interpreter effects."""
from contextlib import contextmanager
from pathlib import Path
import sys

MODULES = ('scripts.ci_scheduling_fixture_scope', 'scripts.ncaaf_transport_scope',
    'scripts.ncaaf_reviewed_target_operator', 'scripts.ncaaf_filtered_history_scope',
    'scripts.ncaaf_private_refresh_scope', 'exact_capacity_parent_guard')
ABSENT = object()


@contextmanager
def predecessor_imports(historical):
    """Only these authenticated guard dependencies belong to this lifetime.

    No unrelated imported module is evicted. Incoming module identities,
    namespace aliases and byte-image caches survive even a setup exception.
    """
    historical=Path(historical).resolve()
    package=sys.modules['scripts'];incoming_path=sys.path;path_values=list(incoming_path)
    namespace=package.__path__;namespace_state=dict(vars(namespace)) if hasattr(namespace,'__dict__') else None
    namespace_lists={k:(v,list(v)) for k,v in (namespace_state or {}).items() if isinstance(v,list)}
    rows=[];caches={}
    for name in MODULES:
        module=sys.modules.get(name,ABSENT)
        attributes=dict(vars(module)) if module is not ABSENT else None
        leaf=name.split('.',1)[1] if name.startswith('scripts.') else None
        alias=vars(package).get(leaf,ABSENT) if leaf else ABSENT
        rows.append((name,module,attributes,leaf,alias))
        for key,value in (attributes or {}).items():
            if key.endswith('_IMAGES') and isinstance(value,dict):caches[id(value)]=(value,dict(value))
    importer=sys.path_importer_cache
    def owned_path(key):
        try:return Path(key).resolve().is_relative_to(historical)
        except (TypeError,ValueError,OSError):return False
    importer_before={k:v for k,v in importer.items() if owned_path(k)}
    try:
        yield
    finally:
        # Restore identities, not replacements with copied module namespaces.
        for name,module,attributes,leaf,alias in reversed(rows):
            if module is ABSENT:sys.modules.pop(name,None)
            else:
                sys.modules[name]=module
                for key in set(vars(module))-set(attributes):delattr(module,key)
                vars(module).update(attributes)
            if leaf:
                if alias is ABSENT:vars(package).pop(leaf,None)
                else:setattr(package,leaf,alias)
        for cache,values in caches.values():cache.clear();cache.update(values)
        incoming_path[:]=path_values;sys.path=incoming_path
        package.__path__=namespace
        if namespace_state is not None:
            vars(namespace).clear();vars(namespace).update(namespace_state)
            for value,items in namespace_lists.values():value[:]=items
        for key in tuple(importer):
            if owned_path(key) and key not in importer_before:del importer[key]
        importer.update(importer_before)
