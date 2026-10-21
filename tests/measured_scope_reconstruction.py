"""Owned reuse of the measured pure leaf, using the unchanged native memo."""
import types
from scripts import scope_reconstruction as native


def install(module):
    method=getattr(module,'_listing_capacity_base_source',None)
    if method is None:return None
    # A private namespace selects one existing pure transformation. No global
    # native registry, mutable historical module or validation result is reused.
    environment=dict(vars(native),NAMES=(*native.NAMES,'_listing_capacity_base_source'))
    installer=types.FunctionType(native.install.__code__,environment)
    # Install the complete declared graph together, so its snapshots account for
    # all deliberate delegate replacements. The module belongs to one context.
    for name in native.NAMES:
        previous=getattr(module,name,None)
        if hasattr(previous,'reconstruction_memo') and hasattr(previous,'__wrapped__'):
            setattr(module,name,previous.__wrapped__)
    module._scope_pure_reconstruction_installed=False
    installer(module)
    return getattr(module._listing_capacity_base_source,'reconstruction_memo',None)
