"""Regenerate the NCAAF production runtime binding after a reviewed source change.

Usage:  python scripts/rebind_ncaaf_runtime.py [--check]

The binding pins production source and frozen evidence that the private NCAAF
Refresh path authenticates at import. Tests, workflows and CI profiles are
deliberately excluded so test-only changes never disable the app. Run this only
in a commit whose production change has been reviewed; --check verifies without
writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ncaaf_private_refresh_scope as scope  # noqa: E402

PRODUCTION_PREFIXES = ('app/', 'app_core/', 'services/', 'docs/research/', 'docs/paid-launch/')
PRODUCTION_FILES = {'streamlit_app.py', 'requirements.txt', 'scripts/ncaaf_reviewed_target_operator.py',
                    'scripts/ncaaf_staged_acquisition.py', 'scripts/ncaaf_filtered_history_scope.py',
                    scope.SELF}


def production_components() -> list[str]:
    """Exactly the production subset of the frozen 2026-11-04 history binding."""
    history = json.loads(scope.physical(ROOT / scope.BINDING))['components']
    chosen = {p for p in history if p in PRODUCTION_FILES or p.startswith(PRODUCTION_PREFIXES)}
    chosen.add(scope.SELF)
    return sorted(chosen)


def build(components: list[str]) -> bytes:
    record = {
        'version': 'ncaaf-runtime-binding-v2',
        'frozen_history_binding': {'path': scope.BINDING, 'sha256': scope._FILTERED_HISTORY_BINDING_SHA256},
        'excluded': 'tests/, .github/, CI timing profiles and retired seal guards are not runtime inputs',
        'components': {p: hashlib.sha256(scope.runtime_component_bytes(ROOT, p)).hexdigest() for p in components},
    }
    return (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    binding_path = ROOT / scope.RUNTIME_BINDING
    components = sorted(json.loads(binding_path.read_bytes())['components']) if binding_path.is_file() else production_components()
    raw = build(components)
    digest = hashlib.sha256(raw).hexdigest()
    if args.check:
        current = binding_path.read_bytes() if binding_path.is_file() else b''
        ok = current == raw and digest == scope._RUNTIME_BINDING_SHA256
        print('NCAAF runtime binding', 'OK' if ok else 'STALE: run python scripts/rebind_ncaaf_runtime.py')
        return 0 if ok else 1
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_bytes(raw)
    source_path = ROOT / scope.SELF
    source = source_path.read_bytes()
    source, count = re.subn(rb"_RUNTIME_BINDING_SHA256='[0-9a-f]{64}'",
                            b"_RUNTIME_BINDING_SHA256='" + digest.encode() + b"'", source)
    if count != 1:
        raise SystemExit('runtime pin not found exactly once')
    source_path.write_bytes(source)
    print(f'Wrote {scope.RUNTIME_BINDING} ({len(components)} components) sha256={digest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
