"""Protected-change guard (re-baselined 2026-11-04; replaces the successor-seal chain).

Rules, checked against a comparison base (the PR base, or the first parent on main):

1. Frozen evidence is append-only: files that already exist under FROZEN_PREFIXES
   may not be modified or deleted. New evidence files may be added.
2. Existing test assertions are preserved: in every modified test module, each
   ``assert`` statement present at the base must still be present (additions OK).
3. No new skips: skip/xfail/importorskip usage in a test module may not increase.
4. Tests are never silently dropped: a test module may leave ``tests/`` only by
   moving to ``tests_archive/`` (retained and listed in the report).
5. The NCAAF production runtime binding must be current
   (``python scripts/rebind_ncaaf_runtime.py --check``).

Production, model and workflow changes are reported for human review; they are
not blocked here. Standard library only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PREFIXES = ('docs/paid-launch/', 'docs/research/', 'docs/audits/', 'docs/integrity/frozen-sources/')
# Last head sealed under the retired successor-seal guard. Frozen files that are
# byte-identical to this commit were already accepted by that guard.
REBASELINE_COMMIT = '8181ff7d063b75985d51871b263425ba2f012861'
TEST_PREFIX = 'tests/'
ARCHIVE_PREFIX = 'tests_archive/'
REVIEW_PREFIXES = {'production': ('app/', 'app_core/', 'core/', 'services/', 'streamlit_app.py'),
                   'models': ('models/', 'core/models/'),
                   'workflows': ('.github/workflows/',)}
PROTECTED_RULES = {'FROZEN_EVIDENCE_CHANGED', 'NCAAF_RUNTIME_BINDING_STALE'}
SKIP_PATTERN = re.compile(rb'pytest\.mark\.(?:skip|skipif|xfail)\b|pytest\.(?:skip|xfail|importorskip)\s*\(')


def git(*args: str) -> bytes:
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True).stdout


def show(rev: str, path: str) -> bytes:
    return git('show', f'{rev}:{path}')


def sealed_at_rebaseline(path: str) -> bool:
    try:
        return (ROOT / path).read_bytes() == show(REBASELINE_COMMIT, path)
    except subprocess.CalledProcessError:
        return False


def assertions(source: bytes) -> Counter:
    tree = ast.parse(source)
    return Counter(ast.dump(node, include_attributes=False) for node in ast.walk(tree) if isinstance(node, ast.Assert))


def is_test_module(path: str) -> bool:
    return path.endswith('.py') and (path.startswith(TEST_PREFIX) or path.startswith(ARCHIVE_PREFIX))


def changes(base: str) -> list[tuple[str, str, str | None]]:
    """(status, old_path, new_path) from git, with rename detection."""
    rows = []
    for line in git('diff', '--name-status', '-M', '--no-color', base, 'HEAD').decode().splitlines():
        parts = line.split('\t')
        status = parts[0][0]
        if status == 'R':
            rows.append(('R', parts[1], parts[2]))
        else:
            rows.append((status, parts[1], parts[1] if status != 'D' else None))
    return rows


def runtime_binding_problem() -> str | None:
    result = subprocess.run([sys.executable, 'scripts/rebind_ncaaf_runtime.py', '--check'],
                            cwd=ROOT, capture_output=True, text=True)
    return None if result.returncode == 0 else (result.stdout + result.stderr).strip()[-400:]


def check(base: str) -> dict:
    violations: list[dict] = []
    review: dict[str, list[str]] = {k: [] for k in REVIEW_PREFIXES}
    retired: list[str] = []
    for status, old, new in changes(base):
        for kind, prefixes in REVIEW_PREFIXES.items():
            if any((new or old).startswith(p) for p in prefixes):
                review[kind].append(new or old)
        if old.startswith(FROZEN_PREFIXES) and status in {'M', 'D', 'R', 'T'}:
            if status == 'M' and sealed_at_rebaseline(old):
                continue
            violations.append({'rule': 'FROZEN_EVIDENCE_CHANGED', 'path': old, 'status': status})
        if not is_test_module(old) or status == 'A':
            continue
        if status == 'D' or (new and not is_test_module(new)):
            violations.append({'rule': 'TEST_MODULE_REMOVED', 'path': old})
            continue
        if old.startswith(TEST_PREFIX) and new.startswith(ARCHIVE_PREFIX):
            retired.append(new)
        before = show(base, old)
        after = (ROOT / new).read_bytes()
        try:
            lost = assertions(before) - assertions(after)
        except SyntaxError:
            violations.append({'rule': 'TEST_MODULE_UNPARSEABLE', 'path': new})
            continue
        if lost:
            violations.append({'rule': 'TEST_ASSERTION_REMOVED_OR_CHANGED', 'path': new, 'count': sum(lost.values())})
        if len(SKIP_PATTERN.findall(after)) > len(SKIP_PATTERN.findall(before)):
            violations.append({'rule': 'TEST_SKIP_ADDED', 'path': new})
    stale = runtime_binding_problem()
    if stale:
        violations.append({'rule': 'NCAAF_RUNTIME_BINDING_STALE', 'detail': stale})
    return {'schema': 'protected-change-guard-v1', 'base': git('rev-parse', base).decode().strip(),
            'head': git('rev-parse', 'HEAD').decode().strip(),
            'status': 'FAIL' if violations else 'PASS', 'violations': violations,
            'protected_changes': [v for v in violations if v['rule'] in PROTECTED_RULES],
            'existing_test_changes': [v for v in violations if v['rule'] not in PROTECTED_RULES],
            'retired_test_modules': sorted(retired), 'changes_for_review': review}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Verify protected-change rules against a base commit.')
    parser.add_argument('--base', default='HEAD^1', help='comparison base (default: first parent)')
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args(argv)
    report = check(args.base)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + '\n', encoding='utf-8')
    summary = {k: report[k] for k in ('status', 'base', 'head', 'violations',
                                      'protected_changes', 'existing_test_changes')}
    summary['retired_test_modules'] = len(report['retired_test_modules'])
    summary['changes_for_review'] = {k: len(v) for k, v in report['changes_for_review'].items()}
    print(json.dumps(summary, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
