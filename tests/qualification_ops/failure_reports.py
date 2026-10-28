"""Bounded, allowlisted evidence for synthetic qualification children.

Missing reports never mean zero tests ran. Journals establish confirmed starts
and outcomes; other executions remain unconfirmed. Raw child streams and result
details are private scratch, never part of the artifact allowlist.
"""
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

MAX_REPORT = 8 * 1024 * 1024
MAX_LOG = 128 * 1024
STATUSES = {'PASS', 'FAIL', 'ERROR', 'SKIP'}


def case_key(suite, name):
    # A script executes as __main__; collection imports its fixed suite name.
    # Normalize only that module alias, preserving the class and method.
    prefix, separator, rest = name.partition('.')
    return (suite, rest if separator and prefix in ('__main__', suite) else name)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def safe_test(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.]{1,240}', value):
        raise ValueError('RESULT_SCHEMA_INVALID')
    return value


def safe_class(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', value):
        raise ValueError('JOURNAL_SCHEMA_INVALID')
    return value


def entries(value):
    if not isinstance(value, list) or len(value) > 1000:
        raise ValueError('RESULT_SCHEMA_INVALID')
    cleaned = []
    for item in value:
        name = safe_test(item['test'])
        if item['status'] not in STATUSES:
            raise ValueError('RESULT_SCHEMA_INVALID')
        seconds = item.get('seconds', item.get('elapsed_seconds', 0))
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('RESULT_SCHEMA_INVALID')
        record = {'test': name, 'status': item['status'], 'seconds': seconds}
        if 'exception_class' in item:
            record['exception_class'] = safe_class(item['exception_class'])
        cleaned.append(record)
    return cleaned


def read_result(path):
    if not path.is_file():
        return None, 'RESULT_MISSING'
    try:
        if path.stat().st_size > MAX_REPORT:
            raise ValueError('RESULT_SIZE_LIMIT')
        value = json.loads(path.read_bytes())
        records = entries(value['results'])
        if (value.get('synthetic_only') is not True or type(value['success']) is not bool
                or type(value['tests_run']) is not int or value['tests_run'] != len(records)
                or type(value['real_socket_attempts']) is not int or value['real_socket_attempts'] < 0):
            raise ValueError('RESULT_SCHEMA_INVALID')
        # Keep measured fields needed for qualification. No exception details,
        # arbitrary strings, fixture bodies or credentials cross this boundary.
        return dict(results=records, tests_run=value['tests_run'], success=value['success'],
                    real_socket_attempts=value['real_socket_attempts']), 'RESULT_VALID'
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, OverflowError):
        return None, 'RESULT_MALFORMED'


def read_journal(path):
    starts, outcomes, events = [], [], []
    valid = True
    try:
        if path.stat().st_size > MAX_REPORT:
            raise ValueError('JOURNAL_SIZE_LIMIT')
        for line in path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            event = item['event']
            if event in ('CHILD_STARTED', 'CHILD_RETURNED'):
                events.append({'event': event})
            elif event == 'CHILD_FAILURE':
                events.append({'event': event, 'exception_class': safe_class(item['exception_class'])})
            elif event == 'CASE_STARTED':
                name = safe_test(item['test']); starts.append(name)
                events.append({'event': event, 'test': name})
            elif event == 'CASE_COMPLETED':
                record = entries([item])[0]; outcomes.append(record)
                events.append(dict(event=event, **{k: v for k, v in record.items() if k != 'seconds'}))
            else:
                raise ValueError('JOURNAL_SCHEMA_INVALID')
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        valid = False
    valid = bool(valid and events and events[0]['event'] == 'CHILD_STARTED'
                 and sum(item['event'] == 'CHILD_STARTED' for item in events) == 1)
    return starts, outcomes, events, valid


def retain_stream(source, destination, known):
    """Positive allowlist, not redaction of a known set of secret patterns."""
    kept, redacted = [], 0
    if source.exists():
        size = source.stat().st_size
        with source.open('rb') as stream:
            head = stream.read(MAX_LOG // 2)
            if size > len(head):
                stream.seek(max(len(head), size - MAX_LOG // 2))
                head += b'\n[BOUNDED_STREAM_GAP]\n' + stream.read(MAX_LOG // 2)
        for line in head.decode('utf-8', errors='replace').splitlines():
            line = line.strip()
            match = re.fullmatch(r'(test_[A-Za-z0-9_]+) \([A-Za-z0-9_.]+\) \.\.\. (ok|FAIL|ERROR|skipped .*)?', line)
            if match and match[1] in known:
                kept.append(match[1] + ': ' + (match[2] if match[2] in ('ok', 'FAIL', 'ERROR') else 'STARTED_OR_SKIPPED'))
            elif line in ('STAGE_WALL_LIMIT', 'OK', '[BOUNDED_STREAM_GAP]') or re.fullmatch(r'Ran \d+ tests? in [0-9.]+s', line) or re.fullmatch(r'FAILED \((?:failures|errors|skipped)=\d+(?:, (?:failures|errors|skipped)=\d+)*\)', line):
                kept.append(line)
            elif line:
                redacted += 1
    else:
        size = 0
    destination.write_text('\n'.join(kept) + '\nDETAIL_LINES_REDACTED=' + str(redacted) + '\n', encoding='utf-8')
    return dict(observed_stream_bytes=size, retained_stream_bytes=destination.stat().st_size,
                detail_lines_redacted=redacted, bounded=size > MAX_LOG)


def execute(here, output, name, extra, result_file, label, env, collection):
    diagnostics = output / 'diagnostics'; diagnostics.mkdir(exist_ok=True)
    private = output / ('.private-' + label)
    # The launcher writes only fixed events and source-defined unittest IDs.
    # Keep this live journal in the allowlist even if the parent is interrupted.
    journal = diagnostics / (label + '-events.jsonl')
    command = [sys.executable, '-B', '-X', 'utf8', str(here / 'child_runner.py'), str(journal), str(here / (name + '.py')), *extra]
    sanitized = ['python', '-B', '-X', 'utf8', 'tests/qualification_ops/child_runner.py',
                 'PRIVATE_EVENT_JOURNAL', 'tests/qualification_ops/' + name + '.py', extra[0],
                 'PRIVATE_TEST_OUTPUT', *extra[2:]]
    begun = time.monotonic(); code = None; category = 'CHILD_EXITED'
    write_json(diagnostics / (label + '-receipt.json'), dict(suite=name, command=sanitized, phase='CHILD_DISPATCH_STARTED', exit_code='UNKNOWN'))
    try:
        with private.open('xb') as log:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    except OSError:
        category = 'CHILD_START_FAILED'
    result, result_category = read_result(output / result_file)
    starts, observed, events, journal_valid = read_journal(journal)
    if result and (not journal_valid or events[-1]['event'] != 'CHILD_RETURNED'):
        result_category = 'RESULT_JOURNAL_INCOMPLETE'
    if result and observed and [(case_key(name, i['test']), i['status']) for i in observed] != [(case_key(name, i['test']), i['status']) for i in result['results']]:
        result_category = 'RESULT_JOURNAL_CONFLICT'
    records = result['results'] if result else observed
    for item in records:
        matching = [i for i in observed if case_key(name, i['test']) == case_key(name, item['test']) and i['status'] == item['status']]
        if len(matching) == 1 and 'exception_class' in matching[0]:
            item['exception_class'] = matching[0]['exception_class']
    # Do not repair a corrupt or substituted result with the journal. Both are
    # separately retained and all discrepancies remain qualification failures.
    record = dict(suite=name, partition=label, command=sanitized, exit_code=code if code is not None else 'UNKNOWN',
                  elapsed_seconds=time.monotonic() - begun, phase=category, result_status=result_category,
                  journal_valid=journal_valid, confirmed_started=starts)
    record.update(retain_stream(private, diagnostics / (label + '-child.log'),
                               {i['test'].rsplit('.', 1)[-1] for i in collection if i['suite'] == name}))
    (diagnostics / (label + '-events.jsonl')).write_text(''.join(json.dumps(item, sort_keys=True) + '\n' for item in events), encoding='utf-8')
    summary = dict(suite=name, partition=label, results=records,
                   tests_run=result['tests_run'] if result else 'UNKNOWN',
                   success=bool(result and result['success'] and code == 0 and journal_valid and result_category == 'RESULT_VALID'),
                   real_socket_attempts=result['real_socket_attempts'] if result else 'UNKNOWN',
                   exit_code=record['exit_code'], result_status=result_category,
                   confirmed_started=starts, journal_outcomes=observed)
    write_json(diagnostics / (label + '-receipt.json'), record)
    write_json(diagnostics / (label + '-outcomes.json'), summary)
    for owned in (private,):
        owned.unlink(missing_ok=True)
    return summary, record


def combined_report(output, collection, partitions, metadata, verify_identities):
    suites = []
    for result, _ in partitions:
        target = next((s for s in suites if s['suite'] == result['suite']), None)
        if target is None:
            target = dict(suite=result['suite'], results=[], confirmed_started=[], success=True, real_socket_attempts=0, exit_code=0)
            suites.append(target)
        target['results'] += result['results']
        target['confirmed_started'] += result['confirmed_started']
        target['success'] = target['success'] and result['success'] and result['result_status'] == 'RESULT_VALID'
        target['exit_code'] = target['exit_code'] or result['exit_code']
        sockets = result['real_socket_attempts']
        target['real_socket_attempts'] = target['real_socket_attempts'] + sockets if type(target['real_socket_attempts']) is int and type(sockets) is int else 'UNKNOWN'
    errors = []
    try:
        verify_identities(suites, collection)
    except AssertionError:
        errors.append('ACCEPTANCE_IDENTITIES_CHANGED')
    expected = [case_key(i['suite'], i['test']) for i in collection]
    observed = [case_key(s['suite'], i['test']) for s in suites for i in s['results']]
    completed = set(observed) & set(expected)
    started = {case_key(s['suite'], n) for s in suites for n in s['confirmed_started']} | completed
    if len(observed) != len(expected):
        errors.append('ACCEPTANCE_EXECUTION_CHANGED')
    complete = not errors and all(r['result_status'] == 'RESULT_VALID' for r, _ in partitions)
    success = complete and all(s['success'] and s['exit_code'] == 0 and s['real_socket_attempts'] == 0 for s in suites) and all(i['status'] in ('PASS', 'SKIP') for s in suites for i in s['results'])
    sockets = sum(s['real_socket_attempts'] for s in suites) if complete and all(type(s['real_socket_attempts']) is int for s in suites) else 'UNKNOWN'
    combined = dict(metadata, collection_count=len(collection), collected=collection, commands=[r for _, r in partitions],
                    tests_run=len(observed) if complete else 'UNKNOWN', completed_count=len(completed),
                    started_confirmed_count=len(started & set(expected)), started_count_exact=complete,
                    unconfirmed=[i for i in collection if case_key(i['suite'], i['test']) not in completed],
                    execution_complete=complete, success=success, identity_errors=errors, real_socket_attempts=sockets, suites=suites)
    report = ET.Element('testsuites')
    for name in dict.fromkeys(i['suite'] for i in collection):
        selected = [i for i in collection if i['suite'] == name]
        cases = [i for s in suites if s['suite'] == name for i in s['results']]
        node = ET.SubElement(report, 'testsuite', name=name, tests=str(len(selected)))
        for item in selected:
            matches = [i for i in cases if case_key(name, i['test']) == case_key(name, item['test'])]
            case = ET.SubElement(node, 'testcase', name=item['test'], classname=name)
            if len(matches) != 1:
                ET.SubElement(case, 'error', type='UNCONFIRMED_OR_DUPLICATE').text = 'No unique confirmed outcome; execution is not assumed.'
            elif matches[0]['status'] == 'SKIP':
                ET.SubElement(case, 'skipped').text = 'Preserved platform-specific result; details withheld.'
            elif matches[0]['status'] != 'PASS':
                ET.SubElement(case, 'failure', type=matches[0]['status']).text = 'Synthetic child reported a failure; see sanitized receipt.'
        node.set('errors', str(len(node.findall('testcase/error'))))
        node.set('failures', str(len(node.findall('testcase/failure'))))
        node.set('skipped', str(len(node.findall('testcase/skipped'))))
    if not success:
        gate = ET.SubElement(report, 'testsuite', name='qualification_gate', tests='1', failures='1')
        ET.SubElement(ET.SubElement(gate, 'testcase', name='complete_exact_once_success'), 'failure').text = 'Incomplete, invalid, nonzero or unsuccessful child qualification.'
    write_json(output / 'combined.json', combined)
    temporary = output / 'combined.xml.tmp'
    ET.ElementTree(report).write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(output / 'combined.xml')
    return combined
