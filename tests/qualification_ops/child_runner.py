"""Journal synthetic unittest lifecycle before an abrupt child can exit.

Only test identifiers and fixed lifecycle enums are retained, never exceptions,
arguments, environment values, fixture bodies, or credentials.
"""
import json
from pathlib import Path
import runpy
import sys
import unittest


def main():
    journal = Path(sys.argv[1])
    script = Path(sys.argv[2])
    with journal.open('x', encoding='utf-8', buffering=1) as stream:
        def emit(event, test=None, status=None, exception_class=None):
            record = {'event': event}
            if test is not None:
                record['test'] = test.id()
            if status is not None:
                record['status'] = status
            if exception_class is not None:
                record['exception_class'] = exception_class
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()

        emit('CHILD_STARTED')
        for method, event, status in (
            ('startTest', 'CASE_STARTED', None),
            ('addSuccess', 'CASE_COMPLETED', 'PASS'),
            ('addFailure', 'CASE_COMPLETED', 'FAIL'),
            ('addError', 'CASE_COMPLETED', 'ERROR'),
            ('addSkip', 'CASE_COMPLETED', 'SKIP'),
            ('addExpectedFailure', 'CASE_COMPLETED', 'ERROR'),
            ('addUnexpectedSuccess', 'CASE_COMPLETED', 'ERROR'),
        ):
            original = getattr(unittest.TestResult, method)

            def observed(self, test, *args, _original=original,
                         _event=event, _status=status, **kwargs):
                error = args[0] if _status in ('FAIL', 'ERROR') and args else None
                error_class = type(error[1]).__name__ if isinstance(error, tuple) and len(error) == 3 else None
                emit(_event, test, _status, error_class)
                return _original(self, test, *args, **kwargs)

            setattr(unittest.TestResult, method, observed)
        sys.argv = [str(script), *sys.argv[3:]]
        try:
            runpy.run_path(str(script), run_name='__main__')
        except BaseException as exc:
            if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
                emit('CHILD_FAILURE', exception_class=type(exc).__name__)
            raise
        finally:
            emit('CHILD_RETURNED')


if __name__ == '__main__':
    main()
