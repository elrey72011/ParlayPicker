import shutil
import subprocess
import pytest


def _run_node(command, *, timeout=15, spawn=subprocess.Popen):
    """Own one synthetic child, preserve its failure, and reap it on all exits."""
    def bounded(value):
        value = value or ''
        return value if len(value) <= 4096 else value[:2048] + '\n[... truncated ...]\n' + value[-2048:]
    def diagnostic(code, stdout, stderr):
        return (f'Public-refresh Node regression {code}\n'
                f'Captured stdout: {bounded(stdout)}\n'
                f'Captured stderr: {bounded(stderr)}')
    with spawn(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except BaseException as error:
            child.kill()
            stdout, stderr = child.communicate(timeout=5)
            if isinstance(error, subprocess.TimeoutExpired):
                raise AssertionError(diagnostic(f'timed out after {timeout}s; child reaped', stdout, stderr)) from error
            error.add_note(diagnostic(f'interrupted; child reaped with exit {child.returncode}', stdout, stderr))
            raise
        if child.returncode:
            raise AssertionError(diagnostic(f'exited {child.returncode}', stdout, stderr))
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def _node():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser transport regression')
    return node


def test_public_refresh_transport():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser transport regression')
    _run_node([node, 'tests/public_refresh.cjs'])


@pytest.mark.parametrize('control', ['delayed-fetch-hash', 'rejections', 'cleanup'])
def test_public_refresh_completion_and_owned_cleanup(control):
    _run_node([_node(), 'tests/public_refresh.cjs', '--control', control])


@pytest.mark.parametrize('control, expected', [('never-fetch', 'phase=version-fetch'), ('never-hash', 'phase=hash'), ('assertion', 'deliberate synthetic JavaScript assertion')])
def test_public_refresh_deliberate_child_failure_is_preserved(control, expected):
    with pytest.raises(AssertionError) as caught:
        _run_node([_node(), 'tests/public_refresh.cjs', '--control', control])
    message = str(caught.value)
    assert 'exited 1' in message and expected in message
    assert 'Captured stderr:' in message and ('at ' in message)
    if control != 'assertion':
        assert 'cleanup=complete' in message


@pytest.mark.parametrize('interruption', ['timeout', 'keyboard'])
def test_public_refresh_interruption_reaps_owned_child(interruption):
    children = []
    def spawn(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        children.append(child)
        if interruption == 'keyboard':
            communicate = child.communicate
            first = True
            def interrupt_once(*a, **kw):
                nonlocal first
                if first:
                    first = False
                    raise KeyboardInterrupt('synthetic interruption')
                return communicate(*a, **kw)
            child.communicate = interrupt_once
        return child
    expected = AssertionError if interruption == 'timeout' else KeyboardInterrupt
    with pytest.raises(expected) as caught:
        _run_node([_node(), '-e', 'setInterval(()=>{},1000)'], timeout=.1, spawn=spawn)
    assert len(children) == 1 and children[0].poll() is not None
    assert 'reaped' in (str(caught.value) + str(getattr(caught.value, '__notes__', [])))


def test_public_refresh_child_diagnostics_are_bounded():
    code = 'process.stdout.write("x".repeat(9000));process.stderr.write("y".repeat(9000));process.exitCode=7'
    with pytest.raises(AssertionError) as caught:
        _run_node([_node(), '-e', code])
    message = str(caught.value)
    assert 'exited 7' in message and len(message) < 8400
    assert message.count('x') <= 4098 and message.count('y') <= 4096
