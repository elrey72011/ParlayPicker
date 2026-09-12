from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

from scripts.run_ci_tests import FileShard


def test_shards_partition_every_collected_item_and_preserve_file_order():
    original = [SimpleNamespace(path=Path(name), nodeid=f'{name}::{i}')
                for name in ('tests/test_z.py', 'tests/test_a.py', 'tests/test_m.py')
                for i in range(3)]
    groups = []
    for shard in range(2):
        deselected = []
        hook = SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items))
        items = original.copy()
        FileShard(shard, 2).pytest_collection_modifyitems(SimpleNamespace(hook=hook), items)
        assert len(items) + len(deselected) == len(original)
        assert items == [item for item in original if item in items]
        groups.append(items)
    ids = [item.nodeid for group in groups for item in group]
    assert sorted(ids) == sorted(item.nodeid for item in original)
    assert len(ids) == len(set(ids))
    assert not {item.path for item in groups[0]} & {item.path for item in groups[1]}


def test_runner_propagates_failure_and_rejects_invalid_shards(tmp_path):
    runner = Path(__file__).resolve().parents[1] / 'scripts/run_ci_tests.py'
    (tmp_path / 'test_a.py').write_text('def test_fails(): assert False\n')
    (tmp_path / 'test_b.py').write_text('def test_passes(): assert True\n')
    def run(shard):
        return subprocess.run([sys.executable, str(runner), '--shard', str(shard), '--shards', '2',
                               '-q', str(tmp_path)], capture_output=True, text=True)
    failed, passed, invalid = run(1), run(2), run(0)
    assert failed.returncode == 1 and '1 failed, 1 deselected' in failed.stdout
    assert passed.returncode == 0 and '1 passed, 1 deselected' in passed.stdout
    assert invalid.returncode == 2 and 'Require 1 <= shard <= shards' in invalid.stderr
