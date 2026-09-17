import shutil
import subprocess
import pytest


def test_public_refresh_transport():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser transport regression')
    subprocess.run([node, 'tests/public_refresh.cjs'], check=True, capture_output=True, text=True)
