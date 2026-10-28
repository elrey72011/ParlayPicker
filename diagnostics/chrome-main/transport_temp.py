"""Owned short Chrome transport temporary root; unrelated to pytest evidence."""
from contextlib import contextmanager
from pathlib import Path
import os,tempfile

SOCKET_SUFFIX='/com.google.Chrome.XXXXXX/SingletonSocket'
SAFE_SOCKET_BYTES=104

def verify_budget(path):
    if len(os.fsencode(str(Path(path).resolve())+SOCKET_SUFFIX))>=SAFE_SOCKET_BYTES:
        raise ValueError('CHROME_TRANSPORT_TMP_PATH_TOO_LONG')

@contextmanager
def short_transport_tmp(parent='/tmp'):
    with tempfile.TemporaryDirectory(prefix='pc-',dir=parent) as value:
        root=Path(value).resolve();verify_budget(root)
        yield root
