"""Owned short Chrome transport temporary root; unrelated to pytest evidence."""
from contextlib import contextmanager
from pathlib import Path
import os,stat,tempfile

SOCKET_SUFFIX='/com.google.Chrome.XXXXXX/SingletonSocket'
SAFE_SOCKET_BYTES=104

def verify_budget(path):
    if len(os.fsencode(str(Path(path).resolve())+SOCKET_SUFFIX))>=SAFE_SOCKET_BYTES:
        raise ValueError('CHROME_TRANSPORT_TMP_PATH_TOO_LONG')

def protection(path):
    value=Path(path).lstat()
    private=stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode)
    if os.name=='posix':private=private and value.st_uid==os.geteuid() and stat.S_IMODE(value.st_mode)==0o700
    if not private:raise ValueError('CHROME_TRANSPORT_TMP_NOT_PRIVATE')
    return dict(private_directory_verified=True,posix_mode=oct(stat.S_IMODE(value.st_mode)) if os.name=='posix' else 'NOT_POSIX',
        current_uid_owned=value.st_uid==os.geteuid() if os.name=='posix' else 'NOT_POSIX',
        socket_path_bytes=len(os.fsencode(str(Path(path).resolve())+SOCKET_SUFFIX)),socket_budget_exclusive=SAFE_SOCKET_BYTES)

@contextmanager
def short_transport_tmp(parent='/tmp'):
    with tempfile.TemporaryDirectory(prefix='pc-',dir=parent) as value:
        root=Path(value).resolve();verify_budget(root);protection(root)
        yield root
