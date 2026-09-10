"""Publish a validated board to a dedicated cPanel subdomain using pinned SFTP."""
import base64
from contextlib import contextmanager
import hashlib
import hmac
import io
import re
import stat
from urllib.parse import urlsplit
from uuid import uuid4

import requests
from scripts.publish_board import render

LIMIT = 10_000_000


def configuration(setting):
    get = lambda name: str(setting('PARLAYPICKER_SFTP_' + name) or '').strip()
    config = {name.lower(): get(name) for name in ('HOST', 'USER', 'DIRECTORY', 'HOST_KEY_SHA256')}
    config['password'] = str(setting('PARLAYPICKER_SFTP_PASSWORD') or '')
    config['url'] = str(setting('PARLAYPICKER_PUBLIC_URL') or '').strip().rstrip('/')
    parsed = urlsplit(config['url'])
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.port or parsed.path or parsed.query or parsed.fragment):
        raise ValueError('Set PARLAYPICKER_PUBLIC_URL to the HTTPS subdomain URL without a path.')
    if not re.fullmatch(r'[A-Za-z0-9.-]+', config['host']):
        raise ValueError('Set PARLAYPICKER_SFTP_HOST to your hosting server hostname.')
    if not re.fullmatch(r'[A-Za-z0-9_]+', config['user']):
        raise ValueError('Set PARLAYPICKER_SFTP_USER to your cPanel username.')
    expected = '/home/' + config['user'] + '/' + parsed.hostname
    if config['directory'] != expected or len(parsed.hostname.split('.')) < 3:
        raise ValueError('SFTP directory must be the dedicated /home/USERNAME/SUBDOMAIN document root.')
    if not re.fullmatch(r'SHA256:[A-Za-z0-9+/]{43}', config['host_key_sha256']):
        raise ValueError('Set the SHA256 SSH host-key fingerprint verified with Namecheap support.')
    if not config['password']:
        raise ValueError('Set PARLAYPICKER_SFTP_PASSWORD in Streamlit secrets.')
    config['port'] = int(get('PORT') or '21098')
    if not 1 <= config['port'] <= 65535:
        raise ValueError('Invalid SFTP port.')
    return config


def check_host_key(key, expected):
    actual = 'SHA256:' + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
    if not hmac.compare_digest(actual, expected):
        raise ValueError('SSH host key does not match the configured fingerprint. Verify it with Namecheap.')


@contextmanager
def connection(config):
    import paramiko
    class PinnedKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            check_host_key(key, config['host_key_sha256'])
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedKey())
    try:
        client.connect(config['host'], port=config['port'], username=config['user'],
                       password=config['password'], timeout=10, banner_timeout=10,
                       auth_timeout=10, allow_agent=False, look_for_keys=False)
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(30)
            directory = config['directory']
            if sftp.normalize(directory) != directory or not stat.S_ISDIR(sftp.lstat(directory).st_mode):
                raise ValueError('The configured document root must be a real, dedicated directory.')
            yield sftp
    except ValueError:
        raise
    except Exception:
        raise RuntimeError('SFTP connection or upload failed. Check hosting credentials, host key and directory; no automatic retry was made.') from None
    finally:
        client.close()


def public_bytes(config):
    try:
        with requests.get(config['url'] + '/?publication_check=' + uuid4().hex,
                          timeout=(10, 30), allow_redirects=False, stream=True,
                          headers={'Cache-Control': 'no-cache'}) as response:
            if response.status_code != 200:
                raise RuntimeError('The HTTPS public page did not return HTTP 200.')
            chunks = []
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > LIMIT:
                    raise RuntimeError('Public page exceeds the verification limit.')
                chunks.append(chunk)
            return b''.join(chunks)
    except requests.RequestException:
        raise RuntimeError('Public HTTPS verification failed. Check DNS and the SSL certificate; certificate checks are never bypassed.') from None


def site_info(config):
    with connection(config):
        pass
    public_bytes(config)
    return {'url': config['url']}


def prepare(package):
    content = render(package).encode('utf-8')
    if len(content) > LIMIT:
        raise ValueError('Public board exceeds the 10 MB upload limit.')
    return content, 'sftp-' + hashlib.sha256(content).hexdigest()


def deploy(content, deploy_id, config):
    if deploy_id != 'sftp-' + hashlib.sha256(content).hexdigest():
        raise ValueError('Publication content mismatch.')
    with connection(config) as sftp:
        directory = config['directory']
        temporary = directory + '/.parlaypicker-' + uuid4().hex + '.tmp'
        try:
            sftp.putfo(io.BytesIO(content), temporary, file_size=len(content), confirm=True)
            sftp.chmod(temporary, 0o644)
            # Atomic replacement; never delete the live page as a fallback.
            sftp.posix_rename(temporary, directory + '/index.html')
        finally:
            try:
                sftp.remove(temporary)
            except OSError:
                pass
    return {'id': deploy_id, 'state': 'uploaded', 'url': config['url']}


def deployment_status(deploy_id, config):
    if not re.fullmatch(r'sftp-[0-9a-f]{64}', deploy_id):
        raise ValueError('Invalid SFTP publication identifier.')
    actual = 'sftp-' + hashlib.sha256(public_bytes(config)).hexdigest()
    return {'id': deploy_id, 'url': config['url'],
            'state': 'ready' if actual == deploy_id else 'content_mismatch'}
