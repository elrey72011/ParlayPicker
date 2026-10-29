"""Fixed, owner-only TLS observations. No connection or trust-setting changes."""
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
import sys

BUNDLE = '/etc/ssl/certs/ca-certificates.crt'
MAPS = '/proc/self/maps'
MAX_BUNDLE = 4 * 1024 * 1024
MAX_MAPS = 2 * 1024 * 1024
MAX_ELF = 8 * 1024 * 1024
PUBLIC_PATHS = {BUNDLE, '/etc/ssl/certs', '/usr/lib/ssl/cert.pem',
                '/usr/lib/ssl/certs', '/etc/ssl/cert.pem'}
ENV_NAMES = ('SSL_CERT_FILE', 'SSL_CERT_DIR', 'OPENSSL_CONF', 'OPENSSL_MODULES',
             'PGSSLROOTCERT')
LIBRARY = re.compile(r'lib(pq|ssl|crypto)(?:-[0-9a-f]+)?\.so(?:\.[0-9]+)*')


def _path(value):
    # Do not export or open paths supplied in environment/settings.
    if value is None:
        return 'UNSET'
    return value if type(value) is str and value in PUBLIC_PATHS else 'OTHER_REDACTED'


def _environment():
    result = {}
    for name in ENV_NAMES:
        value = os.environ.get(name)
        result[name] = {'present': value is not None}
        if name in ('SSL_CERT_FILE', 'SSL_CERT_DIR', 'PGSSLROOTCERT'):
            result[name]['value_class'] = 'system' if value == 'system' else _path(value)
    return result


def _maps():
    if sys.platform != 'linux':
        return None
    with open(MAPS, 'rb') as stream:
        raw = stream.read(MAX_MAPS + 1)
    if len(raw) > MAX_MAPS:
        raise ValueError('MAP_LIMIT')
    rows = {}
    for line in raw.decode('utf-8', 'strict').splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith('/') or parts[5].endswith(' (deleted)'):
            continue
        path = parts[5]
        name = Path(path).name
        # Internal paths are retained only for fixed, already-mapped libraries or
        # the two loaded extension anchors. They are never returned to the UI.
        if not (LIBRARY.fullmatch(name) or re.fullmatch(r'(?:pq|_ssl)\.[A-Za-z0-9_-]+\.so', name)):
            continue
        device = parts[3].split(':')
        if len(device) != 2 or not all(re.fullmatch('[0-9a-f]+', x) for x in device) or not parts[4].isdigit():
            continue
        rows[path] = {'path': path, 'name': name, 'device': [int(x, 16) for x in device],
                      'inode': int(parts[4])}
    if len(rows) > 32:
        raise ValueError('MAP_LIMIT')
    return list(rows.values())


def _mapped_bytes(row):
    # Read only an object identified by this process's procfs mapping. Refuse
    # symlinks, special files and an inode/device changed since the observation.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(row['path'], flags)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ELF or
                info.st_ino != row['inode'] or
                [os.major(info.st_dev), os.minor(info.st_dev)] != row['device']):
            raise ValueError('MAPPED_OBJECT_CHANGED')
        raw = stream.read(MAX_ELF + 1)
    if len(raw) > MAX_ELF:
        raise ValueError('ELF_LIMIT')
    return raw


def _needed(row):
    """Bounded ELF DT_NEEDED reader; never executes a file or external command."""
    raw = _mapped_bytes(row)
    if raw[:4] != b'\x7fELF' or len(raw) < 64 or raw[5] not in (1, 2):
        raise ValueError('ELF_UNSUPPORTED')
    endian = '<' if raw[5] == 1 else '>'
    if raw[4] == 2:
        header = struct.unpack_from(endian + '16sHHIQQQIHHHHHH', raw)
        ph_format, dyn_format = endian + 'IIQQQQQQ', endian + 'qQ'
        offset, entry, count = header[5], header[9], header[10]
        indices = (2, 3, 5)
    elif raw[4] == 1:
        header = struct.unpack_from(endian + '16sHHIIIIIHHHHHH', raw)
        ph_format, dyn_format = endian + 'IIIIIIII', endian + 'iI'
        offset, entry, count = header[5], header[9], header[10]
        indices = (1, 2, 4)
    else:
        raise ValueError('ELF_UNSUPPORTED')
    if not 0 < count <= 128 or not struct.calcsize(ph_format) <= entry <= 128:
        raise ValueError('ELF_UNSUPPORTED')
    loads, dynamic = [], None
    for i in range(count):
        ph = struct.unpack_from(ph_format, raw, offset + i * entry)
        file_offset, address, size = (ph[j] for j in indices)
        if file_offset + size > len(raw):
            raise ValueError('ELF_TRUNCATED')
        if ph[0] == 1:
            loads.append((file_offset, address, size))
        elif ph[0] == 2:
            dynamic = (file_offset, size)
    if dynamic is None:
        raise ValueError('ELF_UNSUPPORTED')
    start, size = dynamic
    stride = struct.calcsize(dyn_format)
    if size // stride > 512:
        raise ValueError('ELF_LIMIT')
    needed, string_address = [], None
    for i in range(size // stride):
        tag, value = struct.unpack_from(dyn_format, raw, start + i * stride)
        if tag == 0:
            break
        if tag == 1:
            needed.append(value)
        elif tag == 5:
            string_address = value
    locations = [file_offset + string_address - address for file_offset, address, size in loads
                 if string_address is not None and address <= string_address < address + size]
    if len(locations) != 1 or len(needed) > 64:
        raise ValueError('ELF_UNSUPPORTED')
    names = []
    for index in needed:
        start = locations[0] + index
        end = raw.find(b'\0', start, start + 129)
        if end < 0 or start < 0:
            raise ValueError('ELF_UNSUPPORTED')
        name = raw[start:end].decode('ascii', 'strict')
        if LIBRARY.fullmatch(name):
            names.append(name)
    return names


def _dependency(row, rows, family):
    names = [n for n in _needed(row) if LIBRARY.fullmatch(n).group(1) == family]
    candidates = [r for r in rows if r['name'] in names]
    return candidates[0] if len(names) == len(candidates) == 1 else None


def _identity(row):
    if row is None:
        return 'UNKNOWN'
    return {'basename': row['name'], 'mapped_inode': row['inode'],
            'mapped_device': row['device'], 'source': 'actual_process_procfs'}


def _default_metadata(value, directory):
    # Metadata only for a compiled OpenSSL getter's conventional public trust
    # location. No environment-supplied path, contents or arbitrary file reader.
    result = {'exists': 'UNKNOWN', 'readable': 'UNKNOWN'}
    if (type(value) is not str or len(value) > 1024 or '\0' in value or
            not value.startswith(('/etc/', '/usr/', '/opt/')) or '..' in Path(value).parts):
        return result
    if not directory and Path(value).name not in ('cert.pem', 'ca-certificates.crt', 'ca-bundle.crt', 'root.crt'):
        return result
    try:
        info = os.stat(value)
        kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        result.update(exists=True, readable=kind and os.access(value, os.R_OK))
    except FileNotFoundError:
        result.update(exists=False, readable=False)
    except Exception:
        pass
    return result


def _crypto_defaults(row):
    """Read fixed getters from the already-mapped libcrypto; NOLOAD only."""
    import ctypes
    import _ctypes
    if not callable(getattr(_ctypes, 'dlclose', None)):
        return {'status': 'UNKNOWN'}
    if not hasattr(os, 'RTLD_NOLOAD') or row is None:
        return {'status': 'UNKNOWN'}
    # Revalidate identity before obtaining a NOLOAD handle. Never fall back to
    # loading a library, searching names or calling arbitrary exported symbols.
    _mapped_bytes(row)
    library = ctypes.CDLL(row['path'], mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    try:
        result = {'status': 'OBSERVED', 'source': 'psycopg_linked_libcrypto_fixed_getters',
                  'library': _identity(row), 'new_library_load_allowed': False}
        version = library.OpenSSL_version
        version.argtypes, version.restype = [ctypes.c_int], ctypes.c_char_p
        value = version(0)
        match = re.match(rb'^OpenSSL ([0-9]+\.[0-9]+\.[0-9]+[a-z]?)(?: |$)', value or b'')
        result['openssl_version'] = match[1].decode('ascii') if match else 'UNKNOWN'
        for field, symbol in (('default_ca_file', 'X509_get_default_cert_file'),
                              ('default_ca_directory', 'X509_get_default_cert_dir')):
            function = getattr(library, symbol)
            function.argtypes, function.restype = [], ctypes.c_char_p
            value = function()
            decoded = value.decode('ascii', 'strict') if value and len(value) <= 1024 else None
            result[field] = _path(decoded) if decoded is not None else 'UNKNOWN'
            result[field + '_metadata'] = _default_metadata(decoded, field.endswith('directory'))
        for field, symbol, expected in (
                ('ca_file_environment_supported', 'X509_get_default_cert_file_env', b'SSL_CERT_FILE'),
                ('ca_directory_environment_supported', 'X509_get_default_cert_dir_env', b'SSL_CERT_DIR')):
            function = getattr(library, symbol)
            function.argtypes, function.restype = [], ctypes.c_char_p
            result[field] = function() == expected
        return result
    finally:
        _ctypes.dlclose(library._handle)  # Release only this NOLOAD reference.



def _bundle(ssl):
    result = {'path': BUNDLE, 'exists': 'UNKNOWN', 'readable': 'UNKNOWN',
              'size_bytes': 'UNKNOWN', 'sha256': 'UNKNOWN',
              'parses_as_ca_bundle': 'UNKNOWN', 'certificate_count': 'UNKNOWN',
              'ca_certificate_count': 'UNKNOWN', 'parser': 'python_ssl_explicit_empty_context',
              'neon_certificate_validation': 'NOT_TESTED'}
    if sys.platform != 'linux':
        result['status'] = 'UNAVAILABLE_PLATFORM'
        return result
    try:
        info = os.stat(BUNDLE, follow_symlinks=False)
        result.update(exists=True, size_bytes=info.st_size)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BUNDLE:
            result['status'] = 'NONREGULAR_OR_OVERSIZE'
            return result
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(BUNDLE, flags)
        with os.fdopen(fd, 'rb') as stream:
            current = os.fstat(stream.fileno())
            if (current.st_dev, current.st_ino, current.st_size) != (info.st_dev, info.st_ino, info.st_size):
                raise ValueError('BUNDLE_CHANGED')
            raw = stream.read(MAX_BUNDLE + 1)
        if len(raw) > MAX_BUNDLE or len(raw) != info.st_size:
            raise ValueError('BUNDLE_CHANGED')
        result.update(readable=True)
    except FileNotFoundError:
        result.update(exists=False, readable=False, status='MISSING')
        return result
    except PermissionError:
        result.update(readable=False, status='UNREADABLE')
        return result
    except Exception:
        result['status'] = 'READ_UNAVAILABLE'
        return result
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cadata=raw.decode('ascii', 'strict'))
        stats = context.cert_store_stats()
        result.update(certificate_count=stats['x509'], ca_certificate_count=stats['x509_ca'],
                      parses_as_ca_bundle=stats['x509_ca'] > 0 and stats['x509_ca'] == stats['x509'],
                      status='PARSED')
        if result['parses_as_ca_bundle']:
            result['sha256'] = hashlib.sha256(raw).hexdigest()
    except Exception:
        result.update(parses_as_ca_bundle=False, status='PARSE_REJECTED')
    return result


def inspect_trust(access, psycopg):
    from app_core.nfl_private_workflow import OwnerAccess, _ISSUER, WorkflowError
    if type(access) is not OwnerAccess or access.proof is not _ISSUER or access.owner != 'Robert Velarde':
        raise WorkflowError('NFL_WORKFLOW_OWNER_REQUIRED')
    report = {'version': 'nfl-tls-trust-preflight-v1', 'network_calls': 0,
              'database_connections': 0, 'settings_changed': False,
              'environment': _environment(), 'loaded_libraries': 'UNKNOWN',
              'psycopg': {'implementation': 'UNKNOWN', 'libpq': 'UNKNOWN',
                          'ssl': 'UNKNOWN', 'crypto': 'UNKNOWN', 'trust_defaults': {'status': 'UNKNOWN'}},
              'neon_certificate_validation': 'NOT_TESTED'}
    pq = getattr(psycopg, 'pq', None)
    implementation = getattr(pq, '__impl__', None)
    if implementation in ('binary', 'c', 'python'):
        report['psycopg']['implementation'] = implementation
    try:
        rows = _maps()
        if rows is not None:
            report['loaded_libraries'] = [_identity(r) for r in rows if LIBRARY.fullmatch(r['name'])]
            name = {'binary': 'psycopg_binary.pq', 'c': 'psycopg_c.pq'}.get(implementation)
            module = sys.modules.get(name) if name else None
            anchor = getattr(module, '__file__', None)
            candidates = [r for r in rows if r['path'] == anchor]
            if len(candidates) == 1:
                libpq = _dependency(candidates[0], rows, 'pq')
                libssl = _dependency(libpq, rows, 'ssl') if libpq else None
                crypto = _dependency(libssl, rows, 'crypto') if libssl else None
                report['psycopg'].update(libpq=_identity(libpq), ssl=_identity(libssl), crypto=_identity(crypto))
                if crypto:
                    report['psycopg']['trust_defaults'] = _crypto_defaults(crypto)
    except Exception:
        # Retain partial nonsecret observations; no exception text or path escapes.
        report['driver_detail_status'] = 'UNKNOWN'
    import ssl
    defaults = ssl.get_default_verify_paths()
    version = re.match(r'^OpenSSL ([0-9]+\.[0-9]+\.[0-9]+[a-z]?)(?: |$)', ssl.OPENSSL_VERSION)
    report['python_ssl'] = {'source': 'python_ssl_not_assumed_equal_to_psycopg',
        'openssl_version': version[1] if version else 'UNKNOWN',
        'default_ca_file': _path(defaults.openssl_cafile),
        'default_ca_directory': _path(defaults.openssl_capath),
        'resolved_ca_file': _path(defaults.cafile), 'resolved_ca_directory': _path(defaults.capath)}
    report['ca_bundle'] = _bundle(ssl)
    report['scope'] = ('Loaded procfs identities and bounded ELF dependency association; fixed OpenSSL '
        'getters use NOLOAD. Bundle parsing uses Python SSL, not a Neon handshake or proof of driver selection.')
    return report
