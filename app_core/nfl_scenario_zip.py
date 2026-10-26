"""Versioned, in-memory static replay of explicitly trusted completed results.

This never calls the live reader/predictor, edits recorded bindings, or admits
an unknown self-sealed result. Production trust comes from the reviewed catalog,
not the uploaded manifest. Synthetic policies can admit SYNTHETIC fixtures only.
"""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import struct
import zlib
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED, ZIP_STORED

VERSION = 'nfl-scenario-static-zip-reader-v1'
TRUST_VERSION = 'nfl-scenario-static-replay-trust-v1'
TRUST_SHA256 = 'ce43edf51b58e9c71c2105803e9c8acbc07f9dff60e454943639c251dc9adff5'
TRUST_PATH = Path(__file__).resolve().parents[1] / 'docs/research/nfl-scenario-static-replay-bindings-v1.json'
MAX_COMPRESSED = 4 * 1024 * 1024
MAX_DECOMPRESSED = 8 * 1024 * 1024
MAX_BATCH = 8 * 1024 * 1024
MAX_SELECTED = 4
MEMBERS = {'manifest.json', 'private-receipt.json', 'scenario-result.json'}
LIMITS = {'manifest.json': 16384, 'private-receipt.json': 16384, 'scenario-result.json': 6 * 1024 * 1024}
LABELS = ['OWNER_REVIEWED', 'PRIVATE_RESEARCH', 'USER-SPECIFIED SCENARIO']
INPUT_FIELDS = set('version evidence_label scoring event event_verification scenario permission features feature_order feature_available_at model reader owner_review original_inference_time'.split())
RESULT_FIELDS = set('version status labels input inference_time completed_at reader model native_output probability probability_stage calibrated probability_semantics publisher_availability source_acceptance wagering_authority qualification wager_action stake'.split())
FINDINGS = set('event_mapping scenario_definition permissions clocks_and_original_custody feature_provenance model_binding'.split())


class ReplayError(ValueError):
    def __init__(self, code, field):
        self.code, self.field = 'NFL_SCENARIO_ZIP_' + code, field
        super().__init__(self.code)


def require(ok, code, field):
    if not ok:
        raise ReplayError(code, field)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'DUPLICATE_JSON_KEY', 'json')
        out[key] = value
    return out


def json_value(raw):
    def invalid(value):
        raise ReplayError('INVALID_JSON', 'json')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def production_trust():
    raw = TRUST_PATH.read_bytes()
    require(sha(raw) == TRUST_SHA256, 'TRUST_CHANGED', 'trusted_catalog')
    policy = json_value(raw)
    require(set(policy) == {'version', 'purpose', 'profiles', 'records'}
            and policy['version'] == TRUST_VERSION
            and policy['purpose'] == 'PRIVATE_RESEARCH_STATIC_REPLAY', 'TRUST_CHANGED', 'trusted_catalog')
    return policy


@dataclass(frozen=True)
class SyntheticPolicy:
    """Explicit CI trust fixture, never an admission path for retained originals.

    The UI does not construct this from uploads, environment variables or URLs.
    Tests supply independently constructed synthetic receipt anchors and use the
    same parser and binding checks. Production always uses the reviewed catalog.
    """
    records_json: str


def first_difference(actual, expected, path):
    if type(actual) is not type(expected):
        return path
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            return path
        for key in sorted(expected):
            different = first_difference(actual[key], expected[key], path + '.' + key)
            if different:
                return different
    elif isinstance(expected, list):
        if actual != expected:
            return path
    elif actual != expected:
        return path
    return None


def clock(value, field):
    require(isinstance(value, str) and re.search(r'(Z|[+-]\d\d:\d\d)$', value), 'CLOCK', field)
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(stamp.tzinfo is not None, 'CLOCK', field)
    return stamp


def _authenticate(result, receipt, policy, synthetic_policy):
    require(isinstance(result, dict) and set(result) == {'payload', 'sha256'}
            and isinstance(result['payload'], dict), 'SCHEMA', 'result')
    p = result['payload']
    require(digest(p) == result['sha256'], 'HASH_MISMATCH', 'result.sha256')
    require(isinstance(receipt, dict) and set(receipt) == {'version', 'export_id', 'result_sha256', 'input_sha256'}
            and receipt['version'] == 'nfl-private-scenario-capture-v1'
            and all(isinstance(receipt[k], str) and re.fullmatch('[0-9a-f]{64}', receipt[k]) for k in ('result_sha256', 'input_sha256'))
            and receipt['export_id'] == 'nfl-scenario-result:' + receipt['result_sha256'], 'RECEIPT_CONFLICT', 'receipt')
    require(receipt['result_sha256'] == result['sha256'], 'RECEIPT_CONFLICT', 'receipt.result_sha256')
    require(set(p) == RESULT_FIELDS, 'SCHEMA', 'result.payload')
    packet = p['input']
    require(isinstance(packet, dict) and set(packet) == {'payload', 'sha256'} and isinstance(packet['payload'], dict), 'SCHEMA', 'input')
    q = packet['payload']
    require(set(q) == INPUT_FIELDS and q['version'] == 'nfl-owner-sporting-scenario-v1'
            and q['original_inference_time'] is None, 'SCHEMA', 'input.payload')
    require(digest(q) == packet['sha256'] == receipt['input_sha256'], 'HASH_MISMATCH', 'input.sha256')
    if synthetic_policy is None:
        records, label = policy['records'], 'RETAINED'
    else:
        require(type(synthetic_policy) is SyntheticPolicy, 'TRUST_CHANGED', 'synthetic_policy')
        records, label = json_value(synthetic_policy.records_json.encode()), 'SYNTHETIC'
    require(q['evidence_label'] == label, 'EVIDENCE_LABEL', 'input.evidence_label')
    anchor = records.get(result['sha256'])
    require(isinstance(anchor, dict) and set(anchor) == {'input_sha256', 'receipt_sha256', 'profile'}, 'UNTRUSTED_RESULT', 'result.sha256')
    require(anchor['input_sha256'] == receipt['input_sha256'] and anchor['receipt_sha256'] == digest(receipt), 'RECEIPT_CONFLICT', 'receipt')
    profile = policy['profiles'].get(anchor['profile'])
    require(isinstance(profile, dict), 'UNSUPPORTED_BINDING', 'producer_profile')
    for key in ('reader', 'model'):
        different = first_difference(p[key], profile[key], 'payload.' + key)
        require(different is None, 'UNSUPPORTED_BINDING', different or key)
        require(q[key] == p[key], 'UNSUPPORTED_BINDING', 'input.' + key)
    require(p['version'] == 'nfl-owner-sporting-scenario-result-v1' and p['status'] == 'COMPLETE'
            and p['labels'] == LABELS and p['probability_stage'] == 'native_model_output'
            and p['calibrated'] is False and p['source_acceptance'] is False
            and p['wagering_authority'] is False and p['qualification'] is False
            and p['wager_action'] == 'PASS' and type(p['stake']) is int and p['stake'] == 0
            and p['publisher_availability'] == 'UNKNOWN'
            and p['probability_semantics'] == 'full_game_including_overtime_cover_conditional_on_completed_sporting_result', 'AUTHORITY_FORBIDDEN', 'result.payload')
    output = p['native_output']
    require(isinstance(output, dict) and set(output) == set('ml_probability ml_probability_source ml_target ml_projection ml_residual_scale'.split())
            and output['ml_probability_source'] == 'score-distribution-v1:nfl'
            and output['ml_target'] == 'spread_cover', 'SCHEMA', 'native_output')
    require(type(p['probability']) is float and math.isfinite(p['probability']) and .2 <= p['probability'] <= .8
            and p['probability'] == output['ml_probability'], 'PROBABILITY_CONFLICT', 'probability')
    require(all(type(output[k]) in (int, float) and math.isfinite(output[k]) for k in ('ml_projection', 'ml_residual_scale'))
            and output['ml_residual_scale'] > 0, 'SCHEMA', 'native_output')
    event, assumption, review = q['event'], q['scenario'], q['owner_review']
    require(set(event) == set('canonical_event_id home away start season neutral_site'.split())
            and event['neutral_site'] is False and event['home'] != event['away'], 'SCHEMA', 'event')
    require(set(assumption) == set('selected_team signed_handicap period overtime condition'.split())
            and assumption['selected_team'] in {event['home'], event['away']}
            and type(assumption['signed_handicap']) in (int, float)
            and math.isfinite(assumption['signed_handicap']) and abs(assumption['signed_handicap'] % 1) == .5
            and assumption['period'] == 'full_game' and assumption['overtime'] is True
            and assumption['condition'] == 'completed_sporting_result', 'SCHEMA', 'scenario')
    require(q['feature_order'] == profile['model']['feature_order'] and set(q['features']) == set(q['feature_order']), 'SCHEMA', 'features')
    require(review['version'] == 'nfl-owner-sporting-scenario-review-v1' and review['status'] == 'OWNER_REVIEWED'
            and review['purpose'] == 'PRIVATE_RESEARCH' and review['reviewer'] == 'Robert Velarde'
            and review['attestation'] == 'I verified the original evidence and sporting assumptions for this exact scenario.'
            and review['subject_sha256'] == digest({k:v for k,v in q.items() if k != 'owner_review'})
            and set(review['findings']) == FINDINGS
            and all(v['conclusion'] == 'VERIFIED' and isinstance(v['basis'], str) and len(v['basis'].strip()) >= 16 for v in review['findings'].values()), 'REVIEW_CONFLICT', 'owner_review')
    inference = clock(p['inference_time'], 'inference_time')
    require(inference <= clock(p['completed_at'], 'completed_at') < clock(event['start'], 'event.start'), 'CLOCK', 'original_inference_interval')
    require(clock(q['feature_available_at'], 'feature_available_at') <= clock(review['reviewed_at'], 'owner_review.reviewed_at') < inference, 'CLOCK', 'owner_review.reviewed_at')
    # Whole-result and input anchors above bind the complete previously validated
    # histories, permissions, review and original transport bytes. Recomputing or
    # re-signing a different result cannot register a new production identity.
    return dict(payload=deepcopy(p), receipt=deepcopy(receipt), result_sha256=result['sha256'],
                input_sha256=packet['sha256'], profile_id=anchor['profile'], evidence_label=label,
                replay_version=VERSION)


def read_archive(raw, *, synthetic_policy=None):
    """Authenticate existing export bytes in memory. No DB, network or predictor."""
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_COMPRESSED, 'COMPRESSED_LIMIT', 'archive')
    try:
        require(len(raw) >= 22 and raw[:4] == b'PK\x03\x04' and raw[-22:-18] == b'PK\x05\x06', 'MALFORMED', 'archive')
        _, disk, directory_disk, disk_count, total_count, directory_size, directory_offset, comment_size = struct.unpack('<4s4H2LH', raw[-22:])
        require(disk == directory_disk == comment_size == 0 and disk_count == total_count == 3
                and directory_offset + directory_size == len(raw) - 22, 'MEMBERS', 'archive.directory')
        with ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            require(len(entries) == 3 and len({x.filename for x in entries}) == 3
                    and {x.filename for x in entries} == MEMBERS, 'MEMBERS', 'archive.members')
            require(not archive.comment, 'MEMBERS', 'archive.comment')
            require(sum(x.file_size for x in entries) <= MAX_DECOMPRESSED, 'DECOMPRESSED_LIMIT', 'archive')
            parts = {}
            for item in entries:
                require(not item.is_dir() and not item.flag_bits & 1 and not item.extra and not item.comment
                        and item.compress_type in (ZIP_STORED, ZIP_DEFLATED)
                        and (item.external_attr >> 16) & 0o170000 != 0o120000, 'MEMBERS', 'archive.members')
                limit = LIMITS[item.filename]
                require(0 < item.file_size <= limit and item.file_size <= 64 * max(item.compress_size, 1), 'DECOMPRESSED_LIMIT', 'archive.members')
                with archive.open(item) as stream:
                    data = stream.read(limit + 1)
                    require(len(data) == item.file_size and len(data) <= limit and not stream.read(1), 'DECOMPRESSED_LIMIT', 'archive.members')
                parts[item.filename] = data
        manifest = json_value(parts['manifest.json'])
        require(isinstance(manifest, dict) and set(manifest) == {'version', 'purpose', 'files'}
                and manifest['version'] == 'nfl-private-scenario-export-v1'
                and manifest['purpose'] == 'PRIVATE_RESEARCH'
                and isinstance(manifest['files'], dict)
                and set(manifest['files']) == {'private-receipt.json', 'scenario-result.json'}, 'SCHEMA', 'manifest')
        require(all(manifest['files'][name] == sha(parts[name]) for name in manifest['files']), 'HASH_MISMATCH', 'manifest.files')
        item = _authenticate(json_value(parts['scenario-result.json']), json_value(parts['private-receipt.json']), production_trust(), synthetic_policy)
        item['archive_sha256'] = sha(raw)
        return item
    except ReplayError:
        raise
    except (BadZipFile, OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError, NotImplementedError, EOFError, zlib.error, struct.error):
        raise ReplayError('MALFORMED', 'archive') from None


def read_archives(raw_files, *, synthetic_policy=None):
    require(isinstance(raw_files, list) and 0 < len(raw_files) <= MAX_SELECTED, 'SELECTION_LIMIT', 'selection')
    require(all(isinstance(x, bytes) for x in raw_files) and sum(map(len, raw_files)) <= MAX_BATCH, 'COMPRESSED_LIMIT', 'selection')
    items = [read_archive(raw, synthetic_policy=synthetic_policy) for raw in raw_files]
    require(len({x['receipt']['export_id'] for x in items}) == len(items), 'RECEIPT_CONFLICT', 'selection')
    return items
