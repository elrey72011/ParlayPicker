"""Network-free prequote features and exact staged-original join. Never review.

The same frozen history selector and ordered feature names serve preparation
and the native predictor. Preparation records an actual clock, not availability.
"""
import base64
from copy import deepcopy
from datetime import timedelta
import hashlib
from pathlib import Path
import platform
from app_core import ncaaf_staged_acquisition as stages, ncaaf_history as history
from app_core import ncaaf_model_compatibility as model, ncaaf_research as research
from app_core import ncaaf_response_custody as custody

VERSION = 'ncaaf-prequote-history-features-v1'
BINDING_VERSION = 'ncaaf-staged-evidence-binding-v1'
require = model.require


def runtime():
    modules = (stages, history, model, research, custody)
    return dict(python=platform.python_version(), implementation=platform.python_implementation(), numpy=research.np.__version__,
        components={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for m in modules})


def batches(bundle):
    b = stages.read_bundle(bundle)
    require(b['version'] == stages.HISTORY_BUNDLE, 'NCAAF_STAGED_PARENT')
    result = []
    for obj in b['objects']:
        meta = obj['metadata']; q = meta['request_scope']
        req = dict(kind='games', year=q['year']) if meta['endpoint'] == 'games' else dict(kind='stats', year=q['year'], week=q['week'], season_type=q['seasonType'])
        raw = base64.b64decode(obj['body_b64'], validate=True)
        batch = dict(request=req, retrieved_at=meta['received_at'], records=history._clean(req['kind'], custody._parse(raw)))
        custody._project_cfbd(custody._parse(raw), meta, batch)
        result.append(batch)
    return result


def prepare(bundle, *, model_packet, prepared_at):
    if bundle.get('payload',{}).get('version')=='ncaaf-named-history-unaccepted-v2':
        from app_core.ncaaf_future_staged import prepare as selected
        return selected(bundle,model_packet=model_packet,prepared_at=prepared_at)
    b = stages.read_bundle(bundle); t = b['plan']['payload']['target']; now = history.timestamp(prepared_at)
    start = history.timestamp(t['start_utc']); receipts = [history.timestamp(o['metadata']['received_at']) for o in b['objects']]
    require(now is not None and all(receipts) and all(t <= now and (now-t).total_seconds() <= 86400 for t in receipts)
            and now < start, 'NCAAF_STAGED_INPUT_CLOCK')
    checked = model.load_model(model_packet)  # Original artifacts/runtime/lineage; no numerical execution.
    require(checked['compatibility']['family'] == 'spread', 'NCAAF_STAGED_MODEL_TARGET')
    data = batches(bundle); games = data[0]['records']
    # School IDs are not supplied by the name review or a future quote.
    candidates = [g for g in games if g.get('homeTeam') == 'Tulane' and g.get('awayTeam') == 'Memphis'
                  and history.timestamp(g.get('startDate')) == start]
    require(len(candidates) == 1, 'NCAAF_STAGED_CFBD_EVENT_CONFLICT')
    game = candidates[0]
    require(type(game.get('id')) is int and type(game.get('homeId')) is int and type(game.get('awayId')) is int
            and game['homeId'] != game['awayId'] and game.get('completed') is False and game.get('startTimeTBD') is False
            and type(game.get('neutralSite')) is bool and game['neutralSite'] == t['neutral_site'], 'NCAAF_STAGED_CFBD_EVENT_CONFLICT')
    # Exact reviewed target's provisional numbers must be confirmed, not silently
    # substituted if the newly captured source contradicts retained evidence.
    require((game['id'], game['homeId'], game['awayId']) == (401862801, 2655, 235), 'NCAAF_STAGED_PROVISIONAL_ID_CONFLICT')
    state = dict(schema=1, years=[start.year], batches=data)
    audit, rows, _ = history.build_dataset(state, feature_targets=[game])
    require(len(rows) == 1, 'NCAAF_STAGED_HISTORY')
    row = rows[0]
    require(all(row[k] >= 3 for k in ('home_prior_games', 'away_prior_games', 'home_yards_games', 'away_yards_games')),
            'NCAAF_COMPAT_MINIMUM_HISTORY_MISSING')
    # Dataset's exclusion semantics are preserved; conflicting or duplicate target
    # and history identities do not become silently useful staged evidence.
    require(not any(i['issue'] in {'duplicate_game', 'conflicting_game_excluded', 'stats_request_mismatch'} for i in audit['issues']), 'NCAAF_STAGED_HISTORY_CONFLICT')
    values = [int(row[k]) if type(row[k]) is bool else row[k] for k in research.FEATURES]
    require(all(type(v) in (int, float, bool) and research.np.isfinite(v) for v in values), 'NCAAF_STAGED_HISTORY')
    event = dict(sport='NCAAF', provider_namespace='odds_api', provider_event_id=t['provider_event_id'], canonical_event_id='cfbd:'+str(game['id']),
                 schedule_game_id=game['id'], home_team=t['home_team'], away_team=t['away_team'], start_utc=t['start_utc'], neutral_site=t['neutral_site'])
    return stages.seal(dict(version=VERSION, evidence_label=b['evidence_label'], history_bundle_sha256=bundle['sha256'], event=event,
        schedule=[game], prepared_at=prepared_at, source_receipts=[o['metadata']['received_at'] for o in b['objects']], publisher_availability='UNKNOWN',
        availability_basis='seven_day_lag_proxy', feature_cutoff=(start-timedelta(days=7)).isoformat(), feature_names=research.FEATURES,
        ordered_features=values, history_row=row, model=deepcopy(model_packet), runtime=runtime(), history_review=None, accepted=False, inference=False))


def prequote_subject(value):
    return model.digest({k:v for k,v in value['payload'].items() if k != 'history_review'})


def prequote_review_subjects(p, bundle):
    return dict(confirmed_event=model.digest(dict(event=p['event'], schedule=p['schedule'])),
        original_histories=model.digest([o['body_sha256'] for o in bundle['payload']['objects']]),
        deterministic_features=model.digest(dict(order=p['feature_names'], values=p['ordered_features'], histories=p['history_row'], prepared_at=p['prepared_at'])),
        model_and_source_basis=model.digest(dict(model=p['model'], runtime=p['runtime'], permission=bundle['payload']['plan']['payload']['advance_permissions'])))


def validate_prequote(value, bundle, *, at, review_required=False):
    if value.get('payload',{}).get('version')=='ncaaf-named-prequote-features-v2':
        from app_core.ncaaf_future_staged import validate_prequote as selected
        return selected(value,bundle,at=at,review_required=review_required)
    p = stages._envelope(value)
    expected = prepare(bundle, model_packet=p['model'], prepared_at=p['prepared_at'])['payload']
    require(p.get('version') == VERSION and set(p) == set(expected)
            and {k:v for k,v in p.items() if k != 'history_review'} == {k:v for k,v in expected.items() if k != 'history_review'}, 'NCAAF_STAGED_PREQUOTE_CONFLICT')
    now = history.timestamp(at); prepared = history.timestamp(p['prepared_at']); start = history.timestamp(p['event']['start_utc'])
    require(now is not None and prepared <= now < start and all(0 <= (now-history.timestamp(t)).total_seconds() <= 86400 for t in p['source_receipts']),
            'NCAAF_STAGED_INPUT_CLOCK')
    review = p['history_review']
    if review is not None or review_required:
        require(isinstance(review, dict) and set(review) == set('version reviewer reviewed_at subject_sha256 qualifications conclusions'.split())
                and review['version'] == 'ncaaf-owner-prequote-review-v1' and review['reviewer'] == 'Robert Velarde'
                and review['subject_sha256'] == prequote_subject(value) and isinstance(review['qualifications'], str)
                and len(review['qualifications'].strip()) >= 16, 'NCAAF_STAGED_PREQUOTE_REVIEW')
        reviewed = history.timestamp(review['reviewed_at'])
        require(reviewed is not None and prepared <= reviewed < now, 'NCAAF_STAGED_PREQUOTE_REVIEW_CLOCK')
        subjects = prequote_review_subjects(p, bundle)
        require(isinstance(review['conclusions'], dict) and set(review['conclusions']) == set(subjects), 'NCAAF_STAGED_PREQUOTE_REVIEW')
        for facet, subject_hash in subjects.items():
            c = review['conclusions'][facet]
            require(isinstance(c, dict) and set(c) == {'conclusion', 'evidence_sha256', 'finding'} and c['conclusion'] == 'VERIFIED'
                    and c['evidence_sha256'] == subject_hash and isinstance(c['finding'], str) and len(c['finding'].strip()) >= 32, 'NCAAF_STAGED_PREQUOTE_REVIEW')
    return deepcopy(p)


def join(history_bundle, prequote, quote_bundle):
    if quote_bundle.get('payload',{}).get('version')=='ncaaf-reviewed-final-quote-unaccepted-v2':
        from app_core.ncaaf_future_staged import join as selected
        return selected(history_bundle,prequote,quote_bundle)
    h = stages.read_bundle(history_bundle); q = stages.read_bundle(quote_bundle)
    require(h['version'] == stages.HISTORY_BUNDLE and q['version'] == stages.QUOTE_BUNDLE, 'NCAAF_STAGED_PARENT')
    p = q['plan']['payload']; received = q['objects'][0]['metadata']['received_at']
    validate_prequote(prequote, history_bundle, at=q['objects'][0]['requested_at'], review_required=True)
    require(history.timestamp(prequote['payload']['history_review']['reviewed_at']) <= history.timestamp(q['authorization']['payload']['authorized_at']),
            'NCAAF_STAGED_PREQUOTE_AUTHORIZATION_CLOCK')
    require(p['parent_history_sha256'] == history_bundle['sha256'] and p['prequote_sha256'] == prequote['sha256']
            and h['plan']['payload']['target'] == p['target'] and h['evidence_label'] == q['evidence_label']
            and h['plan']['payload']['custody_id'] == p['custody_id'], 'NCAAF_STAGED_PARENT')
    import json
    controls = [json.loads(base64.b64decode(b['control_receipt']['file_b64'], validate=True))['payload'] for b in (h,q)]
    require(controls[0]['custody'] == controls[1]['custody'], 'NCAAF_STAGED_PARENT_CUSTODY')
    require(q['prior_original_bytes'] == h['original_bytes'] and q['prior_active_seconds'] == h['active_seconds']
            and h['original_bytes']+q['original_bytes'] <= stages.LIMITS['max_combined_original_bytes']
            and h['active_seconds']+q['active_seconds'] < 60, 'NCAAF_STAGED_COMBINED_LIMIT')
    value = stages.seal(dict(version=BINDING_VERSION, history=deepcopy(history_bundle), prequote=deepcopy(prequote), quote=deepcopy(quote_bundle)))
    require(len(model.encode(value)) <= stages.MAX_BUNDLE, 'NCAAF_STAGED_BUNDLE_LIMIT')
    return value


def validate_binding(value, packet, *, at):
    if value.get('payload',{}).get('version')=='ncaaf-named-staged-evidence-v2':
        from app_core.ncaaf_future_staged import validate_binding as selected
        return selected(value,packet,at=at)
    b = stages._envelope(value)
    require(set(b) == {'version', 'history', 'prequote', 'quote'} and b['version'] == BINDING_VERSION
            and value == join(b['history'], b['prequote'], b['quote']), 'NCAAF_STAGED_BINDING_CONFLICT')
    pref = validate_prequote(b['prequote'], b['history'], at=at)
    p = packet['payload']; c = p['custody_packet']['payload']; n = c['native_packet']['payload']; o = n['observation']['payload']
    require(p['evidence_label'] == pref['evidence_label'] and o['event'] == pref['event'] and o['schedule'] == pref['schedule']
            and o['model'] == pref['model'] and o['features']['values'] == pref['ordered_features'], 'NCAAF_STAGED_BINDING_CONFLICT')
    require(o['features']['available_at'] == pref['prepared_at'] and history.timestamp(pref['prepared_at']) <= history.timestamp(at), 'NCAAF_STAGED_INPUT_CLOCK')
    native_batches = batches(b['history'])
    expected = [dict(sha256=hashlib.sha256(model.encode(v)).hexdigest(), bytes_b64=base64.b64encode(model.encode(v)).decode()) for v in native_batches]
    require(n['dependency_objects'] == expected, 'NCAAF_STAGED_BINDING_CONFLICT')
    objects = b['history']['payload']['objects']+b['quote']['payload']['objects']
    require(len(c['response_objects']) == len(objects), 'NCAAF_STAGED_BINDING_CONFLICT')
    for original, derived in zip(objects, c['response_objects']):
        v = derived['payload']
        require(all(v[k] == original[k] for k in ('metadata', 'body_b64', 'body_bytes', 'body_sha256')), 'NCAAF_STAGED_BINDING_CONFLICT')
    return deepcopy(pref)
