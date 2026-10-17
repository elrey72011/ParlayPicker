"""Named history preparation and original-bound later identity join. No inference."""
import base64
from copy import deepcopy
from datetime import timedelta
import hashlib
from pathlib import Path
from app_core import ncaaf_staged_acquisition as stage, ncaaf_staged_evidence as prior
from app_core import ncaaf_future_target as target
from app_core import ncaaf_history as history, ncaaf_model_compatibility as model
from app_core import ncaaf_research as research, ncaaf_response_custody as custody

VERSION='ncaaf-named-prequote-features-v2'
BINDING_VERSION='ncaaf-named-staged-evidence-v2'
require=model.require


def batches(bundle):
    b=stage.read_bundle(bundle)
    require(b['version']==target.HISTORY_BUNDLE,'NCAAF_STAGED_PARENT')
    result=[]
    for obj in b['objects']:
        meta=obj['metadata'];q=meta['request_scope'];raw=base64.b64decode(obj['body_b64'],validate=True)
        request=dict(kind='games',year=q['year']) if meta['endpoint']=='games' else dict(kind='stats',year=q['year'],week=q['week'],season_type=q['seasonType'])
        batch=dict(request=request,retrieved_at=meta['received_at'],records=history._clean(request['kind'],custody._parse(raw)))
        custody._project_cfbd(custody._parse(raw),meta,batch);result.append(batch)
    return result


def runtime():
    import sys
    r=prior.runtime()
    r['interpreter_sha256']=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    r['successor_components']={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for m in (target,__import__(__name__,fromlist=['runtime']))}
    return r


def prepare(bundle,*,model_packet,prepared_at):
    b=stage.read_bundle(bundle);p=b['plan']['payload'];t=p['target'];now=history.timestamp(prepared_at)
    require(b['version']==target.HISTORY_BUNDLE and now is not None and now<history.timestamp(t['start_utc'])
        and all(history.timestamp(o['metadata']['received_at']) is not None and 0<=(now-history.timestamp(o['metadata']['received_at'])).total_seconds()<=86400 for o in b['objects']),
        'NCAAF_STAGED_INPUT_CLOCK')
    checked=model.load_model(model_packet)
    require(checked['compatibility']['family']=='spread','NCAAF_STAGED_MODEL_TARGET')
    data=batches(bundle); games=data[0]['records'];start=history.timestamp(t['start_utc'])
    candidates=[g for g in games if (g.get('homeTeam')==t['home_school'] and g.get('awayTeam')==t['away_school'])
        or (g.get('homeTeam')==t['away_school'] and g.get('awayTeam')==t['home_school'])]
    require(len(candidates)==1,'NCAAF_STAGED_CFBD_EVENT_CONFLICT');game=candidates[0]
    require(game.get('homeTeam')==t['home_school'] and game.get('awayTeam')==t['away_school']
        and history.timestamp(game.get('startDate'))==start and game.get('completed') is False and game.get('startTimeTBD') is False
        and all(type(game.get(k)) is int and game[k]>0 for k in ('id','homeId','awayId')) and game['homeId']!=game['awayId']
        and type(game.get('neutralSite')) is bool and game['neutralSite']==t['neutral_site'],'NCAAF_STAGED_CFBD_EVENT_CONFLICT')
    audit,rows,_=history.build_dataset(dict(schema=1,years=[start.year],batches=data),feature_targets=[game])
    require(len(rows)==1 and all(rows[0][k]>=3 for k in ('home_prior_games','away_prior_games','home_yards_games','away_yards_games')),
        'NCAAF_COMPAT_MINIMUM_HISTORY_MISSING')
    require(not any(i['issue'] in {'duplicate_game','conflicting_game_excluded','stats_request_mismatch'} for i in audit['issues']),
        'NCAAF_STAGED_HISTORY_CONFLICT')
    row=rows[0];values=[int(row[k]) if type(row[k]) is bool else row[k] for k in research.FEATURES]
    require(all(type(v) in (int,float,bool) and research.np.isfinite(v) for v in values),'NCAAF_STAGED_HISTORY')
    event=dict(sport='NCAAF',canonical_event_id='cfbd:'+str(game['id']),schedule_game_id=game['id'],
        home_school=t['home_school'],away_school=t['away_school'],home_id=game['homeId'],away_id=game['awayId'],
        venue=t['venue'],start_utc=t['start_utc'],neutral_site=t['neutral_site'],provider_event_id=None)
    return stage.seal(dict(version=VERSION,evidence_label=b['evidence_label'],history_bundle_sha256=bundle['sha256'],event=event,
        schedule=[game],prepared_at=prepared_at,source_receipts=[o['metadata']['received_at'] for o in b['objects']],publisher_availability='UNKNOWN',
        availability_basis='seven_day_lag_proxy',feature_cutoff=(start-timedelta(days=7)).isoformat(),feature_names=research.FEATURES,
        ordered_features=values,history_row=row,model=deepcopy(model_packet),runtime=runtime(),history_review=None,accepted=False,inference=False))


def validate_prequote(value,bundle,*,at,review_required=False):
    p=stage._envelope(value); expected=prepare(bundle,model_packet=p['model'],prepared_at=p['prepared_at'])['payload']
    require(set(p)==set(expected) and {k:v for k,v in p.items() if k!='history_review'}=={k:v for k,v in expected.items() if k!='history_review'},
        'NCAAF_STAGED_PREQUOTE_CONFLICT')
    now,prepared,start=[history.timestamp(x) for x in (at,p['prepared_at'],p['event']['start_utc'])]
    require(now is not None and prepared<=now<start and all(0<=(now-history.timestamp(t)).total_seconds()<=86400 for t in p['source_receipts']),
        'NCAAF_STAGED_INPUT_CLOCK')
    r=p['history_review']
    if r is not None or review_required:
        require(isinstance(r,dict) and set(r)==set('version reviewer reviewed_at subject_sha256 qualifications conclusions'.split())
            and r['version']=='ncaaf-owner-prequote-review-v2' and r['reviewer']==target.OWNER
            and r['subject_sha256']==prior.prequote_subject(value) and isinstance(r['qualifications'],str) and len(r['qualifications'].strip())>=16,
            'NCAAF_STAGED_PREQUOTE_REVIEW')
        reviewed=history.timestamp(r['reviewed_at'])
        require(reviewed is not None and prepared<=reviewed<now,'NCAAF_STAGED_PREQUOTE_REVIEW_CLOCK')
        subjects=prior.prequote_review_subjects(p,bundle)
        require(isinstance(r['conclusions'],dict) and set(r['conclusions'])==set(subjects),'NCAAF_STAGED_PREQUOTE_REVIEW')
        for k,sha in subjects.items():
            c=r['conclusions'][k]
            require(isinstance(c,dict) and set(c)=={'conclusion','evidence_sha256','finding'} and c['conclusion']=='VERIFIED'
                and c['evidence_sha256']==sha and isinstance(c['finding'],str) and len(c['finding'].strip())>=32,'NCAAF_STAGED_PREQUOTE_REVIEW')
    return deepcopy(p)


def validate_quote_parent(h,p,prequote,*,at):
    pref=validate_prequote(prequote,stage.seal(h),at=at,review_required=True)
    require(target.same_target(h['plan']['payload']['target'],p['target'])
        and h['plan']['payload']['target_evidence']==p['target_evidence'],'NCAAF_STAGED_PARENT')
    game=target.validate_identity(p['identity_review'],at=at,target=p['target'],label=p['evidence_label'])
    require(pref['schedule']==[game],'NCAAF_FRESH_IDENTITY_CONFLICT')
    obj=target.discovery_original(p['identity_review']['discovery'],p['evidence_label'])
    # The canonical identities in the separately authorized discovery must agree
    # with the original CFBD confirmation; its literal descriptor may be absent.
    d=p['identity_review']['discovery']['bundle']['payload']['plan']['payload']['target']
    require(d['canonical_event_id']=='cfbd:'+str(game['id']) and d['home_id']==game['homeId'] and d['away_id']==game['awayId']
        and d['neutral_site']==game['neutralSite'] and history.timestamp(d['start_utc'])==history.timestamp(game['startDate']),
        'NCAAF_FRESH_IDENTITY_CONFLICT')
    return pref


def join(history_bundle,prequote,quote_bundle):
    h=stage.read_bundle(history_bundle);q=stage.read_bundle(quote_bundle);p=q['plan']['payload']
    require(h['version']==target.HISTORY_BUNDLE and q['version']==target.QUOTE_BUNDLE
        and p['parent_history_sha256']==history_bundle['sha256'] and p['prequote_sha256']==prequote['sha256']
        and h['evidence_label']==q['evidence_label'] and h['plan']['payload']['custody_id']==p['custody_id'], 'NCAAF_STAGED_PARENT')
    validate_quote_parent(h,p,prequote,at=q['objects'][0]['requested_at'])
    require(history.timestamp(prequote['payload']['history_review']['reviewed_at'])<=history.timestamp(q['authorization']['payload']['authorized_at']),
        'NCAAF_STAGED_PREQUOTE_AUTHORIZATION_CLOCK')
    import json
    controls=[json.loads(base64.b64decode(b['control_receipt']['file_b64'],validate=True))['payload'] for b in (h,q)]
    require(controls[0]['custody']==controls[1]['custody'],'NCAAF_STAGED_PARENT_CUSTODY')
    require(q['prior_original_bytes']==h['original_bytes'] and q['prior_active_seconds']==h['active_seconds']
        and h['original_bytes']+q['original_bytes']<=stage.LIMITS['max_combined_original_bytes']
        and h['active_seconds']+q['active_seconds']<60,'NCAAF_STAGED_COMBINED_LIMIT')
    value=stage.seal(dict(version=BINDING_VERSION,history=deepcopy(history_bundle),prequote=deepcopy(prequote),quote=deepcopy(quote_bundle)))
    require(len(model.encode(value))<=stage.MAX_BUNDLE,'NCAAF_STAGED_BUNDLE_LIMIT')
    return value


def validate_binding(value,packet,*,at):
    b=stage._envelope(value)
    require(set(b)=={'version','history','prequote','quote'} and b['version']==BINDING_VERSION
        and value==join(b['history'],b['prequote'],b['quote']),'NCAAF_STAGED_BINDING_CONFLICT')
    pref=validate_prequote(b['prequote'],b['history'],at=at)
    p=packet['payload'];c=p['custody_packet']['payload'];n=c['native_packet']['payload'];o=n['observation']['payload']
    identity=b['quote']['payload']['plan']['payload']['identity_review']
    require(o['quote']['book']=='draftkings' and p['alias_review']==identity and p['evidence_label']==pref['evidence_label']
        and o['schedule']==pref['schedule'] and o['model']==pref['model']
        and o['features']==dict(order=list(pref['feature_names']),values=pref['ordered_features'],available_at=pref['prepared_at']),
        'NCAAF_STAGED_BINDING_CONFLICT')
    expected=[dict(sha256=hashlib.sha256(model.encode(v)).hexdigest(),bytes_b64=base64.b64encode(model.encode(v)).decode()) for v in batches(b['history'])]
    require(n['dependency_objects']==expected,'NCAAF_STAGED_BINDING_CONFLICT')
    originals=b['history']['payload']['objects']+b['quote']['payload']['objects']
    require(len(c['response_objects'])==len(originals),'NCAAF_STAGED_BINDING_CONFLICT')
    for original,derived in zip(originals,c['response_objects']):
        require(all(derived['payload'][k]==original[k] for k in ('metadata','body_b64','body_bytes','body_sha256')),'NCAAF_STAGED_BINDING_CONFLICT')
    return deepcopy(pref)
