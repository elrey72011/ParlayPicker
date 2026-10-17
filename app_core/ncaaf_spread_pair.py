"""One reviewed observed market, two attributable computations, no wager value.

Only base64 byte carriers are shared. Every reference is checked, bounded and
consumed; hydrated child subjects retain their original exact hashes. No mutable
validation result is cached. Preparation never creates a human review.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from app_core import ncaaf_model_compatibility as model, ncaaf_history as history
from app_core import ncaaf_owner_spread as owner, ncaaf_spread_mass as mass
from app_core import ncaaf_staged_acquisition as stage, ncaaf_spread_execution as execution

HALF_VERSION='ncaaf-owner-observed-pair-inputs-v1'
INTEGER_VERSION='ncaaf-owner-observed-integer-pair-inputs-v2'
VERSIONS={HALF_VERSION,INTEGER_VERSION}
RESULT_VERSIONS={HALF_VERSION:'ncaaf-owner-observed-pair-result-v1',INTEGER_VERSION:'ncaaf-owner-observed-integer-pair-result-v2'}
REVIEW_VERSION='ncaaf-owner-observed-pair-review-v1'
REFERENCE='__ncaaf_shared_bytes__'
CARRIERS=frozenset({'body_b64','bytes_b64','original_record_b64','predecessor_record_b64','file_b64','journal_b64'})
SIDES=('home','away')
require=model.require


def subject(packet):
    return model.digest({k:v for k,v in packet['payload'].items() if k!='owner_review'})


def _pack(value,pool,key=None):
    if key in CARRIERS and isinstance(value,str):
        base64.b64decode(value,validate=True)
        sha=model.digest(value);pool[sha]=value
        return {REFERENCE:sha}
    if isinstance(value,dict):return {k:_pack(v,pool,k) for k,v in value.items()}
    if isinstance(value,list):return [_pack(v,pool) for v in value]
    return deepcopy(value)


def _unpack(value,pool,used,key=None,depth=0,budget=None):
    if budget is None:budget=[stage.MAX_BUNDLE]
    require(depth<64,'NCAAF_PAIR_SHARED_BYTES')
    if key in CARRIERS:
        require(isinstance(value,dict) and set(value)=={REFERENCE} and value[REFERENCE] in pool,'NCAAF_PAIR_SHARED_BYTES')
        sha=value[REFERENCE];used.add(sha)
        budget[0]-=len(pool[sha]);require(budget[0]>=0,'NCAAF_PAIR_LIMIT')
        return pool[sha]
    if isinstance(value,dict):
        require(REFERENCE not in value,'NCAAF_PAIR_SHARED_BYTES')
        return {k:_unpack(v,pool,used,k,depth+1,budget) for k,v in value.items()}
    if isinstance(value,list):return [_unpack(v,pool,used,depth=depth+1,budget=budget) for v in value]
    return deepcopy(value)


def prepare(home,away,*,prepared_at):
    """Produce a review subject. No owner review or inference permission is added."""
    pool={};children={};subjects={}
    for side,packet in zip(SIDES,(home,away)):
        require(isinstance(packet,dict) and set(packet)=={'payload','sha256'} and model.digest(packet['payload'])==packet['sha256'], 'NCAAF_PAIR_INPUT')
        v=deepcopy(packet['payload']);v['owner_review']=None
        subjects[side]=owner.subject_hash(dict(payload=v));children[side]=_pack(v,pool)
    version=INTEGER_VERSION if home['payload']['version']==owner.VERSION else HALF_VERSION
    require(home['payload']['version']==away['payload']['version'] and home['payload']['version'] in owner.INPUT_VERSIONS,'NCAAF_PAIR_INPUT')
    packet=stage.seal(dict(version=version,evidence_label=home['payload']['evidence_label'],prepared_at=prepared_at,
        shared_bytes=pool,children=children,child_subjects=subjects,owner_review=None))
    require(len(model.encode(packet))<=stage.MAX_BUNDLE,'NCAAF_PAIR_LIMIT')
    return packet


def children(packet,*,reviewed=True):
    p=stage._envelope(packet)
    require(set(p)=={'version','evidence_label','prepared_at','shared_bytes','children','child_subjects','owner_review'}
        and p['version'] in VERSIONS and p['evidence_label'] in {'RETAINED','SYNTHETIC'}
        and isinstance(p['shared_bytes'],dict) and len(p['shared_bytes'])<=64
        and isinstance(p['children'],dict) and set(p['children'])==set(SIDES)
        and set(p['child_subjects'])==set(SIDES) and len(model.encode(packet))<=stage.MAX_BUNDLE,'NCAAF_PAIR_INPUT')
    pool=p['shared_bytes'];used=set();out={}
    for sha,value in pool.items():
        require(isinstance(value,str) and len(value)<=((2*1024*1024+2)//3)*4 and model.digest(value)==sha,'NCAAF_PAIR_SHARED_BYTES')
        base64.b64decode(value,validate=True)
    for side in SIDES:
        v=_unpack(p['children'][side],pool,used)
        require(v['owner_review'] is None and owner.subject_hash(dict(payload=v))==p['child_subjects'][side]
            and v['evidence_label']==p['evidence_label'],'NCAAF_PAIR_INPUT')
        if reviewed:v['owner_review']=deepcopy(p['owner_review']['child_reviews'][side])
        child=stage.seal(v)
        require(len(model.encode(child))<=stage.MAX_BUNDLE,'NCAAF_PAIR_LIMIT');out[side]=child
    require(used==set(pool),'NCAAF_PAIR_SHARED_BYTES')
    return out


def _observation(packet):
    return packet['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']


def _same(children):
    h,a=[_observation(children[s]) for s in SIDES];hq,aq=h['quote'],a['quote']
    require(hq['market_type']=='spread_home' and aq['market_type']=='spread_away' and hq['point']==-aq['point'], 'NCAAF_PAIR_MARKET_CONFLICT')
    require(all(h[k]==a[k] for k in ('event','schedule','crosswalk','features','feature_dependencies','model'))
        and all(hq[k]==aq[k] for k in ('book','operator','product','recorded_at','provider_event_id','provider_namespace','event_home_team',
            'event_away_team','event_start_utc','period','rules','period_source','rules_source')), 'NCAAF_PAIR_MARKET_CONFLICT')
    for key in ('staged_binding','alias_review'):
        require(children['home']['payload'][key]==children['away']['payload'][key],'NCAAF_PAIR_MARKET_CONFLICT')
    ht,at=h['source_review']['terms_review'],a['source_review']['terms_review']
    require({k:v for k,v in ht.items() if k not in {'review_id','reviewed_at','listing_id'}}==
        {k:v for k,v in at.items() if k not in {'review_id','reviewed_at','listing_id'}},'NCAAF_PAIR_MARKET_CONFLICT')
    stage_binding=children['home']['payload']['staged_binding']
    return model.digest(dict(event=h['event'],model=h['model'],features=h['features'],stage=stage_binding['sha256'],
        alias=children['home']['payload']['alias_review'],market_clock=hq['recorded_at'],book=hq['book'],terms=ht))


def load(packet):
    p=stage._envelope(packet);r=p.get('owner_review')
    require(isinstance(r,dict) and set(r)=={'version','reviewer','reviewed_at','subject_sha256','qualifications','finding','child_reviews'}
        and r['version']==REVIEW_VERSION and r['reviewer']==owner.OWNER and r['subject_sha256']==subject(packet)
        and all(isinstance(r[k],str) and len(r[k].strip())>=32 for k in ('qualifications','finding'))
        and isinstance(r['child_reviews'],dict) and set(r['child_reviews'])==set(SIDES),'NCAAF_PAIR_REVIEW')
    prepared,reviewed=history.timestamp(p.get('prepared_at')),history.timestamp(r['reviewed_at'])
    require(prepared is not None and reviewed is not None and prepared<=reviewed,'NCAAF_PAIR_REVIEW_CLOCK')
    cs=children(packet)
    expected=owner.VERSION if p['version']==INTEGER_VERSION else owner.HALF_VERSION
    for child in cs.values():
        require(child['payload']['version']==expected and child['payload']['owner_review']['reviewed_at']==r['reviewed_at'], 'NCAAF_PAIR_REVIEW')
        owner.load(child)
        q=child['payload']['staged_binding']['payload']['quote']['payload']['objects'][0]
        require(history.timestamp(q['metadata']['received_at'])<=prepared,'NCAAF_PAIR_REVIEW_CLOCK')
    _same(cs)
    return cs


def inspect_result(packet,saved):
    cs=load(packet);r=stage._envelope(saved)
    require(set(r)=={'version','status','pair_input_sha256','shared_input_binding','children','comparison','finalized_at','authorization_journal_b64','authorization_journal_sha256','wager_action','live_stake','scientific_acceptance'}
        and r['version']==RESULT_VERSIONS[packet['payload']['version']] and r['status']=='COMPLETE'
        and r['pair_input_sha256']==packet['sha256'] and r['shared_input_binding']==_same(cs)
        and r['wager_action']=='PASS' and r['live_stake']==0 and r['scientific_acceptance'] is False
        and set(r['children'])==set(SIDES),'NCAAF_PAIR_RESULT')
    outcomes={};proofs=[]
    for side in SIDES:
        result=r['children'][side];v=result['payload']
        owner.inspect_result(cs[side],result,v['inference_time'])
        outcomes[side]=v['outcome_probabilities'];proofs.append(v['execution_receipt'])
        owner.accepted_dependencies(cs[side],None,r['finalized_at'])
        require(history.timestamp(v['execution_receipt']['completed_at'])<=history.timestamp(r['finalized_at']),'NCAAF_PAIR_RESULT_CLOCK')
    require(r['comparison']==mass.compare(outcomes['home'],outcomes['away']),'NCAAF_PAIR_RESULT')
    authorization=proofs[0]['authorization'];a=stage._envelope(authorization)
    require(proofs[1]['authorization']==authorization and a['subject_sha256']==packet['sha256']
        and a['input_hashes']==[cs[s]['sha256'] for s in SIDES] and a['max_computations']==2,'NCAAF_PAIR_RESULT')
    execution.validate(authorization,subject=packet['sha256'],inputs=[cs[s]['sha256'] for s in SIDES],
        at=r['finalized_at'],reviewed_at=packet['payload']['owner_review']['reviewed_at'])
    raw=base64.b64decode(r['authorization_journal_b64'],validate=True)
    require(len(raw)<=128*1024 and hashlib.sha256(raw).hexdigest()==r['authorization_journal_sha256'],'NCAAF_PAIR_RESULT')
    prefixes=[base64.b64decode(p['journal_prefix_b64'],validate=True) for p in proofs]
    require(prefixes[1].startswith(prefixes[0]) and raw==prefixes[1]+model.encode(dict(kind='finish',status='COMPLETE',attempted=2,completed=2,
        accepted=False,wager_action='PASS',live_stake=0))+b'\n','NCAAF_PAIR_RESULT')
    return deepcopy(r)


def admitted_analysis(packet,sources,*,authorization,custody_root,inventory=None):
    """Two native calls through the actual market caller, under one spent claim."""
    import pandas as pd
    from app_core import ncaaf_pipeline_evidence as adapter
    cs=load(packet);require(isinstance(sources,dict) and set(sources)==set(SIDES),'NCAAF_PAIR_INPUT')
    rows=[];results={};entered=False;root=Path(custody_root)
    try:
        with execution.execution([cs[s] for s in SIDES],authorization,root=root,subject=packet['sha256'],
                reviewed_at=packet['payload']['owner_review']['reviewed_at'],at=adapter.generated_time()):
            entered=True
            for side in SIDES:
                frame=owner.admitted_analysis(cs[side],sources[side],inventory=inventory)
                matched=frame.loc[frame.market_type.eq('spread_'+side)]
                require(len(matched)==1 and matched.iloc[0].ml_inference_status=='success','NCAAF_PAIR_INCOMPLETE')
                metadata=json.loads(matched.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']
                results[side]=metadata['computation'];rows.append(frame)
            finalized=adapter.generated_time()
            for child in cs.values():owner.accepted_dependencies(child,None,finalized)
            execution.validate(authorization,subject=packet['sha256'],inputs=[cs[s]['sha256'] for s in SIDES],
                at=finalized,reviewed_at=packet['payload']['owner_review']['reviewed_at'])
        journal=(root/(authorization['sha256']+'.inference-attempts.jsonl')).read_bytes()
        result=stage.seal(dict(version=RESULT_VERSIONS[packet['payload']['version']],status='COMPLETE',pair_input_sha256=packet['sha256'],
            shared_input_binding=_same(cs),children=results,comparison=mass.compare(*(results[s]['payload']['outcome_probabilities'] for s in SIDES)),
            finalized_at=finalized,authorization_journal_b64=base64.b64encode(journal).decode(),authorization_journal_sha256=hashlib.sha256(journal).hexdigest(),
            wager_action='PASS',live_stake=0,scientific_acceptance=False))
        inspect_result(packet,result)
        stage._exclusive(root/(authorization['sha256']+'.comparison.json'),model.encode(result))
    except BaseException:
        if entered:
            failure=stage.seal(dict(version='ncaaf-observed-pair-incomplete-v1',status='INCOMPLETE',pair_input_sha256=packet['sha256'],
                completed_children=results,reason='NCAAF_PAIR_INCOMPLETE',accepted=False,inference_complete=False,wager_action='PASS',live_stake=0))
            stage._exclusive(root/(authorization['sha256']+'.comparison-incomplete.json'),model.encode(failure))
        raise
    out=pd.concat(rows,ignore_index=True)
    for index,row in out.iterrows():
        metadata=json.loads(row.ml_estimate_metadata);p=metadata['ncaaf_inputs']['payload']
        p['paired_spread_comparison']=dict(input=deepcopy(packet),result=deepcopy(result))
        metadata['ncaaf_inputs']['sha256']=model.digest(p)
        out.at[index,'ml_estimate_metadata']=model.encode(metadata).decode()
    out.attrs['ncaaf_pair_result']=deepcopy(result)
    return out


def inspect_row(child,metadata):
    saved=metadata.get('paired_spread_comparison')
    require(isinstance(saved,dict) and set(saved)=={'input','result'},'NCAAF_PAIR_INCOMPLETE')
    cs=load(saved['input']);r=inspect_result(saved['input'],saved['result'])
    sides=[s for s in SIDES if cs[s]==child]
    require(len(sides)==1 and r['children'][sides[0]]==metadata['computation'],'NCAAF_PAIR_RESULT')
    return deepcopy(r['comparison'])


def private_display(source):
    from app_core import ncaaf_pipeline_evidence as adapter
    require(adapter.diagnose(source)==dict(status='COMPLETE',reason='AVAILABLE'),'NCAAF_SPREAD_DISPLAY_UNAVAILABLE')
    p=json.loads(source['ml_estimate_metadata'])['ncaaf_inputs']['payload']
    require(p['original_packet']['payload']['version'] in owner.INPUT_VERSIONS,'NCAAF_SPREAD_DISPLAY_UNAVAILABLE')
    masses=mass.normalize(p['outcome_probabilities'])
    return dict(version='ncaaf-owner-spread-private-display-v1',labels=['OWNER_REVIEWED','PRIVATE_RESEARCH','UNCALIBRATED'],
        outcome_probabilities=masses,conditional_no_push=mass.conditional_no_push(masses),
        comparison=inspect_row(p['original_packet'],p) if 'paired_spread_comparison' in p else None,
        input_sha256=p['original_packet']['sha256'],result_sha256=p['computation']['sha256'],
        inference_time=p['inference_time'],wager_action='PASS',live_stake=0,ev=None,edge=None)


def capture_exports(analysis,*,custody_root,database):
    """Retain both private children through existing capture/export mechanisms.

    Each child uses its own PASS card, without asking the public EV selector to
    rank opposing private estimates. Both originals carry the complete shared
    envelope and two receipts. No collection, inference or authorization occurs.
    """
    from core import streamlit_pipeline as pipeline
    from core.live_wager_contract import finalize_live_wagers
    from app_core import prediction_evidence as store,research_replay
    from app_core.candidate_evidence_schema import project
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    root=Path(custody_root);database=Path(database)
    require(root.is_dir() and execution.inspect_security(root)['status']=='VERIFIED'
        and database.parent.resolve()==root.resolve(),'NCAAF_SPREAD_PRIVATE_CUSTODY')
    rows=analysis.to_dict('records');require(len(rows)==2,'NCAAF_PAIR_INPUT')
    displays=[private_display(row) for row in rows]
    metadata=[json.loads(row['ml_estimate_metadata'])['ncaaf_inputs']['payload'] for row in rows]
    require(metadata[0]['paired_spread_comparison']==metadata[1]['paired_spread_comparison']
        and {row['market_type'] for row in rows}=={'spread_home','spread_away'},'NCAAF_PAIR_MARKET_CONFLICT')
    output=[]
    for i in range(2):
        frame=analysis.iloc[[i]].copy();diagnostics={}
        best=pipeline.build_best_picks_df(frame,diagnostics_out=diagnostics)
        authority=diagnostics['candidate_authority_df']
        from app_core.candidate_chronology import now_utc
        # The existing finalizer requires a positive arithmetic denominator.
        # One nominal unit is never funding: empty policies plus the private
        # source gate force every production allocation to zero, checked below.
        card,_=finalize_live_wagers(project(authority),best,1,now=now_utc(),policies={},config={})
        context=store.begin_run({},path=database,root=root)
        captured,card=store.capture_run(context,project(authority),card,frame,path=database,authoritative_candidates=True)
        frames=[per_game_board(card,captured,family=f,novig_only=True,college_fallback=True,private_research=True)
            for f in ('overall','sides','totals')]
        package=build_package(*frames);validate_package(package)
        require('body_b64' not in json.dumps(package) and not captured.production_eligible.fillna(False).any()
            and all(x['production_bet_amount']==0 for x in card.wager_contract)
            and all(game.get('research_display',{}).get('probability') is None
                for games in package['games'].values() for game in games),'NCAAF_PAIR_PUBLIC_EXCLUSION')
        receipt=research_replay.retain_export(frames,package,card,captured,path=database)
        output.append(dict(receipt=receipt,card=card,captured=captured,frames=frames,package=package,display=displays[i]))
    return output
