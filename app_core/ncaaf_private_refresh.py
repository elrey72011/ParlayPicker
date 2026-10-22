"""Normal Refresh entry to existing reviewed, separately authorized private callers.

No collection, fitting, acceptance registration or public delivery occurs here.
"""
import base64
from copy import deepcopy
import json
from pathlib import Path
from scripts.ncaaf_private_refresh_scope import authenticate
authenticate(Path(__file__).resolve().parents[1])
import pandas as pd
from app_core import ncaaf_model_compatibility as model
from app_core import ncaaf_spread_pair as pair, ncaaf_owner_spread as owner
from app_core import ncaaf_pipeline_evidence as evidence
from app_core import ncaaf_staged_acquisition as stage

VERSION='ncaaf-normal-private-refresh-selection-v1'
RESULT_VERSION='ncaaf-normal-private-refresh-result-v1'
require=model.require


def load(raw):
    authenticate(Path(__file__).resolve().parents[1])
    require(isinstance(raw,bytes) and 0<len(raw)<=stage.MAX_BUNDLE,'NCAAF_PRIVATE_REFRESH_LIMIT')
    value=json.loads(raw)
    require(set(value)=={'version','packet','authorization'} and value['version']==VERSION,
            'NCAAF_PRIVATE_REFRESH_SCHEMA')
    # Static admission only. This never creates an inference claim or computes.
    children=pair.load(value['packet'])
    require(all(owner.conditional(child) for child in children.values()),
            'NCAAF_PRIVATE_REFRESH_CONTRACT')
    stage._envelope(value['authorization'])
    return deepcopy(value)


def _source(packet):
    o=pair._observation(packet);q=o['quote'];e=o['event']
    from zoneinfo import ZoneInfo
    from app_core.ncaaf_history import timestamp
    day=timestamp(e['start_utc']).astimezone(ZoneInfo('America/New_York')).date().isoformat()
    return dict(league='NCAAF',sport='NCAAF',matchup_id='|'.join((e['home_team'],e['away_team'],day)),
        canonical_event_id=e['canonical_event_id'],
        home_team=e['home_team'],away_team=e['away_team'],game_start_utc=e['start_utc'],
        market_type=q['market_type'],spread_line=q['point'],odds_american=q['price'],
        odds_book=q['book'],bookmaker=q['book'],odds_timestamp=q['recorded_at'],
        provider_event_id=q['provider_event_id'],provider_namespace=q['provider_namespace'],
        event_home_team=q['event_home_team'],event_away_team=q['event_away_team'],event_start_utc=q['event_start_utc'],
        provider_quotes=json.dumps([deepcopy(q)],sort_keys=True,separators=(',',':')))


def custody_root(children):
    roots=[]
    for child in children.values():
        history=child['payload']['staged_binding']['payload']['history']['payload']
        configuration=json.loads(base64.b64decode(history['control_receipt']['file_b64'],validate=True))
        roots.append(Path(configuration['payload']['custody']['root']).resolve())
    require(len(roots)==2 and roots[0]==roots[1],'NCAAF_SPREAD_PRIVATE_CUSTODY')
    return roots[0]


def ranking(analysis):
    rows=[]
    for source in analysis.to_dict('records'):
        display=pair.private_display(source)
        metadata=json.loads(source['ml_estimate_metadata'])['ncaaf_inputs']['payload']
        packet=metadata['original_packet'];q=pair._observation(packet)['quote']
        p=display['outcome_probabilities'];price=q['price']
        observation=pair._observation(packet)
        decimal=1+price/100 if price>0 else 1+100/abs(price)
        side='home' if q['market_type']=='spread_home' else 'away'
        rows.append(dict(team=observation['event'][side+'_team'],side=side,
            line=q['point'],price=price,cover=p['cover'],push=p['push'],loss=p['loss'],
            evidence_label=metadata['paired_spread_comparison']['input']['payload']['evidence_label'],
            event=deepcopy(observation['event']),
            model_artifact_sha256=observation['model']['compatibility']['artifact_sha256'],
            model_binding_sha256=observation['model']['compatibility']['binding_sha256'],
            inference_time=display['inference_time'],market_observed_at=q['recorded_at'],
            conditional_on_no_push=display['conditional_no_push'],
            estimated_ev_per_unit=p['cover']*decimal+p['push']-1,
            input_sha256=packet['sha256'],result_sha256=metadata['computation']['sha256'],
            assumptions=dict(period=q['period'],settlement=q['rules'],source=q['rules_source']),
            applicability='UNVERIFIED_CONDITIONAL_RESEARCH',calibration='UNVALIDATED',
            wager_action='PASS',stake=0))
    require(len(rows)==2 and {r['side'] for r in rows}=={'home','away'},'NCAAF_PAIR_INCOMPLETE')
    rows.sort(key=lambda r:(-r['cover'],r['side']))
    from app_core.ncaaf_spread_mass import compare
    sides={row['side']:row for row in rows}
    comparison=compare({k:sides['home'][k] for k in ('cover','push','loss')},
                       {k:sides['away'][k] for k in ('cover','push','loss')})
    for r in rows:r['selection']='TIE' if comparison['higher_cover']=='tie' else 'HIGHER_MODEL_COVER' if r['side']==comparison['higher_cover'] else 'OTHER_SIDE'
    return rows


def run(selection):
    """One actual Refresh action; existing journals enforce one-use authorization."""
    value=load(model.encode(selection))
    packet=value['packet'];children=pair.load(packet);root=custody_root(children)
    rows={side:_source(child) for side,child in children.items()}
    analysis=pair.admitted_analysis(packet,rows,authorization=value['authorization'],custody_root=root)
    exports=pair.capture_exports(analysis,custody_root=root,database=root/'ncaaf-private-refresh.sqlite')
    ranked=ranking(analysis)
    # The qualified wagering card deliberately withholds private probabilities.
    # Export an explicit research view, derived again from retained originals.
    replayed=replay_ranked([saved['receipt'] for saved in exports],database=root/'ncaaf-private-refresh.sqlite')
    require(replayed==ranked,'NCAAF_PRIVATE_REFRESH_REPLAY')
    basename=value['authorization']['sha256']+'.private-refresh'
    csv=private_csv(ranked)
    stage._exclusive(root/(basename+'.csv'),csv)
    import hashlib
    view=stage.seal(dict(version=RESULT_VERSION,rows=ranked,
        export_receipts=[saved['receipt'] for saved in exports],
        csv_name=basename+'.csv',csv_sha256=hashlib.sha256(csv).hexdigest(),
        ranking_criterion='unconditional_model_cover',calibration='UNVALIDATED',
        applicability='UNVERIFIED_CONDITIONAL_RESEARCH',wager_action='PASS',stake=0))
    stage._exclusive(root/(basename+'.json'),model.encode(view))
    require(read_private_export(model.encode(view),root=root,database=root/'ncaaf-private-refresh.sqlite')==ranked,
        'NCAAF_PRIVATE_REFRESH_REPLAY')
    for saved in exports:
        saved['private_result']=dict(path=str(root/(basename+'.json')),sha256=view['sha256'],
            csv_path=str(root/(basename+'.csv')),csv_sha256=view['payload']['csv_sha256'])
    return analysis,exports,ranked


def replay_ranked(receipts,*,database):
    from app_core import research_replay
    require(isinstance(receipts,list) and len(receipts)==2,'NCAAF_PRIVATE_REFRESH_REPLAY')
    rows=[]
    for receipt in receipts:
        _,originals=research_replay.read_export(receipt['export_id'],path=database)
        require(bool(originals),'NCAAF_PRIVATE_REFRESH_REPLAY')
        for original in originals.values():
            frame=research_replay.frame_from_payload(original['original']['producer'])
            rows.extend(row for row in frame.to_dict('records') if owner.is_private_source(row))
    require(len(rows)==2,'NCAAF_PRIVATE_REFRESH_REPLAY')
    # ranking validates every native receipt and the shared pair contract.
    return ranking(pd.DataFrame(rows))


def private_csv(rows):
    flattened=[]
    for row in rows:
        value={key:row[key] for key in ('team','side','line','price','cover','push','loss','evidence_label',
            'estimated_ev_per_unit','selection','input_sha256','result_sha256',
            'model_artifact_sha256','model_binding_sha256','inference_time',
            'market_observed_at','applicability','calibration','wager_action','stake')}
        value.update(home_team=row['event']['home_team'],away_team=row['event']['away_team'],
            kickoff_utc=row['event']['start_utc'],canonical_event_id=row['event']['canonical_event_id'],
            conditional_cover_given_no_push=row['conditional_on_no_push']['value'],
            conditional_no_push_state=row['conditional_on_no_push']['state'],
            period=row['assumptions']['period'],settlement=row['assumptions']['settlement'],
            ranking_criterion='unconditional_model_cover')
        flattened.append(value)
    return pd.DataFrame(flattened).to_csv(index=False).encode('utf-8')


def read_private_export(raw,*,root,database):
    import hashlib,re
    require(isinstance(raw,bytes) and 0<len(raw)<=stage.MAX_BUNDLE,'NCAAF_PRIVATE_REFRESH_LIMIT')
    view=json.loads(raw);stage._envelope(view);p=view['payload']
    require(set(p)=={'version','rows','export_receipts','csv_name','csv_sha256',
        'ranking_criterion','calibration','applicability','wager_action','stake'}
        and p['version']==RESULT_VERSION and p['ranking_criterion']=='unconditional_model_cover'
        and p['calibration']=='UNVALIDATED' and p['applicability']=='UNVERIFIED_CONDITIONAL_RESEARCH'
        and p['wager_action']=='PASS' and p['stake']==0,'NCAAF_PRIVATE_REFRESH_REPLAY')
    require(isinstance(p['csv_name'],str) and re.fullmatch(r'[0-9a-f]{64}\.private-refresh\.csv',p['csv_name']),
        'NCAAF_PRIVATE_REFRESH_REPLAY')
    rows=replay_ranked(p['export_receipts'],database=database)
    require(rows==p['rows'],'NCAAF_PRIVATE_REFRESH_REPLAY')
    csv=(Path(root)/p['csv_name']).read_bytes()
    require(hashlib.sha256(csv).hexdigest()==p['csv_sha256'] and csv==private_csv(rows),
        'NCAAF_PRIVATE_REFRESH_REPLAY')
    return rows


def pipeline(controls):
    """Same normal button, explicitly selected private mode; no live fallback."""
    empty=pd.DataFrame()
    updates={k:empty.copy() for k in ('analysis_df','best_picks_df','parlays_df','portfolio_df',
        'odds_df','theover_df','kalshi_df','gemini_df','parlays_2_df','parlays_3_df')}
    updates.update(simulation_results={},pipeline_running=False,pipeline_status='UNAVAILABLE',
                   diagnostics={},ncaaf_private_refresh_ranked=[],ncaaf_private_refresh_receipts=[])
    try:
        require(controls.get('ncaaf_private_refresh_selection') is not None,'NCAAF_PRIVATE_REFRESH_INPUTS_MISSING')
        analysis,exports,ranked=run(controls['ncaaf_private_refresh_selection'])
    except (ValueError,TypeError,KeyError,OSError):
        return updates,[],['UNAVAILABLE: private NCAAF requires complete current evidence, exact review, separately configured inference authority and verified custody.']
    updates.update(analysis_df=analysis,best_picks_df=pd.DataFrame(ranked),
        pipeline_status='PRIVATE_CONDITIONAL_RESEARCH',ncaaf_private_refresh_ranked=ranked,
        ncaaf_private_refresh_receipts=[x['receipt'] for x in exports],
        ncaaf_private_refresh_result=exports[0]['private_result'],
        diagnostics=dict(private_research=True,ranking_criterion='unconditional_model_cover',
                         evidence_label=controls['ncaaf_private_refresh_selection']['packet']['payload']['evidence_label']))
    return updates,[],[]
