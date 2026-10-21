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
    return dict(league='NCAAF',sport='NCAAF',matchup_id=e['canonical_event_id'],
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
        decimal=1+price/100 if price>0 else 1+100/abs(price)
        rows.append(dict(side='home' if q['market_type']=='spread_home' else 'away',
            line=q['point'],price=price,cover=p['cover'],push=p['push'],loss=p['loss'],
            conditional_on_no_push=display['conditional_no_push'],
            estimated_ev_per_unit=p['cover']*decimal+p['push']-1,
            input_sha256=packet['sha256'],result_sha256=metadata['computation']['sha256'],
            assumptions=dict(period=q['period'],settlement=q['rules'],source=q['rules_source']),
            applicability='UNVERIFIED_CONDITIONAL_RESEARCH',calibration='UNVALIDATED',
            wager_action='PASS',stake=0))
    require(len(rows)==2 and {r['side'] for r in rows}=={'home','away'},'NCAAF_PAIR_INCOMPLETE')
    rows.sort(key=lambda r:(-r['cover'],r['side']))
    for r in rows:r['selection']='TIE' if rows[0]['cover']==rows[1]['cover'] else 'HIGHER_MODEL_COVER' if r is rows[0] else 'OTHER_SIDE'
    return rows


def run(selection):
    """One actual Refresh action; existing journals enforce one-use authorization."""
    value=load(model.encode(selection))
    packet=value['packet'];children=pair.load(packet);root=custody_root(children)
    rows={side:_source(child) for side,child in children.items()}
    analysis=pair.admitted_analysis(packet,rows,authorization=value['authorization'],custody_root=root)
    exports=pair.capture_exports(analysis,custody_root=root,database=root/'ncaaf-private-refresh.sqlite')
    ranked=ranking(analysis)
    # Static replay from SQLite is required before reporting success.
    from app_core import research_replay
    for saved in exports:
        _,originals=research_replay.read_export(saved['receipt']['export_id'],path=root/'ncaaf-private-refresh.sqlite')
        require(bool(originals),'NCAAF_PRIVATE_REFRESH_REPLAY')
        for original in originals.values():
            frame=research_replay.frame_from_payload(original['original']['producer'])
            for row in frame.to_dict('records'):
                if owner.is_private_source(row):
                    require(evidence.diagnose(row)==dict(status='COMPLETE',reason='AVAILABLE'),'NCAAF_PRIVATE_REFRESH_REPLAY')
    return analysis,exports,ranked


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
    updates.update(analysis_df=analysis,best_picks_df=pd.concat([x['card'] for x in exports],ignore_index=True),
        pipeline_status='PRIVATE_CONDITIONAL_RESEARCH',ncaaf_private_refresh_ranked=ranked,
        ncaaf_private_refresh_receipts=[x['receipt'] for x in exports],
        diagnostics=dict(private_research=True,ranking_criterion='unconditional_model_cover',
                         evidence_label=controls['ncaaf_private_refresh_selection']['packet']['payload']['evidence_label']))
    return updates,[],[]
