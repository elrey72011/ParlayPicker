"""Streamlit-only NFL preparation, actual approval, calculation and saved view."""
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
import re

import streamlit as st

from app.ui.nfl_workflow_state import revoke, invalidate_review

LIMITATIONS = ('Uncalibrated private research. PASS — zero stake. Preferred side means the higher conditional '
    'cover estimate, not validated accuracy, value or a sportsbook recommendation. The fixed model has no '
    'explicit quarterback, injury or inactive adjustment. Player availability is not collected here. '
    'Prices stay separate; no EV or wager admission. One game cannot establish predictive advantage.')
MESSAGES = {
    'PRIVATE_STORE_CONFIGURATION_REQUIRED':'Private durable storage is not configured. Owner deployment setup is required before this workflow can run.',
    'PRIVATE_STORE_UNAVAILABLE':'Private storage is unavailable. No automatic retry or replacement acquisition was started.',
    'PRIVATE_OPERATION_ALREADY_SPENT_NO_RETRY':'This operation was already claimed. It may have completed, failed or been interrupted. Its allowance cannot be reset. View saved evidence or explicitly prepare a successor.',
    'NFL_WORKFLOW_QUOTE_STALE':'The observed line is stale. It cannot authorize a calculation. Its original clock is preserved; a new observation requires a new explicit request approval.',
    'NFL_WORKFLOW_INTEGER_LINE_UNSUPPORTED':'The observed line is an integer. Push cases are unsupported; it will not be rounded or replaced automatically.',
    'NFL_WORKFLOW_KICKOFF_REACHED':'Kickoff has been reached. Prospective calculation is unavailable; saved results remain viewable.',
    'NFL_WORKFLOW_HISTORY_MISSING':'At least one team has no qualifying completed histories in this original. No features or scores were invented.',
    'NFL_WORKFLOW_SOURCE_PERMISSION_EXPIRED':'The retained source-use permission does not cover this action. A new reviewed source capture may be required.',
    'NFL_WORKFLOW_MARKET_IDENTITY_CONFLICT':'The captured event or opposing selections conflict with the reviewed schedule. Original evidence is preserved; no replacement request was sent.',
    'NFL_WORKFLOW_LATEST_LAUNCH_PASSED':'The latest launch time passed. A full 60-second transport budget must remain. No request was sent; a successor needs new approval.',
    'NFL_WORKFLOW_OPERATION_EXPIRED':'The approved capture window expired. It was not automatically restarted.',
    'NFL_WORKFLOW_ACQUISITION_NOT_ENABLED':'Acquisition is disabled pending owner-approved deployment and operational configuration.',
    'NFL_WORKFLOW_INFERENCE_NOT_ENABLED':'New calculations are disabled pending owner-approved operational configuration. Saved viewing is separate.',
}


def error(exc):
    code=str(exc) if isinstance(exc,ValueError) else ''
    if not re.fullmatch(r'(NFL_|PRIVATE_)[A-Z_]+',code):code='PRIVATE_STORE_UNAVAILABLE'
    st.error(MESSAGES.get(code,'This action stopped. The original evidence and spent accounting are preserved.'))
    st.caption('Reason: '+code)


def eastern(value):
    from app_core.nfl_workflow_inputs import instant
    return instant(value).astimezone(ZoneInfo('America/New_York')).strftime('%b %d, %Y %I:%M:%S %p %Z')


def render(setting, *, service_factory=None):
    st.subheader('Guided private NFL forecasts')
    st.caption('Select games, review original evidence, authorize bounded actions and reopen saved forecasts here. No normal Refresh, JSON packets or routine ZIP transfer.')
    state=getattr(st,'session_state',None)
    if state is None:
        return  # Lightweight historical owner-panel consumers have no session.
    if not st.checkbox('Enable guided private NFL workflow',value=False,key='nfl_workflow_enabled'):
        revoke(state,keep_enabled=True)
        return  # Actual control-flow guard BEFORE storage/provider imports.
    try:
        from app_core.nfl_private_workflow import authenticate, production_service
        access=authenticate(state.get('publication_token',''),str(setting('PARLAYPICKER_PUBLISH_TOKEN')))
        service=service_factory(access) if service_factory is not None else production_service(setting,access)
        service.verify()
    except Exception as exc:
        revoke(state,keep_enabled=True)
        error(exc)
        return
    if service.store.synthetic:
        st.warning('SYNTHETIC PRIVATE FIXTURE — these are not authentic games, approvals, provider observations or predictions.')
    st.caption(LIMITATIONS)
    task=st.radio('Private NFL task',['Prepare a forecast','View saved forecasts'],key='nfl_workflow_task')
    try:
        if task=='View saved forecasts':
            render_saved(service)
        else:
            render_prepare(service)
    except Exception as exc:
        error(exc)


def render_saved(service):
    page=int(st.number_input('Saved forecast page',min_value=1,max_value=1_000_000,value=1,step=1,key='nfl_workflow_saved_page'))
    st.caption('100 saved sides per page. Change the page to reopen older forecasts; viewing performs no calculation or storage mutation.')
    rows=service.saved(page=page);by_id={row['id']:row['payload'] for row in rows}
    if not by_id:
        for key in ('nfl_workflow_saved_receipts','nfl_workflow_saved_loaded_ids','nfl_workflow_saved_selected'):
            st.session_state.pop(key,None)
        st.info('No saved forecasts in this owner’s configured private store. No calculation or database setup was performed.')
        return
    def label(key):
        row=by_id[key];s=row['scenario'];e=row['event']
        return e['away']+' at '+e['home']+' — '+s['selected_team']+f" {s['signed_handicap']:+g}"+' — calculated '+eastern(row['inference_time'])
    previous=st.session_state.get('nfl_workflow_saved_selected',[])
    st.session_state['nfl_workflow_saved_selected']=[key for key in previous if key in by_id]
    selected=st.multiselect('Saved private NFL forecasts',list(by_id),format_func=label,max_selections=8,key='nfl_workflow_saved_selected')
    if selected!=st.session_state.get('nfl_workflow_saved_loaded_ids',[]):
        st.session_state.pop('nfl_workflow_saved_receipts',None)
    if st.button('Load saved NFL forecasts',disabled=not selected,key='nfl_workflow_saved_load'):
        st.session_state['nfl_workflow_saved_receipts']=[by_id[key]['receipt'] for key in selected]
        st.session_state['nfl_workflow_saved_loaded_ids']=list(selected)
        st.session_state['nfl_workflow_saved_viewed_at']=service.clock()
    for receipt in st.session_state.get('nfl_workflow_saved_receipts',[]):
        show_saved(service,receipt,st.session_state['nfl_workflow_saved_viewed_at'])


def show_saved(service,receipt,viewed_at):
    item=service.view(receipt);p=item['payload'];q=p['input']['payload'];e=q['event'];s=q['scenario']
    st.write(e['away']+' AWAY at '+e['home']+' HOME — '+eastern(e['start']))
    st.write(s['selected_team']+(' HOME' if s['selected_team']==e['home'] else ' AWAY')+f" {s['signed_handicap']:+g}")
    st.metric('Uncalibrated native cover estimate',f"{p['probability']:.4%}")
    st.write(f"Selected-team margin projection: {p['native_output']['ml_projection']:+g} points before handicap. Residual scale: {p['native_output']['ml_residual_scale']:g} points.")
    st.write('Original calculation: '+eastern(p['inference_time']))
    st.caption('Viewed at '+eastern(viewed_at)+'. Static replay; this is not a new calculation or current quote. PRIVATE RESEARCH / PASS / ZERO STAKE.')
    with st.expander('Saved evidence details — '+receipt['result_sha256'][:10]):
        st.json(dict(result_sha256=receipt['result_sha256'],input_sha256=receipt['input_sha256'],original_inference_time=p['inference_time'],recorded_runtime=p['reader']['runtime']))
        st.download_button('Optional private evidence ZIP',data=service.store.download_native(receipt),
            file_name=receipt['result_sha256']+'.private-scenario.zip',mime='application/zip',key='nfl_workflow_download_'+receipt['result_sha256'])


def render_prepare(service):
    from app_core.nfl_workflow_inputs import instant, FINDINGS, ATTESTATION
    from app_core.nfl_workflow_store import digest
    state=st.session_state
    st.write('1. Schedule and completed scoring histories')
    st.caption('A valid retained original can serve multiple games without another scoring request. Original observation and source-permission times stay unchanged; the NFL reader has no invented maximum scoring-age rule.')
    if st.button('Load retained NFL games',key='nfl_workflow_load_games'):
        state['nfl_workflow_games']=service.games()
        state['nfl_workflow_source_id']=service.source()['id']
        invalidate_review(state)
        state.pop('nfl_workflow_prepared',None)
    if st.button('Prepare bounded scoring capture',key='nfl_workflow_prepare_source'):
        state['nfl_workflow_source_operation']=service.source_operation()['id']
    source_id=state.get('nfl_workflow_source_operation')
    if source_id:
        op=service.store.get(source_id,kind='operation')
        unavailable=service.store.spent('operation:'+source_id) or instant(service.clock())>=instant(op['payload']['window']['not_after'])-timedelta(seconds=60)
        st.info('At most two public CSV HTTP requests: one nflverse games.csv GET and one validated release-assets redirect. $0; zero Odds credits and zero CFBD/CBBD usage. No retry or fallback.')
        st.caption('Launch before '+eastern((instant(op['payload']['window']['not_after'])-timedelta(seconds=60)).isoformat())+'; capture completion before '+eastern(op['payload']['window']['not_after']))
        if unavailable:
            st.warning('This scoring operation is spent or expired. It will not be restarted.')
            if st.button('Prepare scoring-capture successor',key='nfl_workflow_successor_source'):
                state['nfl_workflow_source_operation']=service.source_operation(predecessor=source_id)['id'];st.rerun()
        with st.form('nfl_workflow_source_form_'+source_id):
            retained=st.checkbox('I reviewed the source permission for private retention.',key='nfl_workflow_source_retention_'+source_id)
            research=st.checkbox('I reviewed private research use of this original and the displayed request limits.',key='nfl_workflow_source_research_'+source_id)
            end=st.date_input('Private source-use permission ends after this UTC date',value=datetime.now(timezone.utc).date()+timedelta(days=7),key='nfl_workflow_source_end_'+source_id)
            submitted=st.form_submit_button('Authorize and capture NFL schedule/history',disabled=unavailable or not service.allow_acquisition)
        st.caption('Source: nflverse/nflverse-data, games.csv; CC-BY-4.0 attribution retained. Permission findings are your actual review, not an automated legal determination.')
        if submitted:
            permission_end=datetime.combine(end,time(23,59,59),tzinfo=timezone.utc).isoformat()
            service.capture_source(source_id,private_retention=retained,private_research=research,permission_end=permission_end)
            state['nfl_workflow_games']=service.games();state['nfl_workflow_source_id']=service.source()['id']
            state.pop('nfl_workflow_prepared',None);invalidate_review(state)
            st.success('Original schedule/history captured and preserved. No prediction was made.')
    catalog=state.get('nfl_workflow_games')
    if not catalog:
        st.info('Load retained games, or prepare and authorize the bounded scoring capture above. Nothing is acquired automatically.')
        return
    rows={row['event']['canonical_event_id']:row for row in catalog['games']}
    if catalog['rejected']:
        with st.expander('Unavailable schedule rows'):
            st.json(catalog['rejected'])
    def label(key):
        e=rows[key]['event'];return e['away']+' AWAY at '+e['home']+' HOME — '+eastern(e['start'])
    state['nfl_workflow_selected_games']=[key for key in state.get('nfl_workflow_selected_games',[]) if key in rows]
    selected=st.multiselect('Future non-neutral NFL games',list(rows),format_func=label,max_selections=4,key='nfl_workflow_selected_games')
    selection_scope=digest(dict(games=selected,source_id=state.get('nfl_workflow_source_id')))
    if selection_scope!=state.get('nfl_workflow_selection_scope'):
        invalidate_review(state);state.pop('nfl_workflow_prepared',None);state['nfl_workflow_selection_scope']=selection_scope
    if st.button('Prepare selected NFL games',disabled=not selected,key='nfl_workflow_prepare_games'):
        state['nfl_workflow_prepared']=service.prepare(selected)
        invalidate_review(state)
    prepared=state.get('nfl_workflow_prepared')
    if not prepared:return
    st.write('2. Review histories before acquiring a short-lived line')
    for entry in prepared:
        facts=entry['prepared'];e=facts['selected']['event'];source=facts['source']['object']
        st.write(e['away']+' at '+e['home']+' — '+str(len(facts['histories']['away']))+' away histories / '+str(len(facts['histories']['home']))+' home histories.')
        st.caption('Retained venue: '+str(facts['selected']['venue'])+'; kickoff '+eastern(e['start'])+'. Verify these original schedule facts before authorization.')
        st.caption('Original scoring observed '+eastern(source['receipt']['observed_at'])+'. Home/away slots are determined from each original row, never the selected side. No player-availability adjustment.')
        with st.expander('History and feature details — '+e['canonical_event_id']):
            st.json(dict(histories=facts['histories'],features=facts['features'],feature_order=facts['feature_order'],publisher_first_availability='UNKNOWN'))
        end=instant(entry['operation']['payload']['window']['not_after'])
        st.caption('Quote capture: launch before '+eastern((end-timedelta(seconds=60)).isoformat())+'; observe the complete original before '+eastern(end.isoformat())+'.')
    operation_ids=[entry['operation']['id'] for entry in prepared]
    scope=digest(operation_ids)
    unavailable=any(service.store.spent('operation:'+key) or instant(service.clock())>=instant(service.store.get(key)['payload']['window']['not_after'])-timedelta(seconds=60) for key in operation_ids)
    st.info(f"Quote inventory: at most {len(operation_ids)} Novig NFL spreads GETs, one per selected event; at most {len(operation_ids)} included Odds credits; $0 additional spending; no scoring/CFBD/CBBD requests. No retry, fallback or automatic replacement.")
    with st.expander('Exact quote requests and limits'):
        st.json([entry['operation']['payload'] for entry in prepared])
    if unavailable:
        st.caption('Existing quote operations are spent or expired. Reviewing their originals sends no request. Stale/unsupported lines remain unavailable for new calculation.')
        if st.button('Prepare new quote observations',key='nfl_workflow_successor_quotes'):
            predecessors={entry['prepared']['selected']['event']['canonical_event_id']:entry['operation']['id'] for entry in prepared}
            state['nfl_workflow_prepared']=service.prepare(selected,predecessors=predecessors)
            invalidate_review(state);st.rerun()
    with st.form('nfl_workflow_quote_form_'+scope):
        credit=st.number_input('Verified remaining included Odds credits',min_value=0,max_value=1000000,value=len(operation_ids),step=1,key='nfl_workflow_credit_'+scope)
        reset_date=st.date_input('Verified allowance reset date (UTC)',value=datetime.now(timezone.utc).date()+timedelta(days=1),key='nfl_workflow_reset_'+scope)
        reset_time=st.time_input('Verified allowance reset time (UTC)',value=time(0,0),key='nfl_workflow_reset_time_'+scope)
        private=st.checkbox('I verified current included entitlement and private use for these exact requests.',key='nfl_workflow_private_'+scope)
        no_overage=st.checkbox('I authorize these bounded requests only, with $0 additional spending, and am ready to review the lines promptly.',key='nfl_workflow_no_overage_'+scope)
        acquire=st.form_submit_button('Authorize and capture reviewed NFL lines',disabled=unavailable or not service.allow_acquisition)
    if acquire:
        service.capture_quotes(operation_ids,remaining_credits=int(credit),reset_at=datetime.combine(reset_date,reset_time,tzinfo=timezone.utc).isoformat(),private_use=private,no_overage=no_overage)
        state['nfl_workflow_pair_ids']=[row['id'] for row in service.stage(operation_ids)]
        st.success('Original lines captured. No numerical prediction yet.')
    if st.button('Review captured lines without another request',key='nfl_workflow_review_originals'):
        state['nfl_workflow_pair_ids']=[row['id'] for row in service.stage(operation_ids)]
    pair_ids=state.get('nfl_workflow_pair_ids')
    if not pair_ids:return
    st.write('3. Review and authorize both mathematical scenarios')
    pairs=[service.store.get(key,kind='scenario-pair') for key in pair_ids]
    from app_core.nfl_workflow_inputs import observed_pair
    for pair in pairs:
        value=pair['payload'];e=value['prepared']['selected']['event']
        observed=observed_pair(value['market']['object'],value['prepared'],service.clock())
        st.write(e['away']+' AWAY at '+e['home']+' HOME')
        for packet in value['packets']:
            s=packet['payload']['scenario'];quote=observed['pair'][s['selected_team']]
            st.write(s['selected_team']+f" {s['signed_handicap']:+g}"+f"; separately observed price {quote['price']:+g}")
        st.caption('Market updated '+eastern(observed['updated_at'])+'; observed '+eastern(observed['observed_at'])+'. Inference must remain within 15 minutes of the market update and before kickoff. Full-game/OT is your scenario assumption, not verified bookmaker settlement.')
        with st.expander('Exact scenario review subjects — '+e['canonical_event_id']):
            from app_core.nfl_scenario_research import subject
            st.json([dict(subject_sha256=subject(p['payload']),scenario=p['payload']['scenario']) for p in value['packets']])
    review_scope=digest(pair_ids)
    labels={
        'event_mapping':'I verified each displayed game, named home/away orientation, venue and kickoff.',
        'scenario_definition':'I select both displayed opposing half-point assumptions, completed full game including overtime.',
        'permissions':'I verified private source-use permissions for these exact scenarios.',
        'clocks_and_original_custody':'I reviewed the original observation times, freshness, private custody and actual pregame limits.',
        'feature_provenance':'I reviewed each team’s qualifying histories and derived features; home slots stay with the home team for both sides.',
        'model_binding':'I accept the unchanged fixed reliability 0.45 model/runtime and its unvalidated limitations.'}
    with st.form('nfl_workflow_final_form_'+review_scope):
        findings={key:st.checkbox(labels[key],key='nfl_workflow_review_'+review_scope+'_'+key) for key in FINDINGS}
        attest=st.checkbox('For each exact displayed scenario: “'+ATTESTATION+'”',key='nfl_workflow_review_'+review_scope+'_attest')
        limits=st.checkbox('My review relies on displayed originals/reproducible checks, not an independent technical audit. I accept uncalibrated private research, PASS and zero stake; no explicit QB/injury adjustment, EV or sportsbook recommendation.',key='nfl_workflow_review_'+review_scope+'_limits')
        context=st.text_area('Optional material player/QB context (recorded separately; never a numerical adjustment)',key='nfl_workflow_review_'+review_scope+'_context')
        calculate=st.form_submit_button('Authorize one calculation per side and save privately',disabled=not service.allow_inference)
    if calculate:
        results=service.calculate(pair_ids,findings,attestation=attest,limitations=limits,material_context=context or 'Not supplied; no explicit player adjustment.')
        state['nfl_workflow_result_receipts']=[item['receipt'] for item in results]
        state['nfl_workflow_result_viewed_at']=service.clock()
        st.success('Both sides saved before comparison. No wager or public publication was created.')
    receipts=state.get('nfl_workflow_result_receipts',[])
    if receipts:
        by_event={}
        for receipt in receipts:
            item=service.view(receipt);p=item['payload'];key=p['input']['payload']['event']['canonical_event_id']
            by_event.setdefault(key,[]).append(item)
        for values in by_event.values():
            if len(values)!=2:
                st.write('No conditional model preference: both opposing estimates are required.')
                continue
            if values[0]['payload']['probability']==values[1]['payload']['probability']:
                st.write('No conditional model preference: opposing native cover estimates are tied.')
                continue
            preferred=max(values,key=lambda item:item['payload']['probability'])
            s=preferred['payload']['input']['payload']['scenario']
            st.write('Conditional model preference: '+s['selected_team']+f" {s['signed_handicap']:+g}"+'. This is not established betting value.')
        for receipt in receipts:show_saved(service,receipt,state['nfl_workflow_result_viewed_at'])
