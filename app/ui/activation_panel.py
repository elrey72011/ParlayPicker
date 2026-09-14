"""Authenticated owner tools. No provider calls occur during rendering."""
import json,os
from pathlib import Path
from datetime import datetime,timezone
import streamlit as st
from core.sport_policy import SPORTS
from core.exposure_ledger import append,snapshot,digest

def render(games):
    if not st.checkbox('Show wager readiness and exposure',key='activation_tools_visible'):return
    st.caption('Recommendation settings and records only. Placement happens manually outside ParlayPicker.')
    rows=[]
    for sport in SPORTS:
        path=Path('output/sport-validation',sport+'.json')
        value=json.loads(path.read_text()) if path.exists() else {}
        rows.append({'Sport':sport,'State':value.get('deployment_state','UNVALIDATED'),
            'Validation':value.get('validation_hash'),'Expires':value.get('expires_at'),
            'Strict rows':value.get('strict_n',0),'Effective evidence':value.get('effective_n',0),
            'Closing quotes':value.get('metrics',{}).get('closing_n',0),
            'Model':value.get('versions',{}).get('model_version'),
            'Calibration':value.get('versions',{}).get('calibration_version'),
            'Blockers':'; '.join(value.get('blockers',['Run sport validation']))})
    st.dataframe(rows,hide_index=True)
    from app_core.evidence_health import evidence_health
    st.json(evidence_health())
    policy_path=Path(os.environ.get('PARLAYPICKER_WAGER_POLICY_PATH','data/policies/active_wager_policy.json'))
    ledger=os.environ.get('PARLAYPICKER_EXPOSURE_LEDGER','data/exposure/exposure.sqlite3')
    with st.expander('Review and activate validated policy'):
        uploaded=st.file_uploader('Candidate policy JSON',type=['json'],key='activation_candidate')
        if uploaded:
            try:
                from core.activation_policy import verify,activate
                candidate=verify(json.loads(uploaded.getvalue()))
                st.json({s:{'state':v['deployment_state'],'validation_id':v['validation_id']} for s,v in candidate['sports'].items()})
                confirmed=st.checkbox('I reviewed this candidate policy',key='activation_confirm')
                if st.button('Activate validated policy',disabled=not confirmed,key='activation_apply'):
                    source=policy_path.parent/('candidate-'+candidate['policy_hash']+'.json');source.parent.mkdir(parents=True,exist_ok=True);source.write_text(json.dumps(candidate),encoding='utf-8')
                    activate(source,policy_path,confirm=True);st.success('Policy activated. Refresh picks to evaluate current candidates.')
            except (ValueError,KeyError,TypeError):st.error('Candidate validation failed; no policy activated.')
    with st.expander('Configure bankroll and exposure limits'):
        with st.form('exposure_configuration'):
            bankroll=st.number_input('Confirmed bankroll',min_value=0.,value=0.)
            unit=st.number_input('Dollar value per unit',min_value=0.,value=0.)
            caps={k:st.number_input(k.replace('_',' ').title()+' (fraction)',min_value=0.,max_value=1.,value=0.) for k in ('total_cap','daily_cap','weekly_cap','game_cap','team_cap')}
            if st.form_submit_button('Save bankroll and limits'):
                try:append(ledger,dict(status='CONFIGURED',bankroll=bankroll,unit_value=unit,currency='USD',**caps),confirmed=True);st.success('Configuration recorded.')
                except ValueError as e:st.error(str(e))
    try:st.caption('Fresh exposure: '+json.dumps(snapshot(ledger)['committed']))
    except ValueError:st.info('Configure bankroll and limits before any funded recommendation.')
    render_placement_form(games,ledger)
    with st.expander('Record a manually placed wager'):
        st.caption('Upload the actual wager record, including source snapshot and every underlying game/team. This records exposure only; it does not place a wager.')
        uploaded=st.file_uploader('Placement or settlement JSON',type=['json'],key='placement_record')
        confirm=st.checkbox('I confirm this reflects my actual external wager activity',key='placement_confirm')
        if uploaded:
            try:
                value=json.loads(uploaded.getvalue());st.json(value)
                if st.button('Record confirmed activity',disabled=not confirm,key='placement_apply'):
                    append(ledger,value,confirmed=True);st.success('Activity recorded in the append-only exposure ledger.')
            except (ValueError,KeyError,TypeError) as e:st.error(str(e))


def render_placement_form(games, ledger):
    import ast, uuid
    from core.owner_wager_records import placement_record
    if games is None or games.empty:return
    choices={str(i):r for i,r in games.iterrows() if isinstance(r.get('wager_contract'),dict) and r['wager_contract'].get('production_eligible') is True}
    if not choices:return
    with st.expander('Mark a recommended wager as placed'):
        key=st.selectbox('Recommended wager',list(choices),format_func=lambda k:choices[k]['wager_contract']['selection'])
        row=choices[key];c=row['wager_contract']
        with st.form('mark_recommendation_placed'):
            book=st.text_input('Actual sportsbook',value=c.get('sportsbook',c.get('book','')) or '')
            line=st.number_input('Actual line',value=float(c.get('line') or 0))
            odds=st.number_input('Actual American odds',value=float(c.get('odds') or 0))
            stake=st.number_input('Actual stake dollars',min_value=0.,value=0.)
            reference=st.text_input('Your unique ticket/reference ID')
            confirmed=st.checkbox('I manually placed this wager and confirm the actual details')
            submit=st.form_submit_button('Record placed wager')
        if submit:
            try:
                if not confirmed:raise ValueError('Confirm the actual placement first.')
                teams=row.get('team_ids');teams=ast.literal_eval(teams) if isinstance(teams,str) else teams
                record=placement_record(c,sportsbook=book,line=line,odds=odds,stake=stake,bet_id=reference,snapshot_id=row.get('snapshot_id'),team_ids=teams)
                if record['value_warning']:st.warning(record['value_warning'])
                append(ledger,record,confirmed=True);st.success('Actual exposure recorded. No wager was placed by the app.')
            except (ValueError,KeyError,TypeError) as e:st.error(str(e))
    from core.exposure_ledger import events
    history=events(ledger);latest={e['bet_id']:e for e in history if e.get('bet_id')}
    open_ids=[k for k,v in latest.items() if v['status']=='COMMITTED']
    if open_ids:
        with st.expander('Settle an open exposure record'):
            bet=st.selectbox('Open record',open_ids)
            status=st.selectbox('Record status',['SETTLED','VOID','CANCELLED'])
            confirm=st.checkbox('I confirm this record is no longer open',key='settle_confirm')
            if st.button('Record settlement',disabled=not confirm):
                append(ledger,{'status':status,'bet_id':bet},confirmed=True);st.success('Settlement appended.')


def render_ticket_confirmation(package):
    tickets=package.get('parlays',[])
    tickets=[t for t in tickets if t.get('parlay_id') and t.get('legs')]
    if not tickets:return
    with st.expander('Confirm actual combined parlay price'):
        st.caption('Enter the current combined price from the same sportsbook. This calculates a recommendation; it never places the ticket.')
        selected=st.selectbox('Saved ticket',range(len(tickets)),format_func=lambda i:tickets[i]['parlay_id'])
        st.json(tickets[selected])
        with st.form('confirm_actual_parlay'):
            book=st.text_input('Ticket sportsbook')
            price=st.number_input('Actual combined decimal odds',min_value=1.01,value=1.01)
            checked=st.checkbox('I verified this current combined price and unchanged legs')
            submit=st.form_submit_button('Recheck parlay value and exposure')
        if submit:
            try:
                from core.owner_parlay_confirmation import confirm_ticket
                from app_core.prediction_evidence import database_path
                result=confirm_ticket(package,tickets[selected]['parlay_id'],book,price,
                    os.environ.get('PARLAYPICKER_WAGER_POLICY_PATH','data/policies/active_wager_policy.json'),
                    os.environ.get('PARLAYPICKER_EXPOSURE_LEDGER','data/exposure/exposure.sqlite3'),database_path(),confirmed=checked)
                st.json(result['recommendation'])
            except (ValueError,KeyError,TypeError,OSError,StopIteration) as e:st.error('No recommendation: '+str(e))
