"""One finalized pick and one independently ranked side/total per game."""
import math
import pandas as pd


def text(row, *names):
    for name in names:
        value=row.get(name)
        if value is not None and pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ''


def number(row, name):
    value=pd.to_numeric(row.get(name),errors='coerce')
    return float(value) if pd.notna(value) and math.isfinite(value) else None


def identity(row):
    day=pd.to_datetime(text(row,'game_date','Local Date','Game Date'),errors='coerce',utc=True)
    return (text(row,'league','League').casefold(),text(row,'home_team','Home').casefold(),text(row,'away_team','Away').casefold(),day.date().isoformat() if pd.notna(day) else '')


def family_of(row):
    market=text(row,'market_type').lower()
    if market in {'total_over','total_under','totals_over','totals_under'}: return 'totals'
    if market in {'spread_home','spread_away','moneyline_home','moneyline_away','h2h_home','h2h_away','spreads_home','spreads_away'}: return 'sides'
    return ''


def per_game_board(board, candidates=None, family='overall'):
    if family not in {'overall','sides','totals'}: raise ValueError('Unknown family')
    if board is None or board.empty: return pd.DataFrame()
    candidates=candidates if isinstance(candidates,pd.DataFrame) else pd.DataFrame()
    rows=[]
    for _, final in board.iterrows():
        selected=final if family=='overall' else None
        reason='Final overall selection'
        if family!='overall':
            pool=[]
            key=identity(final)
            for _, candidate in candidates.iterrows():
                if family_of(candidate)!=family: continue
                run, other_run=text(final,'export_run_id'),text(candidate,'export_run_id')
                if run and other_run!=run: continue
                fid,cid=text(final,'matchup_id'),text(candidate,'matchup_id')
                if fid and cid:
                    if fid!=cid: continue
                    if all(key) and identity(candidate)!=key: continue
                elif not all(key) or identity(candidate)!=key: continue
                rank=number(candidate,'best_available_family_rank')
                if rank is None or rank<1: continue
                pool.append(candidate)
            # Without an event ID, same-team doubleheaders are ambiguous.
            event_ids={text(c,'matchup_id') for c in pool if text(c,'matchup_id')}
            runs={text(c,'export_run_id') for c in pool if text(c,'export_run_id')}
            if not text(final,'export_run_id') and len(runs)>1:
                pool=[]
            if not text(final,'matchup_id') and (len(event_ids)>1 or sum(identity(row)==key for _,row in board.iterrows())>1):
                pool=[]
            if pool:
                pool.sort(key=lambda c:(number(c,'best_available_family_rank'),number(c,'best_available_rank') or math.inf,text(c,'best_pick')))
                selected=pool[0]
                reason='Highest-ranked '+family+' candidate in this game'
            elif family_of(final)==family:
                selected=final
                reason='Final overall pick; no matching family audit available'
            else:
                reason='No matching ranked '+family+' candidate available; rerun analysis to refresh the audit'
        same = selected is not None and family_of(selected)==family_of(final) and text(selected,'best_pick')==text(final,'best_pick') and number(selected,'odds_american')==number(final,'odds_american') and text(selected,'odds_source')==text(final,'odds_source')
        # Only the exact final ticket can inherit the finalized approval or stake.
        source=final if same or family=='overall' else selected
        approved=source is not None and (same or family=='overall') and text(final,'Bettable').lower() in {'true','1','yes'} and (number(final,'Play_Stake') or 0)>0
        probability=None; basis='Unavailable'; edge=None; ev=None
        if source is not None:
            if same or family=='overall':
                probability=number(source,'production_win_probability');basis='Final production estimate'
                edge=number(source,'production_edge');ev=number(source,'production_expected_value')
            else:
                probability=number(source,'calibrated_probability');basis='Candidate calibrated estimate'
                ev=number(source,'expected_value')
                odds=number(source,'odds_american')
                if probability is not None and odds is not None and abs(odds)>=100:
                    break_even=100/(100+odds) if odds>0 else abs(odds)/(100+abs(odds))
                    edge=probability-break_even
            if probability is None or not 0<=probability<=1: probability=None;basis='Unavailable'
        approval_reason = text(final,'Production_Gate_Reason','Status_Reason','qualification_reason') if same or family=='overall' else ''
        if source is None:
            approval_reason = 'No matching ranked market available; refresh analysis'
        elif approved:
            approval_reason = 'Passed final wager checks with a positive approved stake'
        elif not (same or family=='overall'):
            approval_reason = 'Alternative selection; has not passed final wager and portfolio checks'
            if ev is not None and ev <= 0:
                approval_reason += '; estimated EV is not positive'
        elif not approval_reason or approval_reason.lower() == 'qualified':
            approval_reason = 'No final wager authorization with a positive approved stake'
        rows.append({'league':text(final,'league','League'),'matchup':text(final,'Away','away_team')+' at '+text(final,'Home','home_team'),
                     'matchup_id':text(final,'matchup_id'),'game_date':text(final,'Local Date','game_date'),
                     'start':text(final,'Commence (Local)','game_time_est'),
                     'pick':text(source,'display_pick','best_pick') if source is not None else 'No Bet — market unavailable',
                     'market_type':text(source,'market_type') if source is not None else '',
                     'odds':number(source,'odds_american') if source is not None else None,
                     'Bettable':approved,'Play_Stake':number(final,'Play_Stake') if approved else 0.0,
                     'selection_label': {'overall':'Best Overall','sides':'Best Side','totals':'Best Total'}[family] if source is not None else 'Unavailable',
                     'status':'APPROVED' if approved else 'PASS', 'win_probability':probability,'probability_basis':basis,
                     'edge':edge,'ev':ev,'selection_score':number(selected,'best_available_score') if selected is not None else None,
                     'reason':reason,'approval_reason':approval_reason,
                     'export_run_id':text(final,'export_run_id')})
    return pd.DataFrame(rows)
