"""One finalized pick and one independently ranked side/total per game."""
from app_core.quote_freshness import QUOTE_MAX_AGE_SECONDS
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


def exact_book_quote(row, book):
    """Require exact original team/market/line/price evidence from the requested bookmaker."""
    from app_core.prediction_evidence import bind_quote
    candidate = dict(row)
    # Reject only explicit cross-provider/event conflicts. Older quotes without
    # scoped identity retain their existing behavior; never modify the evidence.
    provider_id = text(row, 'provider_event_id')
    namespace = text(row, 'provider_namespace')
    if provider_id and namespace:
        import json
        try:
            quotes = json.loads(candidate.get('provider_quotes') or '[]')
        except (TypeError, ValueError):
            quotes = None
        if isinstance(quotes, list):
            candidate['provider_quotes'] = json.dumps([
                q for q in quotes if not isinstance(q, dict)
                or not (text(q, 'provider_event_id') and text(q, 'provider_namespace'))
                or (text(q, 'provider_event_id') == provider_id and text(q, 'provider_namespace') == namespace)
            ])
    candidate['opposing_odds_source'] = book
    bound = bind_quote(candidate)
    if not bound['quote_binding_verified'] or bound['quote_bookmaker'] != book:
        return None
    quoted = pd.to_datetime(bound['odds_recorded_at'], utc=True, errors='coerce')
    run = text(row, 'export_run_id')
    at = pd.to_datetime(run, utc=True, errors='coerce')
    if pd.isna(at):
        at = pd.to_datetime(run, format='%Y%m%dT%H%M%S.%fZ', utc=True, errors='coerce')
    if pd.isna(at) or not 0 <= (at-quoted).total_seconds() <= QUOTE_MAX_AGE_SECONDS:
        return None
    return bound['odds_recorded_at']


def espn_observed_quote(row):
    """Owner-authorized NCAAF research snapshot; not sportsbook update evidence."""
    from app_core.prediction_evidence import matching_quotes
    from core.selector_validation import timestamp
    if text(row, 'league', 'League').upper() != 'NCAAF' or text(row, 'odds_feed_source') != 'espn_ncaaf_fcs_scoreboard':
        return None
    matches = matching_quotes(dict(row, opposing_odds_source='draftkings'))
    if len(matches) != 1:
        return None
    quote = matches[0]
    if quote.get('observation_source') != 'espn_ncaaf_fcs_scoreboard' or quote.get('recorded_at'):
        return None
    observed = timestamp(quote.get('observed_at'))
    run = text(row, 'export_run_id')
    at = pd.to_datetime(run, utc=True, errors='coerce')
    if pd.isna(at):
        at = pd.to_datetime(run, format='%Y%m%dT%H%M%S.%fZ', utc=True, errors='coerce')
    if pd.isna(observed) or pd.isna(at) or not 0 <= (at-observed).total_seconds() <= QUOTE_MAX_AGE_SECONDS:
        return None
    return observed.isoformat()


def novig_quote(row):
    return exact_book_quote(row, 'novig')


def public_quote(row, college_fallback=False, *, nfl_fallback=False, research_fallback=False):
    books = [('novig', 'Novig')]
    league = text(row, 'league', 'League').upper()
    if ((college_fallback and league == 'NCAAF') or (nfl_fallback and league == 'NFL')
            or (research_fallback and league in {'MLB', 'WNBA'})):
        books += [('draftkings', 'DraftKings'), ('fanduel', 'FanDuel'), ('betmgm', 'BetMGM')]
    for book, label in books:
        at = exact_book_quote(row, book)
        if at:
            return label, at
    if college_fallback:
        observed = espn_observed_quote(row)
        if observed:
            return 'DraftKings', observed
    return None


def novig_unavailable_reason(final, candidates, family):
    import json
    matches=[]
    for _,row in candidates.iterrows():
        if text(row,'matchup_id')!=text(final,'matchup_id') or text(row,'export_run_id')!=text(final,'export_run_id'):
            continue
        if family!='overall' and family_of(row)!=family:
            continue
        matches.append(row)
    if not matches:
        return 'No ranked candidate evidence; refresh analysis'
    offered=False
    for row in matches:
        try:
            quotes=json.loads(row.get('provider_quotes') or '[]')
            offered |= any(q.get('book')=='novig' and q.get('market_type')==row.get('market_type') for q in quotes)
        except (TypeError,ValueError):
            pass
    return ('Novig quote could not be verified: line, price or timestamp mismatch' if offered
            else 'Novig market missing from this API snapshot')


def supported_unavailable_reason(final, candidates, family):
    """Explain untimestamped exact offers without treating retrieval as quote time."""
    import json
    offers = []
    for row in [final, *(r for _, r in candidates.iterrows())]:
        if (text(row, 'matchup_id') != text(final, 'matchup_id')
                or text(row, 'export_run_id') != text(final, 'export_run_id')
                or identity(row) != identity(final)):
            continue
        if family != 'overall' and family_of(row) != family:
            continue
        try:
            quotes = json.loads(row.get('provider_quotes') or '[]')
        except (TypeError, ValueError):
            continue
        if not isinstance(quotes, list):
            continue
        market = text(row, 'market_type')
        line = number(row, 'total_line' if market.startswith('total') else 'spread_line')
        price = number(row, 'odds_american')
        for quote in quotes:
            if not isinstance(quote, dict) or quote.get('book') not in {'novig', 'draftkings', 'fanduel', 'betmgm'}:
                continue
            same_line = market.startswith('moneyline') or (line is not None and number(quote, 'point') == line)
            if quote.get('market_type') == market and price is not None and number(quote, 'price') == price and same_line:
                offers.append((quote, text(row, 'odds_feed_source')))
    if offers and all(pd.isna(pd.to_datetime(q.get('recorded_at'), utc=True, errors='coerce')) for q, _ in offers):
        if all(q['book'] == 'draftkings' and source == 'espn_ncaaf_fcs_scoreboard' for q, source in offers):
            return 'DraftKings via ESPN: quote timestamp missing; freshness cannot be verified'
        return 'Sportsbook quote timestamp missing; freshness cannot be verified'
    return 'No exact fresh Novig or supported sportsbook quote in this analysis'


def per_game_board(board, candidates=None, family='overall', *, novig_only=False, college_fallback=False, nfl_fallback=False, research_fallback=False):
    if family not in {'overall','sides','totals'}: raise ValueError('Unknown family')
    if board is None or board.empty: return pd.DataFrame()
    candidates=candidates if isinstance(candidates,pd.DataFrame) else pd.DataFrame()
    from core.market_policy import production_market
    candidate_rows = [row for _, row in candidates.iterrows() if production_market(text(row, 'market_type'))]
    rows=[]
    for _, final in board.iterrows():
        league = text(final, 'league', 'League').upper()
        allow_fallback = ((college_fallback and league == 'NCAAF') or (nfl_fallback and league == 'NFL')
                          or (research_fallback and league in {'MLB', 'WNBA'}))
        selected=final if family=='overall' and not novig_only else None
        reason='Final overall selection'
        if family!='overall' or novig_only:
            pool=[]
            key=identity(final)
            for candidate in candidate_rows:
                if family!='overall' and family_of(candidate)!=family: continue
                run, other_run=text(final,'export_run_id'),text(candidate,'export_run_id')
                if run and other_run!=run: continue
                fid,cid=text(final,'matchup_id'),text(candidate,'matchup_id')
                if fid and cid:
                    if fid!=cid: continue
                    if all(key) and identity(candidate)!=key: continue
                elif not all(key) or identity(candidate)!=key: continue
                # Bind quotes only after matching the run and game identity.
                if novig_only and not public_quote(candidate, college_fallback, nfl_fallback=nfl_fallback, research_fallback=research_fallback): continue
                rank=number(candidate,'best_available_rank' if family=='overall' else 'best_available_family_rank')
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
                pool.sort(key=lambda c:((0 if not novig_only or novig_quote(c) else 1),number(c,'best_available_rank' if family=='overall' else 'best_available_family_rank'),number(c,'best_available_rank') or math.inf,text(c,'best_pick')))
                strict_contract=final.get('wager_contract')
                trial_contract=final.get('controlled_trial_contract')
                authority_contract=strict_contract if isinstance(strict_contract,dict) and strict_contract.get('production_eligible') else trial_contract
                exact_final=[c for c in pool if text(c,'best_pick')==text(final,'best_pick') and number(c,'odds_american')==number(final,'odds_american') and text(c,'market_type')==text(final,'market_type')]
                selected=exact_final[0] if family=='overall' and isinstance(authority_contract,dict) and len(exact_final)==1 else pool[0]
                reason='Highest-ranked '+family+' candidate in this game'
            elif (family=='overall' or family_of(final)==family) and (not novig_only or public_quote(final, college_fallback, nfl_fallback=nfl_fallback, research_fallback=research_fallback)):
                selected=final
                reason='Final overall pick; no matching family audit available'
            else:
                reason='No matching ranked '+family+' candidate available; rerun analysis to refresh the audit'
        if selected is not None and not production_market(text(selected, 'market_type')):
            selected = None
            reason = 'No qualifying spread or total; moneyline is context only'
        quote = public_quote(selected, college_fallback, nfl_fallback=nfl_fallback, research_fallback=research_fallback) if selected is not None and novig_only else None
        fallback_selected = quote is not None and quote[0] != 'Novig'
        observed_selected = bool(quote and quote[0] == 'DraftKings' and not exact_book_quote(selected, 'draftkings') and espn_observed_quote(selected))
        same = selected is not None and family_of(selected)==family_of(final) and text(selected,'best_pick')==text(final,'best_pick') and number(selected,'odds_american')==number(final,'odds_american') and text(selected,'odds_source')==text(final,'odds_source')
        canonical=final.get('wager_contract')
        trial_contract=final.get('controlled_trial_contract')
        if not (isinstance(canonical,dict) and canonical.get('production_eligible')) and isinstance(trial_contract,dict) and trial_contract.get('trial_eligible'):
            canonical=trial_contract
        if isinstance(canonical,dict) and selected is not None:
            same = (text(selected,'best_pick')==canonical.get('selection') and number(selected,'odds_american')==canonical.get('odds') and text(selected,'market_type')==canonical.get('market_type') and (not novig_only or bool(quote and quote[0].casefold()==str(canonical.get('sportsbook') or '').casefold())))
        # Only the exact final ticket can inherit the finalized approval or stake.
        source=selected if selected is None or novig_only else final if same or family=='overall' else selected
        final_ticket = same or (family=='overall' and not novig_only)
        research_only_fallback = fallback_selected and league in {'MLB', 'WNBA'}
        approved=not research_only_fallback and not observed_selected and (not fallback_selected or (text(final,'production_eligible').lower() in {'true','1','yes'} and text(final,'wager_approved').lower() in {'true','1','yes'})) and source is not None and final_ticket and text(final,'Bettable').lower() in {'true','1','yes'} and (number(final,'Play_Stake') or 0)>0
        trial=not approved and not observed_selected and not fallback_selected and source is not None and final_ticket and isinstance(trial_contract,dict) and trial_contract.get('trial_eligible') is True and (number(trial_contract,'recommended_bet_amount') or 0)>0 and same
        probability=None; basis='Unavailable'; edge=None; ev=None
        if source is not None:
            if trial:
                probability=number(trial_contract,'estimated_probability');basis='Controlled-trial estimate (unvalidated)'
                edge=number(trial_contract,'estimated_price_edge');ev=number(trial_contract,'estimated_expected_value')
            elif final_ticket:
                probability=number(final,'production_win_probability');basis='Final production estimate'
                edge=number(final,'production_edge');ev=number(final,'production_expected_value')
            else:
                probability=number(source,'calibrated_probability');basis='Candidate calibrated estimate'
                ev=number(source,'expected_value')
                odds=number(source,'odds_american')
                if probability is not None and odds is not None and abs(odds)>=100:
                    break_even=100/(100+odds) if odds>0 else abs(odds)/(100+abs(odds))
                    edge=probability-break_even
            # New runs expose the same probability that chose the candidate.
            # Production risk adjustments still govern funding independently.
            if not trial and text(source, 'best_available_selection_policy') == 'probability-first-v1':
                probability=number(source, 'best_available_probability')
                basis='Candidate win estimate (pair-normalized)' if text(source, 'best_available_probability_source') == 'calibrated_probability_pair_normalized' else 'Candidate win estimate'
                odds=number(source, 'odds_american')
                edge=None;ev=None
                if probability is None or not 0<=probability<=1: approved=False
                if probability is not None and 0<=probability<=1 and odds is not None and abs(odds)>=100:
                    decimal=1+odds/100 if odds>0 else 1+100/abs(odds)
                    edge=probability-1/decimal
                    ev=probability*decimal-1
            if not trial and league == 'NFL' and probability is not None:
                if text(source, 'ml_probability_source').lower() == 'score-distribution-v1:nfl':
                    basis = 'NFL score model + market (recent form and injury context; unvalidated)'
                elif text(source, 'selection_probability_source') == 'football_research_blend_no_independent_model':
                    basis = 'Market-implied estimate (NFL context model unavailable)'
            if probability is None or not 0<=probability<=1: probability=None;basis='Unavailable'
        approval_reason = text(final,'Production_Gate_Reason','Status_Reason','qualification_reason') if final_ticket else ''
        if source is None:
            approval_reason = 'No matching ranked market available; refresh analysis'
        elif trial:
            approval_reason = str(trial_contract.get('reason') or 'Owner-authorized controlled trial')
        elif fallback_selected and not approved:
            approval_reason = ('ESPN snapshot; sportsbook update time unknown; research selection, not wager approval' if observed_selected else 'Sportsbook fallback; research selection, not wager approval')
        elif approved:
            approval_reason = 'Passed final wager checks with a positive approved stake'
        elif not final_ticket:
            approval_reason = 'Alternative selection; has not passed final wager and portfolio checks'
            if ev is not None and ev <= 0:
                approval_reason += '; estimated EV is not positive'
        elif not approval_reason or approval_reason.lower() == 'qualified':
            approval_reason = 'No final wager authorization with a positive approved stake'
        if novig_only:
            from app_core.recommendation_quality import quality_reason, positive_price_edge
            quality = quality_reason(final)
            positive = bool(quote and positive_price_edge(probability, number(source, 'odds_american') if source is not None else None, ev))
            if trial and not positive:
                trial = False
                approval_reason = 'Controlled trial held: no verified positive estimated edge at the quoted price'
            if approved and (quality or not positive):
                approved = False
                approval_reason = quality or 'No verified positive estimated edge at the quoted price'
        from app_core.total_signal_quality import public_fields as total_quality_fields
        rows.append({**(total_quality_fields(source) if source is not None else {}), 'league':text(final,'league','League'),'matchup':text(final,'Away','away_team')+' at '+text(final,'Home','home_team'),
                     'matchup_id':text(final,'matchup_id'),'game_date':text(final,'Local Date','game_date'),
                     'start':text(final,'Commence (Local)','game_time_est'),
                     'pick':text(source,'display_pick','best_pick') if source is not None else ('Sportsbook quote unavailable' if allow_fallback else 'Novig quote unavailable' if novig_only else 'No Bet — market unavailable'),
                     'market_type':text(source,'market_type') if source is not None else '',
                     'odds':number(source,'odds_american') if source is not None else None,
                     'Bettable':approved,'Play_Stake':number(final,'Play_Stake') if approved else 0.0,
                     'Trial_Stake':number(trial_contract,'recommended_bet_amount') if trial else 0.0,
                     'selection_label': {'overall':'Best Overall','sides':'Best Side','totals':'Best Total'}[family] if source is not None else 'Unavailable',
                     'status':'APPROVED' if approved else 'TRIAL' if trial else 'PASS', 'win_probability':probability,'probability_basis':basis,
                     'edge':edge,'ev':ev,'selection_score':number(selected,'best_available_score') if selected is not None else None,
                     'reason':reason,'approval_reason':approval_reason,
                     **({'qualification_reason':approval_reason, 'quote_source':quote[0] if quote else 'Unavailable', 'quote_time':quote[1] if quote else '', 'quote_reason':('Sportsbook fallback: no eligible Novig candidate in this view' if fallback_selected else '') if source is not None else (supported_unavailable_reason(final,candidates,family) if allow_fallback else novig_unavailable_reason(final,candidates,family))} if novig_only else {}),
                     **({'quote_time_basis':'espn_observed'} if observed_selected else {}),
                     **({k:final[k] for k in ('maturity','gemini_review_status','gemini_reviewed_at','gemini_review_model','gemini_review_input_hash','gemini_verified_context','gemini_supporting_evidence','gemini_missing_information','conservative_ev','espn_event_id','mlb_game_pk','game_number') if k in final} if final_ticket else {}),
                     **({'wager_contract':final['wager_contract']} if final_ticket and isinstance(final.get('wager_contract'),dict) else {}),
                     **({'controlled_trial_contract':trial_contract} if trial else {}),
                     **({
                         'nfl_context_status': text(source, 'nfl_context_status'),
                         'home_recent_result': text(source, 'feature_home_last_game_summary'),
                         'away_recent_result': text(source, 'feature_away_last_game_summary'),
                         'home_injury_context': text(source, 'injury_home_summary'),
                         'away_injury_context': text(source, 'injury_away_summary'),
                         'injury_context_source': text(source, 'injury_context_source'),
                         'injury_context_status': text(source, 'injury_context_status'),
                     } if source is not None and league == 'NFL' else {}),
                     'export_run_id':text(final,'export_run_id')})
    return pd.DataFrame(rows)
