"""Results overview: reads local evidence and never refreshes providers."""
import json
import pandas as pd
import streamlit as st
from app_core.results_overview import summarize_results


def render_recap_overview(container, frame):
    with container.container():
        st.subheader('Game results overview')
        st.caption('Scope: the loaded recap only, not a lifetime record. Counts are selections; win rate excludes pushes, voids and unresolved rows.')
        summary = summarize_results(frame)
        if summary.empty:
            st.info('Load a graded game export below or refresh final scores to populate this record.')
        else:
            sports = ['All sports', *sorted(summary['Sport'].unique())]
            if st.session_state.get('results_overview_sport', 'All sports') not in sports:
                st.session_state['results_overview_sport'] = 'All sports'
            sport = st.selectbox('Sport', sports, key='results_overview_sport')
            visible = summary if sport == 'All sports' else summary[summary.Sport.eq(sport)]
            for _, row in visible.iterrows():
                with st.container(border=True):
                    st.markdown('**'+row['Sport']+' · '+row['Record']+'**')
                    a, b, c = st.columns(3)
                    a.metric('Wins · Losses · Pushes', f"{row['Wins']} · {row['Losses']} · {row['Pushes']}")
                    b.metric('Win rate', f"{row['Win rate']:.1%}" if pd.notna(row['Win rate']) else 'Not available')
                    c.metric('Pending / unresolved', str(row['Pending / unresolved']))
                    st.caption(f"Sample: {row['Selections']} selections · {row['Decisions (W+L)']} win/loss decisions · {row['Void / DNP']} void/DNP")
                    profit = f"{row['Paper profit (units)']:+.2f} units" if pd.notna(row['Paper profit (units)']) else 'Not available'
                    roi = f"{row['Paper ROI']:+.1%}" if pd.notna(row['Paper ROI']) else 'Not available'
                    st.write('Paper profit: '+profit+' · Paper ROI: '+roi)
                    st.caption(f"Return sample: {row['Priced settled rows']} priced settled selections · {row['Unpriced settled rows']} settled selections excluded for missing/invalid odds")
            st.download_button('Download game results overview', summary.to_csv(index=False), 'game-results-overview.csv', 'text/csv', key='download_results_overview')
        st.caption('Paper returns assume one unit per priced settled selection at the exported odds. Missing or invalid odds are excluded, never replaced with a default price. Research selections were not approved wagers.')
        st.info('Actual betting returns: unavailable. App approval and suggested stakes do not prove that a bet was placed.')


def render_evidence_overview():
    from app_core.evidence_health import evidence_health
    health = evidence_health()
    with st.expander('Saved evidence & operational status', expanded=False):
        st.caption('Local records visible to this Streamlit process. Opening this panel does not fetch scores, capture predictions, or restore Drive backups.')
        remote = health['remote_storage']
        st.table(pd.DataFrame([
            {'Evidence': 'General prediction store', 'Status': health['status'], 'Latest recorded time (UTC)': health.get('latest_generated_at') or 'Not recorded'},
            {'Evidence': 'General score revision', 'Status': str(health['score_revisions'])+' revisions', 'Latest recorded time (UTC)': health.get('latest_score_recorded_at') or 'Not recorded'},
            {'Evidence': 'General Drive sync (this process)', 'Status': remote['status'], 'Latest recorded time (UTC)': remote.get('last_success_at') or 'Not recorded'},
        ]).set_index('Evidence'))
        st.caption('A general-store sync does not verify the separate sport stores or the scheduler. Missing timestamps mean unknown, not a successful update.')
        from app_core import mlb_prospective, ncaaf_prospective, nfl_market
        for sport, module in [('MLB', mlb_prospective), ('NCAAF', ncaaf_prospective), ('NFL', nfl_market)]:
            st.markdown('**'+sport+' research evidence**')
            from app_core.prediction_evidence import database_path
            filename = {'MLB': 'mlb-prospective.sqlite3', 'NCAAF': 'ncaaf-prospective.sqlite3', 'NFL': 'nfl-market.sqlite3'}[sport]
            location = database_path().with_name(filename)
            if not location.exists():
                st.info('No local '+sport+' evidence store. Restore it from Settings & research to inspect saved records.')
                continue
            try:
                report = module.report()
                records = module.store.records()
                def latest(kind):
                    valid = [pd.to_datetime(r.get('created_at'), utc=True, errors='coerce') for r in records if r['kind']==kind and (kind!='capture' or r.get('data',{}).get('events'))]
                    valid = [v for v in valid if pd.notna(v)]
                    return max(valid).isoformat() if valid else 'Not recorded'
                st.caption('Last nonempty capture: '+latest('capture')+' · Last saved scores: '+latest('scores'))
                if sport == 'NFL':
                    st.write(f"Captured games: {report['captured_games']} · Graded games: {report['graded_games']} · Pending: {report['captured_games']-report['graded_games']}")
                    st.caption('Market observations only. No model win rate or betting returns.')
                    if report['unresolved_past_score_window']:
                        st.warning(f"{len(report['unresolved_past_score_window'])} game(s) are unresolved beyond the score lookup window.")
                elif sport == 'MLB':
                    st.write(f"Captured game/cohort pairs: {report['captured_games_by_cohort']} · Graded pairs: {report['graded_games_by_cohort']} · Pending pairs: {report['captured_games_by_cohort']-report['graded_games_by_cohort']}")
                    st.caption('Paired score forecasts, not betting selections. Errors remain separated by frozen cohort and model.')
                    rows = [{'Cohort': s['cohort'], 'Verified games': s['games'], 'Model': model, 'Margin MAE': mae['margin'], 'Total MAE': mae['total']} for s in report['summary'] for model, mae in s['mae'].items()]
                    if rows: st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
                else:
                    graded_pairs = len({(r['model_id'],r['game_id']) for r in report['results']})
                    st.write(f"Captured game/cohort pairs: {report['captured_games_by_cohort']} · Pairs with graded selections: {graded_pairs} · Pending pairs: {report['captured_games_by_cohort']-graded_pairs}")
                    st.caption('Paper evaluation. Models and frozen cohorts are separate samples; their selections must not be pooled into one win rate.')
                    rows = []
                    for item in report['summary']:
                        bets = [r for r in report['results'] if r['model_id']==item['model_id'] and r['model']==item['model'] and r['paper_unit']>0]
                        rows.append({'Cohort': item['model_id'], 'Model': item['model'], 'Paper selections': len(bets),
                                     'Wins': sum(r['outcome']=='win' for r in bets), 'Losses': sum(r['outcome']=='loss' for r in bets),
                                     'Pushes': sum(r['outcome']=='push' for r in bets), 'Paper profit (units)': item['paper_profit_units'],
                                     'Paper win rate': item['paper_hit_rate']})
                    if rows: st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
                st.download_button('Download '+sport+' saved report', json.dumps(report, indent=2), sport.lower()+'-results-evidence.json', 'application/json', key='results_evidence_'+sport)
            except Exception:
                st.warning(sport+' evidence is unavailable or failed validation. Check its research panel; no zero-performance claim is inferred.')
