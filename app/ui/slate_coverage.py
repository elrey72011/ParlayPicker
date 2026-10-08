"""Retained slate diagnostics and owner downloads; no analysis or acquisition."""
import json
import pandas as pd
import streamlit as st


def render_coverage(report, *, key):
    if not report:
        return
    with st.expander('Independent slate coverage', expanded=False):
        st.caption(f"{report['selected_date']} · America/New_York · as of {report['as_of']} · inventory {report['inventory_status']}")
        st.write(report['counts'])
        if not report['fully_reconciled']:
            st.warning('The retained inventory is incomplete or unavailable. Internal row equality does not prove the full schedule.')
        frame = pd.DataFrame(report['decisions'])
        if not frame.empty:
            st.dataframe(frame[['league', 'away_team', 'home_team', 'original_start', 'coverage_decision_state', 'explanation']], hide_index=True)
        st.dataframe(pd.DataFrame(report['inventory_scope']), hide_index=True)
        if report['orphan_events']:
            st.caption('Unmatched provider/candidate/final events are outside the scheduled denominator.')
            st.dataframe(pd.DataFrame(report['orphan_events']), hide_index=True)
        st.download_button('Download slate coverage JSON', json.dumps(report, indent=2, allow_nan=False),
                           'slate-coverage.json', 'application/json', key=key+'_json')
        for col in frame:
            if frame[col].map(lambda value: isinstance(value, (dict, list))).any():
                frame[col] = frame[col].map(lambda value: json.dumps(value, sort_keys=True, allow_nan=False))
        st.download_button('Download slate coverage CSV', frame.to_csv(index=False),
                           'slate-coverage.csv', 'text/csv', key=key+'_csv')
