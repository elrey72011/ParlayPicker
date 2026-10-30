"""Proposed owner-only static NCAAF reopening; no execution/configuration API."""
from datetime import datetime, timezone
import hmac
from pathlib import Path
import re

import streamlit as st

PREFIX = 'ncaaf_saved_refresh_'
NAME = re.compile(r'[0-9a-f]{64}\.private-refresh\.json')
MAX_RECORDS = 256


def revoke(state):
    for key in tuple(state):
        if isinstance(key, str) and key.startswith(PREFIX):
            state.pop(key, None)


def owner_allowed(setting, state):
    expected = setting('PARLAYPICKER_PUBLISH_TOKEN', '')
    supplied = state.get('publication_token', '')
    return (type(expected) is str and len(expected) >= 16 and type(supplied) is str
            and hmac.compare_digest(expected.encode(), supplied.encode()))


def catalog(root):
    # This is an independently configured local custody root, never an uploaded
    # path or a root copied out of a self-hashed result body.
    from app_core.ncaaf_pilot_setup import inspect_security
    from app_core.ncaaf_staged_acquisition import MAX_BUNDLE
    root = Path(root)
    if not root.is_absolute() or inspect_security(root)['status'] != 'VERIFIED':
        raise ValueError('NCAAF_SAVED_CUSTODY_UNAVAILABLE')
    selected = []
    for path in root.glob('*.private-refresh.json'):
        if (not NAME.fullmatch(path.name) or path.is_symlink()
                or path.resolve().parent != root.resolve() or not path.is_file()
                or not 0 < path.stat().st_size <= MAX_BUNDLE):
            raise ValueError('NCAAF_SAVED_RECORD_REJECTED')
        selected.append(path.name)
        if len(selected) > MAX_RECORDS:
            raise ValueError('NCAAF_SAVED_CATALOG_LIMIT')
    return sorted(selected)


def read(root, name):
    from app_core import ncaaf_private_refresh as refresh
    from app_core.ncaaf_staged_acquisition import MAX_BUNDLE
    root = Path(root)
    if name not in catalog(root):
        raise ValueError('NCAAF_SAVED_RECORD_REJECTED')
    path = root / name
    with path.open('rb') as stream:
        raw = stream.read(MAX_BUNDLE + 1)
    if not 0 < len(raw) <= MAX_BUNDLE:
        raise ValueError('NCAAF_SAVED_RECORD_REJECTED')
    # Released static reader reauthenticates SQLite originals and exact CSV;
    # research_replay.read_export opens SQLite with URI mode=ro. Never initialize.
    return refresh.read_private_export(raw, root=root,
        database=root / 'ncaaf-private-refresh.sqlite')


def render(setting):
    state = getattr(st, 'session_state', None)
    if state is None:
        return
    if not owner_allowed(setting, state):
        revoke(state)
        return
    st.subheader('View saved NCAAF Refresh results')
    st.caption('Static replay of this local private store. No acquisition, '
               'calculation, new approval or database mutation. Original clocks stay unchanged.')
    if not st.checkbox('Show saved NCAAF Refresh results', value=False,
                       key=PREFIX+'enabled'):
        for key in (PREFIX+'loaded', PREFIX+'selected', PREFIX+'viewed'):
            state.pop(key, None)
        return  # Actual control-flow guard before custody inspection or SQLite.
    configured = setting('PARLAYPICKER_NCAAF_CUSTODY_ROOT', '')
    if type(configured) is not str or not configured:
        st.info('Configure the existing protected local NCAAF custody root. '
                'This viewer does not create storage or enable Linux custody.')
        return
    if state.get(PREFIX+'root') != configured:
        for key in (PREFIX+'loaded', PREFIX+'selected', PREFIX+'viewed'):
            state.pop(key, None)
        state[PREFIX+'root'] = configured
    try:
        names = catalog(configured)
        if not names:
            st.info('No saved normal-Refresh results in this configured private store.')
            return
        state[PREFIX+'selected'] = [x for x in state.get(PREFIX+'selected', []) if x in names]
        selected = st.multiselect('Saved NCAAF comparisons', names, max_selections=4,
                                 key=PREFIX+'selected')
        if selected != state.get(PREFIX+'loaded', []):
            state.pop(PREFIX+'loaded', None)
        if st.button('Load saved NCAAF results', disabled=not selected, key=PREFIX+'load'):
            # A load selection is not a cached authentication verdict. Every
            # render below invokes the original static reader afresh.
            state[PREFIX+'loaded'] = list(selected)
            state[PREFIX+'viewed'] = datetime.now(timezone.utc).isoformat()
        for name in state.get(PREFIX+'loaded', []):
            rows = read(configured, name)
            event = rows[0]['event']
            st.write(event['away_team']+' AWAY at '+event['home_team']+' HOME')
            st.caption('Viewed at '+state[PREFIX+'viewed']+
                       '. Original inference times appear in each saved row. '
                       'UNCALIBRATED PRIVATE CONDITIONAL RESEARCH / PASS / ZERO STAKE.')
            st.dataframe(rows, hide_index=True)
    except Exception:
        state.pop(PREFIX+'loaded', None)
        st.error('Saved NCAAF evidence is missing, changed, incompatible or inaccessible. '
                 'No acquisition or recalculation was started.')
