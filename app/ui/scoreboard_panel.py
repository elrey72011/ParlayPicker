"""Results-tab scoreboard: is the model beating the market? (roadmap E1.3)

Display only. It reads graded exports and never changes picks, stakes,
locks or publication. The bundled baseline in data/backtest_exports is always
included. Graded exports uploaded here (Drive downloads: .csv, .xlsx or .txt)
are added for this session only and are not saved.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import streamlit as st

from app_core.evaluation import scoreboard

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'data' / 'backtest_exports'


@st.cache_data(show_spinner=False)
def _board(uploads: tuple[tuple[str, bytes], ...], baseline: str):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for path in Path(baseline).iterdir():
            if path.suffix in ('.csv', '.xlsx', '.txt'):
                shutil.copy(path, folder / path.name)
        for name, raw in uploads:
            safe = Path(name).name
            if Path(safe).suffix.lower() in ('.csv', '.xlsx', '.txt'):
                (folder / ('upload-' + hashlib.sha256(raw).hexdigest()[:12] + '-' + safe)).write_bytes(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            rows = scoreboard.load(folder)
    return scoreboard.scoreboard(rows)


def _table(groups, minimum):
    records = []
    for name, m in groups.items():
        records.append({
            'Group': name, 'Graded': m.get('n', 0), 'Win rate': m.get('win_rate'),
            'Model log loss': m.get('model_log_loss'), 'Market log loss': m.get('market_log_loss'),
            'Skill vs market': m.get('skill_vs_market'),
            'Skill 95% CI': str(m.get('skill_vs_market_ci', '—')),
            'ROI / unit': m.get('roi_per_unit'), 'ROI 95% CI': str(m.get('roi_ci', '—')),
            'Verdict': scoreboard.verdict(m, minimum),
        })
    return pd.DataFrame(records)


def render_scoreboard(*, baseline: Path = BASELINE, minimum: int = 150):
    with st.expander('Pick quality vs market (scoreboard)', expanded=False):
        st.caption('Read-only. "Skill vs market" is market log loss minus model log loss: above 0 means the model '
                   'beat the de-vigged market. Trust a verdict only when its 95% interval excludes 0, with at least '
                   f'{minimum} graded picks. A coin flip has log loss 0.693. CLV needs closing prices (not yet captured).')
        files = st.file_uploader('Add graded exports (Drive downloads)', type=['csv', 'xlsx', 'txt'],
                                 accept_multiple_files=True, key='scoreboard_uploads')
        # Lazy: parsing every export takes seconds, so nothing runs until requested.
        if st.button('Compute scoreboard', key='scoreboard_compute'):
            st.session_state['scoreboard_requested'] = True
        if not st.session_state.get('scoreboard_requested'):
            return
        uploads = tuple(sorted((f.name, f.getvalue()) for f in (files or [])))
        with st.spinner('Scoring graded picks...'):
            board = _board(uploads, str(baseline))
        overall, coverage = board['overall'], board['coverage']
        if not overall.get('n'):
            st.info('No graded picks found.')
            return
        st.markdown(f"**{scoreboard.verdict(overall, minimum)}** — {overall['n']} graded picks, "
                    f"{coverage['first_date']} to {coverage['last_date']}.")
        cols = st.columns(4)
        cols[0].metric('Win rate', f"{overall['win_rate']:.1%}")
        cols[1].metric('Model log loss', f"{overall['model_log_loss']:.3f}")
        cols[2].metric('Market log loss', f"{overall.get('market_log_loss', float('nan')):.3f}")
        cols[3].metric('ROI / unit', f"{overall.get('roi_per_unit', float('nan')):+.1%}")
        st.markdown('**By league and market**')
        st.dataframe(_table(board['by_league_market'], minimum), hide_index=True, use_container_width=True)
        st.markdown('**By month**')
        st.dataframe(_table(board['by_month'], minimum), hide_index=True, use_container_width=True)
        st.markdown('**Calibration (overall)**')
        st.dataframe(pd.DataFrame(overall.get('calibration', [])), hide_index=True, use_container_width=True)
