from datetime import date, time
from streamlit.testing.v1 import AppTest
from app.ui.publish_panel import dfs_lock_timestamp


def picker_app():
    import streamlit as st
    from app.ui.publish_panel import render_dfs_lock_picker
    st.session_state['selected_lock']=render_dfs_lock_picker()


def test_picker_produces_timezone_aware_lock():
    at=AppTest.from_function(picker_app).run()
    at.date_input(key='dfs_lock_date').set_value(date(2026,9,13))
    at.time_input(key='dfs_lock_clock').set_value(time(13,0)).run()
    assert not at.exception
    assert at.session_state['selected_lock']=='2026-09-13T13:00:00-04:00'
    assert '01:00 PM' in at.caption[0].value
    at.time_input(key='dfs_lock_clock').set_value(time(16,25)).run()
    assert at.session_state['selected_lock']=='2026-09-13T16:25:00-04:00'


def test_eastern_offset_changes_with_season():
    assert dfs_lock_timestamp(date(2026,9,13),time(13))=='2026-09-13T13:00:00-04:00'
    assert dfs_lock_timestamp(date(2026,12,13),time(13))=='2026-12-13T13:00:00-05:00'
