from streamlit.testing.v1 import AppTest


def _navigation_app():
    import streamlit as st
    from streamlit_app import _render_main_tabs

    if st.session_state.get("show_notice"):
        st.info("Results updated")
    tabs = _render_main_tabs()
    with tabs[2]:
        st.checkbox("Enable Sweet Spot Filter", key="show_notice")
        st.number_input("Min EV", value=0.04, key="test_min_ev")
    st.caption(st.session_state["main_navigation"])


def test_best_picks_selection_survives_filter_reruns_and_layout_changes():
    app = AppTest.from_function(_navigation_app).run()
    assert not app.exception
    assert app.session_state["main_navigation"] == "Odds"
    def browser_rerun():
        # AppTest 1.59 does not expose tab selection or serialize tab state.
        # Include the tab's string value just as the browser does on each rerun.
        states = app._tree.get_widget_states()
        tab_id = app.get("tab_container")[0].proto.tab_container.id
        assert tab_id
        states.widgets.add(id=tab_id, string_value="Best Picks")
        app._run(states)

    browser_rerun()
    app.checkbox[0].check()
    browser_rerun()
    assert not app.exception
    assert app.session_state["main_navigation"] == "Best Picks"
    app.number_input[0].set_value(0.06)
    browser_rerun()
    assert not app.exception
    assert app.caption[-1].value == "Best Picks"
    assert app.get("tab_container")[0].proto.tab_container.default_tab_index == 2
    assert app.session_state["test_min_ev"] == 0.06
