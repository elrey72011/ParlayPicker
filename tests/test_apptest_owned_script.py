"""Real AppTest lifecycle controls; no application or production deadline changes."""
from pathlib import Path
import tempfile

import pytest

from tests.apptest_owned_script import app_from_owned_script


def test_from_string_requires_its_module_owned_directory(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest, app_test

    original = app_test.TMP_DIR
    controlled = tempfile.TemporaryDirectory(dir=tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(app_test, "TMP_DIR", controlled)
        app = AppTest.from_string("import streamlit as st\nst.write('before cleanup')")
        assert Path(app._script_path).is_file()
        controlled.cleanup()
        with pytest.raises(FileNotFoundError):
            AppTest.from_string("import streamlit as st\nst.write('after cleanup')")
    assert app_test.TMP_DIR is original


def test_owned_script_survives_unrelated_cleanup_and_widget_reruns(tmp_path, monkeypatch):
    from streamlit.testing.v1 import app_test

    controlled = tempfile.TemporaryDirectory(dir=tmp_path)
    original = app_test.TMP_DIR
    script = """import streamlit as st
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if st.button('Increment', key='increment'):
    st.session_state['count'] += 1
st.write(st.session_state['count'])
"""
    with monkeypatch.context() as patch:
        patch.setattr(app_test, "TMP_DIR", controlled)
        app = app_from_owned_script(tmp_path, script)
        assert app.default_timeout == 3
        controlled.cleanup()
        app.run()
        assert not app.exception and app.session_state['count'] == 0
        app.button(key='increment').click().run()
        assert not app.exception and app.session_state['count'] == 1
        app.run()
        assert not app.exception and app.session_state['count'] == 1
        app.button(key='increment').click().run()
        assert not app.exception and app.session_state['count'] == 2
        assert (tmp_path / 'app.py').is_file()
    assert app_test.TMP_DIR is original


@pytest.mark.parametrize('failure', [False, True])
def test_fixture_owner_cleans_script_after_success_or_failure(tmp_path, failure):
    class ControlledFailure(Exception):
        pass

    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    def exercise():
        with tempfile.TemporaryDirectory(dir=tmp_path) as owner:
            owned = Path(owner)
            app = app_from_owned_script(owned, "import streamlit as st\nst.write('owned')")
            app.run()
            assert not app.exception and (owned / 'app.py').is_file()
            if failure:
                raise ControlledFailure(owned)
            return owned
    if failure:
        with pytest.raises(ControlledFailure) as caught:
            exercise()
        owned = caught.value.args[0]
    else:
        owned = exercise()
    assert not owned.exists() and unrelated.is_dir()


def test_two_test_owned_scripts_and_states_are_isolated(tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    a = app_from_owned_script(first, "import streamlit as st\nst.write('first')")
    b = app_from_owned_script(second, "import streamlit as st\nst.write('second')")
    a.run()
    b.run()
    a.run()
    assert not a.exception and not b.exception
    assert a.markdown[0].value == 'first' and b.markdown[0].value == 'second'
    assert Path(a._script_path) != Path(b._script_path)
    assert (first / 'app.py').read_text(encoding='utf-8') != (second / 'app.py').read_text(encoding='utf-8')
