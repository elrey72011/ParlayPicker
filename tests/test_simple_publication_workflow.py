from copy import deepcopy
from types import SimpleNamespace
from app.ui import sftp_publish, public_results
from test_sftp_publishing import package, config
from test_public_history import pub, scores, report, History, Memory


def test_uncertain_upload_blocks_new_preview_without_retry(monkeypatch):
    calls=[]
    monkeypatch.setattr(sftp_publish.remote,'site_info',lambda c:None)
    def fail(*args):
        calls.append('upload')
        raise RuntimeError('lost response')
    monkeypatch.setattr(sftp_publish.remote,'deploy',fail)
    store=SimpleNamespace(archive=lambda p:'hash',submitted=lambda *a:None)
    jobs={}
    first=sftp_publish.publish_once(package(),config(),store,jobs)
    changed=deepcopy(package());changed['built_at']='2026-09-10T12:00:00+00:00'
    second=sftp_publish.publish_once(changed,config(),store,jobs)
    assert second is first
    assert calls==['upload']
    assert not first.get('history_saved')


def test_mismatched_live_page_not_confirmed(monkeypatch):
    confirmed=[]
    monkeypatch.setattr(sftp_publish.remote,'site_info',lambda c:None)
    monkeypatch.setattr(sftp_publish.remote,'deploy',lambda *a:{'state':'uploaded'})
    monkeypatch.setattr(sftp_publish.remote,'deployment_status',lambda *a:{'state':'content_mismatch'})
    store=SimpleNamespace(archive=lambda p:'hash',submitted=lambda *a:None,confirm=lambda *a:confirmed.append(a))
    job=sftp_publish.publish_once(package(),config(),store,{})
    assert not confirmed and not job.get('history_saved')


def test_update_results_persists_before_reporting(monkeypatch):
    monkeypatch.setattr("app_core.public_record.START_DATE","2026-09-09")
    store=History('site-1234','folder',Memory())
    saved={'publications':[pub()],'revisions':[],'imports':[],'locks':[],'rows':report([pub()],[])}
    monkeypatch.setattr(public_results,'history',lambda setting:store)
    monkeypatch.setattr(public_results,'fetch_scores',lambda *a:{'recorded_at':'2026-09-10T00:00:00Z','scores':scores()})
    public_results.update_pending_results(lambda key:'',saved)
    assert len(store.all('scores'))==1
    assert all(r['outcome']!='PENDING' for r in saved['rows'])


def test_history_attempted_once_and_preview_built_without_click(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_publish_panel import app
    calls=[]
    def restore(setting):
        import streamlit as st
        calls.append('restore')
        st.session_state['public_results_']={'publications':[],'revisions':[],'imports':[],'locks':[],'rows':[]}
        return True
    monkeypatch.setattr(public_results,'restore_history',restore)
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','test-only-publish-token')
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','')
    at=AppTest.from_function(app).run()
    at.text_input[0].set_value('test-only-publish-token').run()
    assert not at.exception
    assert 'publication_preview' in at.session_state
    at.run()
    assert calls==['restore']


def test_update_button_publishes_once_not_on_navigation(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_publish_panel import app
    calls=[]
    def restore(setting):
        import streamlit as st
        st.session_state['public_results_']={'publications':[],'revisions':[],'imports':[],'locks':[],'rows':[]}
        return True
    monkeypatch.setattr(public_results,'restore_history',restore)
    monkeypatch.setattr(public_results,'update_pending_results',lambda *a:calls.append('grade'))
    monkeypatch.setattr(sftp_publish,'publish_action',lambda *a:calls.append('publish') or 'Published')
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','test-only-publish-token')
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','')
    at=AppTest.from_function(app).run()
    at.text_input[0].set_value('test-only-publish-token').run()
    assert not calls
    at.button(key='public_update_all').click().run()
    assert not at.exception and calls==['grade','publish']
    at.run()
    assert calls==['grade','publish']


def test_empty_analysis_app_branch_grades_and_publishes_saved_board(monkeypatch):
    """Execute the actual app's early-return branch with the real publishing UI."""
    import ast
    import textwrap
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    tree=ast.parse(Path('streamlit_app.py').read_text(encoding='utf-8'))
    branch=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
                and ast.unparse(node.test)=='analysis_df is None or analysis_df.empty'
                and 'saved_props' in ast.unparse(node))
    source="""
import pandas as pd
import streamlit as st

def main():
    analysis_df = None
    publication_games = pd.DataFrame()
    publication_props = pd.DataFrame()
    publication_dfs = {}
    diagnostics = {}
    tab3, publish_tab = st.tabs(['Full Pick Board', 'Preview & Publish'])
"""+textwrap.indent(ast.unparse(branch),'    ')+"\nmain()\n"
    old=pub()
    expected=deepcopy(old['package'])
    calls=[]
    def restore(setting):
        import streamlit as st
        st.session_state['public_results_']={'publications':[deepcopy(old)],'revisions':[],
            'imports':[],'locks':[],'rows':report([old],[])}
        return True
    def grade(setting,saved):
        calls.append('grade')
        saved['rows']=report(saved['publications'],[{'recorded_at':'2026-09-10T00:00:00Z','scores':scores()}])
    def publish(package,setting):
        calls.append(deepcopy(package))
        return 'Published'
    monkeypatch.setattr(public_results,'restore_history',restore)
    monkeypatch.setattr(public_results,'update_pending_results',grade)
    monkeypatch.setattr(sftp_publish,'publish_action',publish)
    monkeypatch.setattr('app_core.public_record.START_DATE','2026-09-09')
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','test-only-publish-token')
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','')
    at=AppTest.from_string(source).run()
    assert not at.exception
    assert at.text_input(key='publication_token')
    at.text_input(key='publication_token').set_value('test-only-publish-token').run()
    assert not at.exception and not calls
    assert not at.button(key='public_update_all').disabled
    assert not any(b.key=='public_history_grade' for b in at.button)
    at.checkbox(key='public_history_tools').check().run()
    assert at.button(key='public_history_grade')
    at.button(key='public_update_all').click().run()
    assert not at.exception and len(calls)==2 and calls[0]=='grade'
    sent=calls[1]
    assert sent['games']==expected['games']
    assert sent['built_at']==expected['built_at']
    assert sent['props']==expected['props'] and sent['dfs']==expected['dfs']
    assert sent['results'] and all(row['outcome']!='PENDING' for row in sent['results'])
    assert old['package']==expected
    at.run()
    assert len(calls)==2
