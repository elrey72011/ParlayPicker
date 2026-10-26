"""Explicit PRIVATE SYNTHETIC browser fixture entrypoint. Never deploy this file.

streamlit run tests/nfl_private_workflow_browser.py -- /work/SYNTHETIC-db.sqlite3 ISO_ANCHOR
No production settings, credentials, transport or authentic results are used.
"""
from pathlib import Path
from datetime import datetime
import ipaddress
import json
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import streamlit as st
from app.ui.nfl_private_workflow import render
from app.ui.nfl_workflow_state import revoke
from app_core.nfl_workflow_store import Store
from app_core.nfl_private_workflow import Workflow
from nfl_workflow_fixtures import SyntheticProvider,make_workflow,TOKEN,KEY

assert len(sys.argv) in (3,4)
path=Path(sys.argv[1]).resolve();assert path.name.startswith('SYNTHETIC-')
guard_path=path.with_suffix('.guard.json')


def bump(key):
    data=json.loads(guard_path.read_text()) if guard_path.exists() else {}
    data[key]=data.get(key,0)+1
    guard_path.write_text(json.dumps(data,sort_keys=True),encoding='utf-8')


if not getattr(socket,'_nfl_synthetic_browser_guard',False):
    original_connect=socket.socket.connect
    original_resolve=socket.getaddrinfo
    def local(host):
        if host=='localhost':return True
        try:return ipaddress.ip_address(host).is_loopback
        except (ValueError,TypeError):return False
    def connect(self,address):
        if isinstance(address,tuple) and not local(address[0]):
            bump('external_socket_attempts');raise RuntimeError('SYNTHETIC_BROWSER_EXTERNAL_NETWORK_FORBIDDEN')
        return original_connect(self,address)
    def resolve(host,*args,**kwargs):
        if not local(host):
            bump('external_socket_attempts');raise RuntimeError('SYNTHETIC_BROWSER_EXTERNAL_DNS_FORBIDDEN')
        return original_resolve(host,*args,**kwargs)
    socket.socket.connect=connect;socket.getaddrinfo=resolve
    socket._nfl_synthetic_browser_guard=True

st.set_page_config(page_title='PRIVATE SYNTHETIC NFL workflow test',layout='wide')
st.title('PRIVATE SYNTHETIC NFL workflow test')
st.warning('Private local fixture. No authentic provider traffic, game predictions, owner findings or operational authority.')
readonly=st.checkbox('SYNTHETIC guard: saved viewing only',key='SYNTHETIC_readonly')


def profile(frame,event,arg):
    if event=='call' and frame.f_code.co_name=='predict_market_probabilities' and frame.f_code.co_filename.replace('\\','/').endswith('/market_probability_model.py'):
        if readonly:
            bump('prediction_attempts_in_read_only');raise RuntimeError('STATIC_VIEW_PREDICTOR_FORBIDDEN')
        bump('synthetic_native_calls')
sys.setprofile(profile)

supplied=st.text_input('Owner token',type='password',key='publication_token')


def setting(name,default=''):
    return TOKEN if name=='PARLAYPICKER_PUBLISH_TOKEN' else default


if supplied!=TOKEN:
    revoke(st.session_state)
    st.info('Owner access required. No database is opened.')
else:
    def factory(access):
        if 'SYNTHETIC_provider' not in st.session_state:
            st.session_state['SYNTHETIC_provider']=SyntheticProvider(anchor=datetime.fromisoformat(sys.argv[2]))
            st.session_state['SYNTHETIC_provider'].line=float(sys.argv[3]) if len(sys.argv)==4 else 2.5
        provider=st.session_state['SYNTHETIC_provider']
        service,_=make_workflow(path,provider=provider)
        original_factory=service.store._factory
        def connect_store(is_readonly):
            if readonly and not is_readonly:
                bump('db_write_attempts_in_read_only');raise RuntimeError('STATIC_VIEW_DB_WRITE_FORBIDDEN')
            return original_factory(is_readonly)
        service.store._factory=connect_store
        def source_transport(*args,**kwargs):
            if readonly:
                bump('provider_attempts_in_read_only');raise RuntimeError('STATIC_VIEW_PROVIDER_FORBIDDEN')
            bump('synthetic_score_requests');return provider.scores(*args,**kwargs)
        class Market:
            def __call__(self,*args,**kwargs):
                if readonly:
                    bump('provider_attempts_in_read_only');raise RuntimeError('STATIC_VIEW_PROVIDER_FORBIDDEN')
                bump('synthetic_market_requests');return provider(*args,**kwargs)
            def validate_body(self,raw):provider.validate_body(raw)
        return Workflow(service.store,access,allow_acquisition=True,allow_inference=True,
            score_transport=source_transport,market_transport=Market())
    render(setting,service_factory=factory)

with st.expander('SYNTHETIC verification counters'):
    st.json(json.loads(guard_path.read_text()) if guard_path.exists() else {})
