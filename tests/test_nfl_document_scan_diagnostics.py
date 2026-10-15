"""SYNTHETIC only: real CLI/HTTP parser, synthetic network/security boundaries."""
from copy import deepcopy
import gzip
import hashlib
import json
import pytest
from app_core import nfl_document_acquisition as lane, nfl_document_scan_diagnostics as scan
from app_core.nfl_html_document_scan import checked, checked_with_diagnostics
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_document_html_anchors import actual_cli, ANCHOR
from test_nfl_document_acquisition import guide_fixture, security, NOW


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


@pytest.mark.parametrize('markup,rule',[
    ('apiKey=SYNTHETIC_SECRET_CANARY','API_KEY_ASSIGNMENT'),
    ('access_token: SYNTHETIC_SECRET_CANARY','ACCESS_TOKEN_ASSIGNMENT'),
    ('authorization: Bearer SYNTHETIC_SECRET_CANARY','AUTHORIZATION_ASSIGNMENT'),
    ('client_secret="SYNTHETIC_SECRET_CANARY"','CLIENT_SECRET_ASSIGNMENT'),
    ('password=SYNTHETIC_SECRET_CANARY','PASSWORD_ASSIGNMENT'),
    ('apiKey&#61;SYNTHETIC_SECRET_CANARY','API_KEY_ASSIGNMENT'),
    ('%61piKey%3DSYNTHETIC_SECRET_CANARY','API_KEY_ASSIGNMENT'),
    (r'\u0061piKey=SYNTHETIC_SECRET_CANARY','API_KEY_ASSIGNMENT'),
    ('apiKey=YOUR_API_KEY_SYNTHETIC_SECRET_CANARY','API_KEY_ASSIGNMENT'),
    ('apiKey={apiKey}SYNTHETIC_SECRET_CANARY','API_KEY_ASSIGNMENT'),
    ('<input name="api-key" value="SYNTHETIC_SECRET_CANARY">','API_KEY_ASSIGNMENT'),
    ('<section data-key="api-key">SYNTHETIC ambiguous label</section>','API_KEY_ASSIGNMENT'),
])
def test_failure_facts_actual_cli_transport_journal_bundle_readback(monkeypatch,tmp_path,capsys,markup,rule):
    raw=ANCHOR[:-7]+b' apiKey=YOUR_API_KEY '+markup.encode()+b'</html>'
    code,report,bundle,plan,sent=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V4)
    assert code==1 and report['reason']=='DOCUMENT_CREDENTIAL_BODY'
    expected=dict(version=scan.VERSION,scanner_identity=scan.SCANNER,stage='HTML_ASSIGNMENT',
        matched_rule=rule,structural_syntax_applied=True,public_placeholder_applied=True)
    assert report['scanner_diagnostics']==bundle['payload']['scanner_diagnostics']==expected
    assert report['diagnostics']['phase']=='BODY_VALIDATION' and report['diagnostics']['http_status']==200
    assert not bundle['payload']['objects'] and len(sent)==1 and report['attempts']==1
    retained=json.dumps(bundle)+json.dumps(report)
    assert 'SYNTHETIC_SECRET_CANARY' not in retained and markup not in retained
    assert 'matched_text' not in retained and 'excerpt' not in retained
    journal=tmp_path/(plan['sha256']+'.documents-attempts.jsonl');original=journal.read_bytes()
    p,a,c=guide_fixture(tmp_path,version=lane.VERSION_V4)
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security):
        with pytest.raises(ValueError,match='DOCUMENT_SPENT_NO_RESUME'):
            lane.acquire(p,a,root=tmp_path,wall=lambda:NOW,transport=lambda *a:pytest.fail('RETRY'))
    assert journal.read_bytes()==original and len(sent)==1


@pytest.mark.parametrize('compressed',[False,True])
@pytest.mark.parametrize('markup',[
    '<section id="api-key"><a href="#api-key">SYNTHETIC heading</a></section>',
    'apiKey=YOUR_API_KEY', 'apiKey={apiKey}',
])
def test_syntax_and_exact_placeholders_remain_accepted_without_byte_changes(monkeypatch,tmp_path,capsys,compressed,markup):
    raw=('<html>SYNTHETIC '+markup+'</html>').encode()
    code,report,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V4,
        wire=gzip.compress(raw) if compressed else None)
    assert code==0 and report['scanner_diagnostics'] is None
    assert bundle['payload']['version']==lane.BUNDLE_V4
    assert lane.read_bundle(bundle)['clock'][0]==raw and report['bodies'][0]['body_sha256']==hashlib.sha256(raw).hexdigest()
    from app_core import nfl_html_guide_custody
    assert lane.project(bundle,'clock')['version']==nfl_html_guide_custody.VERSION


def test_old_contract_has_no_diagnostic_extension_and_same_rejection(monkeypatch,tmp_path,capsys):
    raw=b'<html>SYNTHETIC apiKey=SYNTHETIC_SECRET_CANARY</html>'
    code,report,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V3)
    assert code==1 and report['reason']=='DOCUMENT_CREDENTIAL_BODY'
    assert 'scanner_diagnostics' not in report and 'scanner_diagnostics' not in bundle['payload']
    with pytest.raises(ValueError,match='DOCUMENT_CREDENTIAL_BODY'):checked(raw,public_guide_examples=True)


@pytest.mark.parametrize('case',['truncated','oversize','expansion','read_timeout','format','deep_encoding'])
def test_unchanged_bounds_and_unknown_facts(monkeypatch,tmp_path,capsys,case):
    raw=ANCHOR;wire=None;declared=None
    expected={'truncated':'DOCUMENT_INCOMPLETE','oversize':'DOCUMENT_SIZE','expansion':'DOCUMENT_SIZE',
        'read_timeout':'DOCUMENT_TRANSPORT_FAILURE','format':'DOCUMENT_FORMAT','deep_encoding':'DOCUMENT_FORMAT'}[case]
    if case=='truncated':declared=len(raw)+1
    elif case in {'oversize','expansion'}:
        raw+=b' '*(512*1024+1-len(raw))
        if case=='expansion':wire=gzip.compress(raw)
    elif case=='read_timeout':
        def fail(self):raise TimeoutError('SYNTHETIC_SECRET_CANARY');yield b''
        monkeypatch.setattr(lane.Response,'chunks',fail)
    elif case=='format':raw=b'SYNTHETIC not HTML'
    else:raw=b'<html>%252561piKey%25253DSYNTHETIC_SECRET_CANARY</html>'
    code,report,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,raw,version=lane.VERSION_V4,wire=wire,declared=declared)
    assert code==1 and report['reason']==expected and not bundle['payload']['objects']
    facts=report['scanner_diagnostics']
    if case in {'format','deep_encoding'}:
        assert facts['matched_rule']=='UNKNOWN' and facts['structural_syntax_applied']=='UNKNOWN'
    else:assert facts is None
    assert 'SYNTHETIC_SECRET_CANARY' not in json.dumps(bundle)+json.dumps(report)


@pytest.mark.parametrize('key,value',[
    ('matched_rule','SYNTHETIC_SECRET_CANARY'),('stage','SYNTHETIC_SECRET_CANARY'),
    ('scanner_identity','SYNTHETIC_SECRET_CANARY'),('version','SYNTHETIC_SECRET_CANARY'),
    ('structural_syntax_applied','SYNTHETIC_SECRET_CANARY'),('public_placeholder_applied',1),
    ('matched_text','SYNTHETIC_SECRET_CANARY'),
])
def test_allowlist_rejects_diagnostic_injection(key,value):
    facts=scan.initial();facts[key]=value
    with pytest.raises(ValueError,match='^DOCUMENT_SCAN_DIAGNOSTICS_INVALID$'):scan.checked(facts)


def test_rehashed_bad_diagnostics_reject_on_readback_and_report(monkeypatch,tmp_path,capsys):
    _,_,bundle,_,_=actual_cli(monkeypatch,tmp_path,capsys,b'<html>apiKey=SYNTHETIC_SECRET_CANARY</html>',version=lane.VERSION_V4)
    p=deepcopy(bundle['payload']);p['scanner_diagnostics']['matched_rule']='SYNTHETIC_SECRET_CANARY'
    for reader in (lane.read_bundle,lane.report):
        with pytest.raises(ValueError,match='DOCUMENT_SCAN_DIAGNOSTICS_INVALID'):reader(lane.seal(p))
    p=deepcopy(bundle['payload']);p['version']=lane.BUNDLE_V3
    with pytest.raises(ValueError,match='DOCUMENT_READBACK'):lane.read_bundle(lane.seal(p))


def test_previous_authorization_cannot_select_new_diagnostics(tmp_path):
    p,a,c=guide_fixture(tmp_path,version=lane.VERSION_V4)
    old,old_auth,_=guide_fixture(tmp_path,version=lane.VERSION_V3)
    assert p['sha256']!=old['sha256']
    with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security):
        with pytest.raises(ValueError,match='DOCUMENT_AUTHORIZATION'):
            lane.acquire(p,old_auth,root=tmp_path,wall=lambda:NOW,transport=lambda *a:pytest.fail('OLD APPROVAL'))
    assert not list(tmp_path.iterdir())
