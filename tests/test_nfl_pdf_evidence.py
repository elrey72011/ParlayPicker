"""SYNTHETIC documents/artifacts only; authentic PDFs receive static inspection."""
import base64
from copy import deepcopy
import hashlib
import io
import json
import pytest
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_source_interpretation import interpreted_fixture
from test_nfl_scoring_profile import sign_new
from test_source_contract_pipeline import INFERENCE

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def make_pdf(text, *, pages=1, active=False, compressed=False):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
    writer=PdfWriter()
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica'),NameObject('/Encoding'):NameObject('/WinAnsiEncoding')})
    for _ in range(pages):
        page=writer.add_blank_page(612,792);commands=['BT /F1 10 Tf 22 760 Td 12 TL']
        for line in text.splitlines():
            escaped=line.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
            commands.append('('+escaped+') Tj T*')
        commands.append('ET');stream=DecodedStreamObject();stream.set_data('\n'.join(commands).encode('latin-1'))
        page[NameObject('/Contents')]=writer._add_object(stream.flate_encode() if compressed else stream)
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
    if active:writer.add_js('SYNTHETIC_ONLY; never execute this content')
    out=io.BytesIO();writer.write(out);return out.getvalue()


def fixture(monkeypatch,away=False,reference_away=False):
    from app_core import nfl_pdf_evidence as pdf,nfl_pdf_research as reader,nfl_pdf_interpretation as interpretation
    packet,baseline=interpreted_fixture(monkeypatch,away,reference_away);p=packet['payload']
    from app_core.nfl_observed_custody import read
    _,text=read(p['objects']['rules']);receipt=deepcopy(p['objects']['rules']['receipt']);receipt['format']='pdf'
    raw=make_pdf(text);p['objects']['rules']=pdf.retained_pdf(raw,receipt=receipt,prepared_at='2026-10-06T19:25:02.200Z')
    p['version']=reader.VERSION
    p['projections']['schedule']['payload']['facts']['provider_namespace']=dict(method='receipt_namespace',status='OWNER_INTERPRETED',supports=[dict(object='odds',path=None,text=None,operation='receipt_source_namespace')])
    page=p['objects']['rules']['extraction']['pages'][0]
    for fact in p['projections']['rules']['payload']['facts'].values():
        for support in fact['supports']:
            exact=support['text']['exact'];assert exact in page['text'],(exact,page['text'])
            start=page['text'].index(exact)
            support.update(path=None,text=None,operation='pdf_page_text',pdf=dict(page=0,start=start,end=start+len(exact),exact=exact,page_sha256=page['text_sha256'],extraction_sha256=pdf.digest(p['objects']['rules']['extraction'])))
    resign(p)
    return interpretation.seal(p),baseline


def resign(p):
    from app_core import nfl_pdf_evidence as pdf,nfl_pdf_research as reader,nfl_pdf_interpretation as interpretation
    identities={k:dict(body_sha256=o['body_sha256'],receipt_sha256=pdf.digest(o['receipt'])) for k,o in p['objects'].items()}
    for k,o in p['objects'].items():
        if o.get('version')==pdf.VERSION:identities[k]['extraction_sha256']=pdf.digest(o['extraction'])
    for name in ('schedule','listing','rules','clock_document'):
        v=p['projections'][name]['payload'];v['version']=interpretation.VERSION;v['method_version']=interpretation.METHOD;v['original_objects']=deepcopy(identities)
        v['review']['version']=interpretation.REVIEW;v['review']['subject_sha256']=interpretation.subject(v)
        p['projections'][name]=interpretation.seal(v)
    sign_new(p);p['owner_review']['version']='nfl-owner-pdf-verification-v5';p['owner_review']['subject_sha256']=reader.subject(p)


def test_original_v4_and_custody_reject_pdf(monkeypatch):
    from app_core import nfl_interpreted_research as old,nfl_observed_custody as bodies,nfl_pdf_interpretation as interpretation
    packet,_=fixture(monkeypatch);p=packet['payload'];p['version']=old.VERSION
    with pytest.raises(ValueError,match='NFL_PRIVATE_CUSTODY_MISSING'):old.checked(interpretation.seal(p),INFERENCE)
    obj=deepcopy(p['objects']['rules']);obj['version']=bodies.VERSION;obj.pop('extraction')
    with pytest.raises(ValueError,match='NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN'):bodies.read(obj)
    obj['receipt']['format']='text'
    with pytest.raises(ValueError,match='NFL_PRIVATE_RESPONSE_SCHEMA'):bodies.read(obj)


@pytest.mark.parametrize('away,reference_away',[(False,False),(False,True),(True,False),(True,True)])
def test_pdf_actual_caller_capture_export_replay_and_browser(monkeypatch,tmp_path,away,reference_away):
    from app_core import nfl_pdf_research as reader,nfl_private_routes as routes,research_replay,nfl_pdf_evidence as pdf
    from app_core.per_game_boards import per_game_board
    from test_nfl_owner_research import actual_pipeline,private_app
    from test_source_contract_pipeline import exported
    from streamlit.testing.v1 import AppTest
    packet,baseline=fixture(monkeypatch,away,reference_away);original=deepcopy(packet)
    features,_=reader.checked(packet,INFERENCE);assert features==packet['payload']['features']
    with routes.selected([packet]):analysis=actual_pipeline(monkeypatch)
    i=int(away);row=analysis.iloc[i];assert row.ml_probability==pytest.approx(baseline.iloc[i].ml_probability,abs=1e-12)
    result=exported(monkeypatch,tmp_path,analysis)
    receipt=research_replay.retain_export(result['frames'],result['package'],result['card'],result['captured'],path=result['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    display=routes.private_display(retained.iloc[i]);assert display['raw_probability']==row.ml_probability and display['calibrated'] is False
    metadata=json.loads(retained.iloc[i].ml_estimate_metadata)['nfl_private_inputs']['payload']
    assert metadata['original_packet']==original
    obj=metadata['original_packet']['payload']['objects']['rules'];raw,_=pdf.read_pdf(obj)
    assert hashlib.sha256(raw).hexdigest()==obj['body_sha256']
    for family in ('overall','sides','totals'):
        assert len(per_game_board(result['card'],result['captured'],family=family,novig_only=True))==1
        assert all(r['status']=='PASS' and r['win_estimate'] is None and r['ev'] is None for r in saved['package']['games'][family])
    assert 'bytes_base64' not in json.dumps(saved['package']) and 'layout_text' not in json.dumps(saved['package'])
    assert all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    app=AppTest.from_function(private_app);app.session_state['private_candidates']=retained;app.run()
    assert not app.exception and any(m.value==f'{row.ml_probability:.2%}' for m in app.metric)
    assert any('Original PDF bytes' in c.value for c in app.caption)
    assert original==packet


@pytest.mark.parametrize('attack',['body','extraction','locator','output','review','future_extraction','stale','started','edition','applicability','price','line','orientation','cross_mode'])
def test_pdf_rejects_before_numerical_calls(monkeypatch,attack):
    from app_core import nfl_pdf_research as reader,nfl_pdf_interpretation as interpretation
    packet,baseline=fixture(monkeypatch);p=packet['payload']
    if attack=='body':p['objects']['rules']['bytes_base64']=base64.b64encode(b'SYNTHETIC ALTERED BODY').decode()
    elif attack=='extraction':p['objects']['rules']['extraction']['pages'][0]['text']+='SYNTHETIC ALTERATION'
    elif attack=='locator':p['projections']['rules']['payload']['facts']['period']['supports'][0]['pdf']['start']+=1
    elif attack=='output':p['projections']['rules']['payload']['values']['period']='first_half'
    elif attack=='review':p['projections']['rules']['payload']['review']['subject_sha256']='0'*64
    elif attack=='future_extraction':p['objects']['rules']['extraction']['prepared_at']='2026-10-06T19:25:05Z'
    elif attack=='stale':p['quote']['recorded_at']='2026-10-06T18:00:00Z'
    elif attack=='started':p['event']['start']='2026-10-06T18:00:00Z'
    elif attack=='edition':p['projections']['listing']['payload']['values']['rule_edition']='UNREVIEWED'
    elif attack=='applicability':p['projections']['listing']['payload']['values']['applicability']['listing_id']='OTHER'
    elif attack=='price':p['quote']['price']+=10
    elif attack=='line':p['quote']['point']+=1
    elif attack=='orientation':p['event']['home'],p['event']['away']=p['event']['away'],p['event']['home']
    else:p['owner_review']['version']='nfl-owner-interpreted-verification-v4'
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda *a,**k:pytest.fail('SYNTHETIC REJECTED FACTS REACHED INFERENCE'))
    try:
        with reader.selected([interpretation.seal(p)]):result=reader.predict(baseline.iloc[0])
    except ValueError as exc:assert str(exc).startswith('NFL_PRIVATE_')
    else:
        assert result['ml_inference_status']=='unavailable'
        assert result['ml_unavailable_reason'].startswith('NFL_PRIVATE_')


@pytest.mark.parametrize('attack',['locator','extraction','future_extraction','output','edition','price'])
def test_rehashed_pdf_attacks_do_not_supply_facts(monkeypatch,attack):
    from app_core import nfl_pdf_research as reader,nfl_pdf_interpretation as interpretation
    packet,_=fixture(monkeypatch);p=packet['payload']
    if attack=='locator':p['projections']['rules']['payload']['facts']['period']['supports'][0]['pdf']['exact']='first half'
    elif attack=='extraction':p['objects']['rules']['extraction']['pages'][0]['text']='SYNTHETIC forged page'
    elif attack=='future_extraction':p['objects']['rules']['extraction']['prepared_at']='2026-10-06T19:25:05Z'
    elif attack=='output':p['projections']['rules']['payload']['values']['period']='first_half'
    elif attack=='edition':p['projections']['rules']['payload']['values']['rule_edition']='NFL-001 edition 2026-10-02'
    else:p['quote']['price']=120
    resign(p)
    with pytest.raises(ValueError):reader.checked(interpretation.seal(p),INFERENCE)


@pytest.mark.parametrize('attack',['truncated','active','unreadable','oversize','pages','canary'])
def test_inert_parser_bounds_and_credential_exclusion(attack):
    from app_core import nfl_pdf_evidence as pdf
    raw=make_pdf('SYNTHETIC readable complete document.')
    if attack=='truncated':raw=raw[:-9]
    elif attack=='active':raw=make_pdf('SYNTHETIC only.',active=True)
    elif attack=='unreadable':raw=make_pdf('')
    elif attack=='oversize':raw=b'X'*(pdf.MAX_BODY+1)
    elif attack=='pages':raw=make_pdf('SYNTHETIC only.',pages=pdf.MAX_PAGES+1)
    else:raw=make_pdf('SYNTHETIC api_key="DO_NOT_RETAIN_CANARY"')
    with pytest.raises(ValueError):pdf.extract(raw)


def test_package_method_and_pure_cache_cannot_supply_review(monkeypatch):
    from app_core import nfl_pdf_evidence as pdf,nfl_pdf_research as reader,nfl_pdf_interpretation as interpretation
    packet,_=fixture(monkeypatch);p=packet['payload'];obj=p['objects']['rules']
    raw=base64.b64decode(obj['bytes_base64']);assert pdf.extract(raw)==pdf.extract(raw)
    assert pdf._pages.cache_info().maxsize==2
    p['owner_review']={}
    with pytest.raises(ValueError):reader.checked(interpretation.seal(p),INFERENCE)
    monkeypatch.setattr(pdf,'PARSER_SHA256','0'*64)
    with pytest.raises(ValueError,match='PDF_METHOD_CONFLICT'):pdf.extract(raw)


def test_compressed_expansion_cannot_escape_stream_limit():
    from app_core import nfl_pdf_evidence as pdf
    raw=make_pdf('SYNTHETIC '+('X'*(pdf.MAX_STREAM+1)),compressed=True)
    assert len(raw)<pdf.MAX_BODY
    with pytest.raises(ValueError,match='PDF_LIMIT'):pdf.extract(raw)


def test_isolated_decoder_never_reexecutes_caller_main(monkeypatch):
    from app_core import nfl_pdf_evidence as pdf
    calls=[];original=pdf.subprocess.run
    def checked(args,**kwargs):
        calls.append((args,kwargs));return original(args,**kwargs)
    monkeypatch.setattr(pdf.subprocess,'run',checked)
    raw=make_pdf('SYNTHETIC UNIQUE ISOLATED DECODER CASE')
    ex=pdf.extract(raw)
    assert ex['complete'] and len(calls)==1
    args,kwargs=calls[0]
    assert args[1:4]==['-I','-B','-c'] and 'runpy' not in args[4]
    assert kwargs['input']==raw and kwargs['timeout']==pdf.DEADLINE


def test_decoder_timeout_is_not_a_complete_extraction(monkeypatch):
    from app_core import nfl_pdf_evidence as pdf
    monkeypatch.setattr(pdf,'DEADLINE',.001)
    with pytest.raises(ValueError,match='PDF_TIMEOUT'):pdf.extract(make_pdf('SYNTHETIC UNIQUE TIMEOUT CASE'))


def test_conflicting_dual_extraction_and_rehashed_page_reject(monkeypatch):
    from app_core import nfl_pdf_evidence as pdf
    packet,_=fixture(monkeypatch);obj=deepcopy(packet['payload']['objects']['rules'])
    obj['extraction']['pages'][0]['layout_text']='SYNTHETIC DIFFERENT FACT'
    obj['extraction']['pages'][0]['layout_sha256']=hashlib.sha256(obj['extraction']['pages'][0]['layout_text'].encode()).hexdigest()
    with pytest.raises(ValueError,match='PDF_EXTRACTION_CONFLICT'):pdf.read_pdf(obj)


def test_v4_rejects_existing_receipt_namespace_proof_but_successor_attributes_it(monkeypatch):
    from app_core import nfl_interpreted_research as old,nfl_source_interpretation as old_projection,nfl_pdf_research as new
    packet,_=interpreted_fixture(monkeypatch);p=packet['payload']
    p['projections']['schedule']['payload']['facts']['provider_namespace']['supports']=[dict(object='odds',path=None,text=None,operation='receipt_source_namespace')]
    from test_nfl_source_interpretation import resign as old_sign
    old_sign(p)
    with pytest.raises(ValueError,match='INTERPRETATION_CONFLICT'):old.checked(old_projection.seal(p),INFERENCE)
    packet,_=fixture(monkeypatch)
    assert new.checked(packet,INFERENCE)[0]==packet['payload']['features']
    assert packet['payload']['projections']['schedule']['payload']['facts']['provider_namespace']['status']=='OWNER_INTERPRETED'


@pytest.mark.parametrize('method,value',[('integer',2026),('neutral_site',False),('neutral_site',True)])
def test_actual_json_primitive_types_are_preserved_not_guessed(method,value):
    from app_core import nfl_source_interpretation as old,nfl_pdf_interpretation as new
    with pytest.raises(ValueError):old.normalize(method,value)
    assert new.normalize(method,value) is value


@pytest.mark.parametrize('metadata',[
    {'/Author':'SYNTHETIC api_key=SECRET_METADATA_CANARY'},
    {'/api_key':'SYNTHETIC_SECRET_METADATA_CANARY'},
    {'/Subject':'https://synthetic.invalid/document?access_token=SECRET_METADATA_CANARY'},
])
def test_original_metadata_cannot_hide_credentials(metadata):
    from app_core import nfl_pdf_evidence as pdf
    from pypdf import PdfReader,PdfWriter
    writer=PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf('SYNTHETIC INERT DOCUMENT'))))
    writer.add_metadata(metadata);out=io.BytesIO();writer.write(out)
    with pytest.raises(ValueError,match='CREDENTIAL_FORBIDDEN'):pdf.extract(out.getvalue())
