"""Explicit original-PDF custody and bounded, inert deterministic extraction.

Original bytes never become a text response. Extraction supplies no interpretation,
permission, review, listing applicability, publication clock or numerical result.
"""
import base64
from copy import deepcopy
from collections import Counter
from functools import lru_cache
import hashlib
import io
import json
import subprocess
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from app_core import nfl_scoring_custody as old, nfl_scoring_profile as profile
from app_core.producer_provenance import clock

VERSION = 'nfl-original-pdf-custody-v1'
EXTRACTION = 'nfl-pdf-page-extraction-v1'
PARSER = '6.20.0'
PARSER_SHA256 = '4819570a7b4cf08f7f7f0f038fec9dc34561f0d75ae463c6a3b8f38e652a304d'
MAX_BODY = 512 * 1024
MAX_PAGES = 32
MAX_TEXT = 128 * 1024
MAX_PAGE_TEXT = 16 * 1024
MAX_STREAM = 512 * 1024
MAX_STREAMS_TOTAL = 4 * 1024 * 1024
MAX_REFERENCES = 4096
DEADLINE = 8
require, digest, known, unique, encode = old.require, old.digest, old.known, old.unique, old.encode


def _credential_free(value):
    text=value.decode('latin-1') if isinstance(value,bytes) else value
    require(not re.search(r'(?i)(api[_-]?key|access[_-]?token|authorization|client_secret|password)\s*["\x27:=]',text)
        and not re.search(r'(?i)[?&](token|api[_-]?key|access[_-]?token|authorization|password|x-amz-[^=&]+|signature|sig|credential)=',text),
        'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')
    if text.startswith(('https://','http://')):
        u=urlsplit(text)
        require(not u.username and not u.password,'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')


def parser_identity():
    import pypdf
    root = Path(pypdf.__file__).parent
    files = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))}
    require(pypdf.__version__ == PARSER and digest(files) == PARSER_SHA256, 'NFL_PRIVATE_PDF_METHOD_CONFLICT')
    return dict(library='pypdf', version=PARSER, package_sources_sha256=PARSER_SHA256)


def _parse_worker(raw, connection):
    # This disposable worker cannot use networking. No PDF actions are executed.
    import socket, logging
    def denied(*a, **k): raise RuntimeError('PDF_NETWORK_FORBIDDEN')
    socket.socket = socket.create_connection = socket.getaddrinfo = denied
    logging.getLogger('pypdf').handlers = [logging.NullHandler()]
    logging.getLogger('pypdf').propagate = False
    try:
        parser_identity()
        from pypdf import PdfReader
        from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject
        from pypdf import overwrite_configuration
        overwrite_configuration(zlib_maximum_output_length=MAX_STREAM,
            lzw_maximum_output_length=MAX_STREAM,jbig2_maximum_output_length=MAX_STREAM,
            maximum_declared_stream_length=MAX_STREAM,array_based_stream_maximum_output_length=MAX_STREAMS_TOTAL,
            stream_filters_maximum_length=1,stream_decoding_work_maximum_length=MAX_STREAMS_TOTAL,
            page_tree_maximum_entries=MAX_REFERENCES,page_tree_maximum_depth=32,
            xform_maximum_invocations_per_extraction=128,jbig2dec_binary=None)
        require(raw.startswith(b'%PDF-') and raw.rstrip().endswith(b'%%EOF'),'NFL_PRIVATE_PDF_INCOMPLETE')
        require(raw.count(b'%PDF-') == raw.count(b'%%EOF') == 1,'NFL_PRIVATE_PDF_UNSUPPORTED')
        reader = PdfReader(io.BytesIO(raw), strict=True)
        require(not reader.is_encrypted and '/Encrypt' not in reader.trailer, 'NFL_PRIVATE_PDF_UNSUPPORTED')
        require(not reader.xref_objStm, 'NFL_PRIVATE_PDF_UNSUPPORTED')
        references = [(generation,number) for generation,table in reader.xref.items() for number in table if number]
        require(len(references) <= MAX_REFERENCES, 'NFL_PRIVATE_PDF_LIMIT')
        seen = set(); stream_total = 0
        forbidden = {'/OpenAction','/AA','/JavaScript','/JS','/Launch','/EmbeddedFiles','/EmbeddedFile','/XFA','/AcroForm','/RichMedia','/OCProperties'}
        def inspect(value, depth=0):
            nonlocal stream_total
            require(depth <= 32, 'NFL_PRIVATE_PDF_LIMIT')
            if isinstance(value, IndirectObject):
                key=(value.idnum,value.generation)
                if key in seen:return
                seen.add(key);require(len(seen)<=MAX_REFERENCES,'NFL_PRIVATE_PDF_LIMIT')
                inspect(value.get_object(),depth+1)
            elif isinstance(value, DictionaryObject):
                require(not forbidden.intersection(value.keys()), 'NFL_PRIVATE_PDF_ACTIVE_CONTENT')
                require(not any(str(k).lstrip('/').lower() in {'api_key','apikey','access_token','authorization','client_secret','password'} for k in value),'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')
                if value.get('/Type') == '/Action' or value.get('/S') in {'/JavaScript','/Launch','/GoToR','/SubmitForm','/ImportData','/Rendition','/Movie','/Sound'}:
                    require(value.get('/S') in {None,'/URI','/GoTo'}, 'NFL_PRIVATE_PDF_ACTIVE_CONTENT')
                if isinstance(value, StreamObject):
                    filters=value.get('/Filter')
                    if value.get('/Subtype')=='/Image':
                        require(filters in (None,'/FlateDecode','/DCTDecode'),'NFL_PRIVATE_PDF_UNSUPPORTED')
                        data=value._data  # Inert image bytes; never decoded or executed.
                    else:
                        require(filters in (None,'/FlateDecode') and not value.get('/DecodeParms'), 'NFL_PRIVATE_PDF_UNSUPPORTED')
                        data=value.get_data()
                    require(len(data)<=MAX_STREAM,'NFL_PRIVATE_PDF_LIMIT')
                    _credential_free(data)
                    stream_total+=len(data);require(stream_total<=MAX_STREAMS_TOTAL,'NFL_PRIVATE_PDF_LIMIT')
                for item in value.values():inspect(item,depth+1)
            elif isinstance(value, ArrayObject):
                require(len(value)<=MAX_REFERENCES,'NFL_PRIVATE_PDF_LIMIT')
                for item in value:inspect(item,depth+1)
            elif isinstance(value,(str,bytes)):_credential_free(value)
        for generation,number in references:inspect(IndirectObject(number,generation,reader))
        pages=[];total=0
        require(0 < len(reader.pages) <= MAX_PAGES, 'NFL_PRIVATE_PDF_LIMIT')
        for index,page in enumerate(reader.pages):
            require(int(page.get('/Rotate',0))==0,'NFL_PRIVATE_PDF_AMBIGUOUS')
            contents=page.get_contents()
            require(contents is not None,'NFL_PRIVATE_PDF_UNREADABLE')
            for operands,op in contents.operations:
                if op==b'Tr':require(int(operands[0])==0,'NFL_PRIVATE_PDF_AMBIGUOUS')
                if op==b'Tm':require(float(operands[1])==float(operands[2])==0,'NFL_PRIVATE_PDF_AMBIGUOUS')
            text=page.extract_text(extraction_mode='plain').replace('\r\n','\n').replace('\r','\n')
            layout=page.extract_text(extraction_mode='layout')
            require(text.strip() and '\ufffd' not in text and '\x00' not in text,'NFL_PRIVATE_PDF_UNREADABLE')
            require(Counter(''.join(text.split()))==Counter(''.join(layout.split())),'NFL_PRIVATE_PDF_AMBIGUOUS')
            size=len(text.encode())+len(layout.encode());total+=size
            require(size<=MAX_PAGE_TEXT and total<=MAX_TEXT,'NFL_PRIVATE_PDF_LIMIT')
            require(not re.search(r'(?i)(api[_-]?key|access[_-]?token|authorization|client_secret|password)\s*["\x27:=]',text), 'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')
            pages.append(dict(page=index,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                              layout_text=layout,layout_sha256=hashlib.sha256(layout.encode()).hexdigest()))
        connection.send_bytes(json.dumps(dict(status='PASS',pages=pages),separators=(',',':')).encode())
    except Exception as e:
        allowed={'NFL_PRIVATE_PDF_LIMIT','NFL_PRIVATE_PDF_INCOMPLETE','NFL_PRIVATE_PDF_UNSUPPORTED','NFL_PRIVATE_PDF_ACTIVE_CONTENT','NFL_PRIVATE_PDF_UNREADABLE','NFL_PRIVATE_PDF_AMBIGUOUS','NFL_PRIVATE_PDF_METHOD_CONFLICT','NFL_PRIVATE_CREDENTIAL_FORBIDDEN'}
        code=str(e) if type(e) is ValueError and str(e) in allowed else 'NFL_PRIVATE_PDF_UNREADABLE'
        if type(e).__name__=='LimitReachedError':code='NFL_PRIVATE_PDF_LIMIT'
        connection.send_bytes(json.dumps(dict(status='FAIL',code=code)).encode())
    finally:connection.close()


@lru_cache(maxsize=2)
def _pages(raw, method_sha256, parser_sha256):
    # Pure exact-content extraction only; no review/admission decisions or clocks.
    import pypdf
    bootstrap = """import sys
sys.path[:0]=sys.argv[1:3]
from app_core import nfl_pdf_evidence as p
class Output:
 def send_bytes(self, value):sys.stdout.buffer.write(value)
 def close(self):sys.stdout.buffer.flush()
p._parse_worker(sys.stdin.buffer.read(p.MAX_BODY+1),Output())
"""
    try:
        # Never import/reexecute a caller's __main__ (e.g. Streamlit's app).
        output=subprocess.run([sys.executable,'-I','-B','-c',bootstrap,str(Path(pypdf.__file__).parent.parent),str(Path(__file__).resolve().parents[1])],
            input=raw,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=DEADLINE,check=False)
        require(output.returncode==0 and 0<len(output.stdout)<=MAX_TEXT*4+65536,'NFL_PRIVATE_PDF_UNREADABLE')
        result=json.loads(output.stdout)
        require(result.get('status')=='PASS',result.get('code','NFL_PRIVATE_PDF_UNREADABLE'))
        return result['pages']
    except subprocess.TimeoutExpired:raise ValueError('NFL_PRIVATE_PDF_TIMEOUT') from None
    except (EOFError,OSError,TypeError,json.JSONDecodeError):raise ValueError('NFL_PRIVATE_PDF_UNREADABLE') from None


def extract(raw):
    require(isinstance(raw,bytes) and 0<len(raw)<=MAX_BODY,'NFL_PRIVATE_RESPONSE_SIZE')
    _credential_free(raw)
    parser=parser_identity();method_sha=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    pages=deepcopy(_pages(raw,method_sha,PARSER_SHA256))
    return dict(version=EXTRACTION,pdf_sha256=hashlib.sha256(raw).hexdigest(),complete=True,
        method=dict(parser,implementation_sha256=method_sha,mode='dual-mode-complete-glyphs-and-located-context-agreement',
                    python=sys.version.split()[0],max_pages=MAX_PAGES,max_text_bytes=MAX_TEXT,max_stream_bytes=MAX_STREAM,deadline_seconds=DEADLINE),
        page_count=len(pages),pages=pages)


def retained_pdf(raw, *, receipt, prepared_at):
    extraction=extract(raw);extraction['prepared_at']=prepared_at
    obj=dict(version=VERSION,body_sha256=hashlib.sha256(raw).hexdigest(),bytes_base64=base64.b64encode(raw).decode(),
             receipt=deepcopy(receipt),extraction=extraction)
    read_pdf(obj);return obj


def read_pdf(obj):
    require(isinstance(obj,dict) and set(obj)=={'version','body_sha256','bytes_base64','receipt','extraction'} and obj['version']==VERSION,'NFL_PRIVATE_CUSTODY_MISSING')
    encoded=obj['bytes_base64'];require(isinstance(encoded,str) and len(encoded)<=4*((MAX_BODY+2)//3),'NFL_PRIVATE_RESPONSE_SIZE')
    try:raw=base64.b64decode(encoded,validate=True)
    except (ValueError,TypeError):raise ValueError('NFL_PRIVATE_CUSTODY_CORRUPT') from None
    require(0<len(raw)<=MAX_BODY and hashlib.sha256(raw).hexdigest()==obj['body_sha256'],'NFL_PRIVATE_CUSTODY_CORRUPT')
    r=obj['receipt'];require(isinstance(r,dict) and set(r)==set('source_id endpoint format evidence_label request_started_at received_at observed_at observation_meaning provider_clock_meaning'.split()),'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    u=urlsplit(r['endpoint'])
    require(u.scheme=='https' and u.hostname and not u.username and not u.password and not u.query and not u.fragment,'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')
    require(known(r['source_id']) and r['evidence_label'] in {'RETAINED','SYNTHETIC'} and r['format']=='pdf'
            and r['observation_meaning']=='actual_local_first_observation_of_complete_decoded_body' and r['provider_clock_meaning']=='attributed_document_no_publication_clock','NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    times=[clock(r[k]) for k in ('request_started_at','received_at','observed_at')];prepared=clock(obj['extraction'].get('prepared_at'))
    require(all(times) and times[0]<=times[1]<=times[2] and prepared and times[2]<=prepared,'NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT')
    expected=extract(raw);expected['prepared_at']=obj['extraction']['prepared_at']
    require(obj['extraction']==expected,'NFL_PRIVATE_PDF_EXTRACTION_CONFLICT')
    return raw,deepcopy(expected)


def read(obj):return read_pdf(obj) if isinstance(obj,dict) and obj.get('version')==VERSION else old.read(obj)


def objects(items):
    require(isinstance(items,dict) and 0<len(items)<=profile.MAX_OBJECTS and 'scores' in items,'NFL_PRIVATE_CUSTODY_MISSING')
    result={};total=0;seen=set()
    for key,obj in items.items():
        require((key=='scores')==(obj.get('version')==old.VERSION),'NFL_PRIVATE_SCORING_PROFILE_CONFLICT')
        raw,value=read(obj);total+=len(raw)
        require(total<=profile.MAX_TOTAL,'NFL_PRIVATE_RESPONSE_SIZE')
        require(obj['body_sha256'] not in seen,'NFL_PRIVATE_DUPLICATE_RESPONSE');seen.add(obj['body_sha256']);result[key]=value
    require(any(o.get('version')==VERSION for k,o in items.items() if k in {'schedule','listing','rules','clock_document'}),'NFL_PRIVATE_PDF_ROUTE_REQUIRED')
    return result


def proof(objects, support, originals):
    from app_core.nfl_source_projection import proof as old_proof
    if support.get('operation')!='pdf_page_text':return old_proof(objects,support,originals)
    require(set(support)=={'object','path','text','operation','pdf'} and support['path'] is support['text'] is None,'NFL_PRIVATE_PDF_LOCATOR_CONFLICT')
    name=support['object'];require(name in objects and originals[name].get('version')==VERSION,'NFL_PRIVATE_PDF_LOCATOR_CONFLICT')
    extraction=objects[name];p=support['pdf']
    require(set(p)=={'page','start','end','exact','page_sha256','extraction_sha256'} and type(p['page']) is int and 0<=p['page']<len(extraction['pages']) and p['extraction_sha256']==digest(extraction),'NFL_PRIVATE_PDF_LOCATOR_CONFLICT')
    page=extraction['pages'][p['page']];text=page['text']
    require(p['page_sha256']==page['text_sha256'] and type(p['start']) is int and type(p['end']) is int and 0<=p['start']<p['end']<=len(text) and text[p['start']:p['end']]==p['exact'],'NFL_PRIVATE_PDF_LOCATOR_CONFLICT')
    context=' '.join(text[max(0,p['start']-64):min(len(text),p['end']+64)].split())
    require(' '.join(page['layout_text'].split()).count(context)==1,'NFL_PRIVATE_PDF_AMBIGUOUS')
    return p['exact']
