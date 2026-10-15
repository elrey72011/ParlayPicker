"""V3 validation-only HTML syntax classification. Original bytes never change."""
from html import unescape
from html.parser import HTMLParser
import re
from urllib.parse import unquote

LABEL = r'(?:api[_-]?key|access[_-]?token|authorization|client_secret|password)'
CREDENTIAL = re.compile(LABEL+r'\s*["\x27:=]', re.I)
STRUCTURAL = frozenset(('a','section','div','nav','article','li','span','p','h1','h2','h3','h4','h5','h6'))
ATTRIBUTE = re.compile(r'(?<![\w:-])(id|name|href)\s*=\s*(["\x27])(#?'+LABEL+r')\2(?=\s|/?>)', re.I)

class Anchors(HTMLParser):
    def __init__(self,text):
        super().__init__(convert_charrefs=False)
        self.text=text;self.ranges=[]
        self.lines=[0]+[m.end() for m in re.finditer('\n',text)]
    def handle_starttag(self,tag,attrs):
        if tag not in STRUCTURAL:return
        # Form values or duplicate attributes are not harmless anchor evidence.
        names=[k for k,v in attrs]
        if len(set(names))!=len(names) or any(k in names for k in ('value','data-value','content')):return
        line,column=self.getpos();start=self.lines[line-1]+column
        original=self.get_starttag_text()
        for match in ATTRIBUTE.finditer(original):
            name,value=match.group(1).lower(),match.group(3)
            allowed=(name=='id' and not value.startswith('#')) or (tag=='a' and name=='name' and not value.startswith('#')) or (tag=='a' and name=='href' and value.startswith('#'))
            if allowed:self.ranges.append((start+match.start(3),start+match.end(3)))
    handle_startendtag=handle_starttag

def checked(raw, *, public_guide_examples=False):
    """Raise a stable code, never the matching content or a parser exception."""
    from app_core.nfl_observed_custody import require
    try:text=raw.decode('utf-8')
    except (UnicodeError,AttributeError):raise ValueError('DOCUMENT_FORMAT') from None
    require(text.strip() and re.search(r'(?i)<(?:!doctype\s+html|html)(?:\s|>)',text[:4096]),'DOCUMENT_FORMAT')
    # Decode bounded textual encodings for validation only. This prevents HTML
    # entities, percent escapes and ASCII Unicode escapes concealing assignments.
    scan=text
    for _ in range(2):
        try:scan=unescape(unquote(scan,errors='strict'))
        except UnicodeError:raise ValueError('DOCUMENT_FORMAT') from None
        scan=re.sub(r'\\u00([0-9a-fA-F]{2})',lambda m:chr(int(m.group(1),16)),scan)
        scan=scan.replace('\\"','"').replace("\\'","'")
    try:require(unescape(unquote(scan,errors='strict'))==scan,'DOCUMENT_FORMAT')
    except UnicodeError:raise ValueError('DOCUMENT_FORMAT') from None
    require(len(scan)<=4*len(text)+16,'DOCUMENT_FORMAT')
    parser=Anchors(scan)
    try:parser.feed(scan);parser.close()
    except (ValueError,TypeError,IndexError):raise ValueError('DOCUMENT_FORMAT') from None
    # Only exact attribute-value locations are excluded from the validation
    # view. No body-wide/domain-wide bypass and no original evidence redaction.
    for start,end in sorted(parser.ranges,reverse=True):scan=scan[:start]+' '*(end-start)+scan[end:]
    if public_guide_examples:
        for literal in ('apiKey=YOUR_API_KEY','apiKey={apiKey}'):
            scan=re.sub(r'(?<![A-Za-z0-9_])'+re.escape(literal)+r'(?=$|[&\s"\x27<>])','PUBLIC_DOCUMENTATION_LITERAL',scan)
    require(not CREDENTIAL.search(scan),'DOCUMENT_CREDENTIAL_BODY')
