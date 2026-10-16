"""Fixed scanner facts only. No content, offsets, lengths or arbitrary strings."""
VERSION='football-html-scan-diagnostics-v1'
SCANNER='football-html-structural-scanner-v3'
UNKNOWN='UNKNOWN'
RULES=frozenset(('NONE','API_KEY_ASSIGNMENT','ACCESS_TOKEN_ASSIGNMENT','AUTHORIZATION_ASSIGNMENT','CLIENT_SECRET_ASSIGNMENT','PASSWORD_ASSIGNMENT','UNKNOWN'))
STAGES=frozenset(('HTML_FORMAT','HTML_NORMALIZATION','HTML_STRUCTURE','HTML_ASSIGNMENT','UNKNOWN'))
FIELDS=frozenset(('version','scanner_identity','stage','matched_rule','structural_syntax_applied','public_placeholder_applied'))


def initial():
    return dict(version=VERSION,scanner_identity=SCANNER,stage=UNKNOWN,matched_rule=UNKNOWN,
        structural_syntax_applied=UNKNOWN,public_placeholder_applied=UNKNOWN)


def rule(match):
    if match is None:return 'NONE'
    # The regex is fixed by the scanner. Only its recognized label selects an
    # enum; no matching value or text is returned, retained or hashed here.
    import re
    label=re.match(r'[a-z_-]+',match.group(0),re.I).group(0).lower().replace('_','').replace('-','')
    return {'apikey':'API_KEY_ASSIGNMENT','accesstoken':'ACCESS_TOKEN_ASSIGNMENT',
        'authorization':'AUTHORIZATION_ASSIGNMENT','clientsecret':'CLIENT_SECRET_ASSIGNMENT',
        'password':'PASSWORD_ASSIGNMENT'}.get(label,UNKNOWN)


def checked(value):
    if not isinstance(value,dict) or set(value)!=FIELDS or value['version']!=VERSION or value['scanner_identity']!=SCANNER:
        raise ValueError('DOCUMENT_SCAN_DIAGNOSTICS_INVALID')
    if value['stage'] not in STAGES or value['matched_rule'] not in RULES:
        raise ValueError('DOCUMENT_SCAN_DIAGNOSTICS_INVALID')
    for key in ('structural_syntax_applied','public_placeholder_applied'):
        if type(value[key]) is not bool and value[key]!=UNKNOWN:raise ValueError('DOCUMENT_SCAN_DIAGNOSTICS_INVALID')
    return dict(value)


class Failure(ValueError):
    def __init__(self,code,facts):
        if code not in {'DOCUMENT_FORMAT','DOCUMENT_CREDENTIAL_BODY'}:raise ValueError('DOCUMENT_SCAN_DIAGNOSTICS_INVALID')
        super().__init__(code);self.scanner_diagnostics=checked(facts)
