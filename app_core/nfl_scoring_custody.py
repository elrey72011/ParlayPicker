"""Separate scoring profile; v1/v2 bodies and their ceilings stay unchanged."""
import base64
import csv
import hashlib
import io
import re
from copy import deepcopy
from app_core import nfl_observed_custody as old, nfl_scoring_profile as profile
from app_core.producer_provenance import clock

VERSION = 'nfl-original-scoring-custody-v3'
require, known, digest, unique, encode = old.require, old.known, old.digest, old.unique, old.encode


def retained_score(body, *, receipt, transfer):
    require(isinstance(body,bytes) and 0 < len(body) <= profile.MAX_BODY, 'NFL_PRIVATE_RESPONSE_SIZE')
    obj=dict(version=VERSION,profile=profile.PROFILE,source=deepcopy(profile.SOURCE),
        body_sha256=hashlib.sha256(body).hexdigest(),bytes_base64=base64.b64encode(body).decode(),
        receipt=deepcopy(receipt),transfer=deepcopy(transfer))
    read_score(obj)
    return obj


def read_score(obj):
    require(isinstance(obj,dict) and set(obj)==set('version profile source body_sha256 bytes_base64 receipt transfer'.split())
        and obj['version']==VERSION and obj['profile']==profile.PROFILE and obj['source']==profile.SOURCE,
        'NFL_PRIVATE_SCORING_PROFILE_CONFLICT')
    encoded=obj['bytes_base64']
    require(isinstance(encoded,str) and len(encoded)<=4*((profile.MAX_BODY+2)//3), 'NFL_PRIVATE_RESPONSE_SIZE')
    try: raw=base64.b64decode(encoded,validate=True)
    except (ValueError,TypeError): raise ValueError('NFL_PRIVATE_CUSTODY_CORRUPT') from None
    require(0 < len(raw) <= profile.MAX_BODY, 'NFL_PRIVATE_RESPONSE_SIZE')
    require(hashlib.sha256(raw).hexdigest()==obj['body_sha256'], 'NFL_PRIVATE_CUSTODY_CORRUPT')
    r=obj['receipt']
    require(isinstance(r,dict) and set(r)==set('source_id endpoint format evidence_label request_started_at received_at observed_at observation_meaning provider_clock_meaning'.split()), 'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    require(r['source_id']==profile.SOURCE_ID and r['endpoint']==profile.ENDPOINT and r['format']=='csv', 'NFL_PRIVATE_SCORING_SOURCE_CONFLICT')
    require(r['evidence_label'] in {'RETAINED','SYNTHETIC'} and r['observation_meaning']=='actual_local_first_observation_of_complete_decoded_body'
        and r['provider_clock_meaning']=='publisher_availability_unknown', 'NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN')
    times=[clock(r[k]) for k in ('request_started_at','received_at','observed_at')]
    require(all(times) and times[0]<=times[1]<=times[2], 'NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT')
    t=obj['transfer']
    require(isinstance(t,dict) and set(t)=={'wire_bytes','wire_sha256','encoding','requests'}
        and type(t['wire_bytes']) is int and 0 < t['wire_bytes'] <= profile.MAX_WIRE
        and isinstance(t['wire_sha256'],str) and re.fullmatch('[0-9a-f]{64}',t['wire_sha256'])
        and t['encoding'] in {'identity','gzip'}, 'NFL_PRIVATE_SCORING_TRANSFER_CONFLICT')
    require(t['encoding']!='identity' or t['wire_bytes']==len(raw) and t['wire_sha256']==obj['body_sha256'], 'NFL_PRIVATE_SCORING_TRANSFER_CONFLICT')
    chain=t['requests']
    require(isinstance(chain,list) and len(chain) in {1,2}, 'NFL_PRIVATE_SCORING_TRANSFER_CONFLICT')
    from app_core.nfl_scoring_acquisition import PRIMARY, validate_hop_descriptor
    previous=times[0]
    for index,step in enumerate(chain):
        require(set(step)=={'attempt','request','started_at','received_at','status','location_sha256'} and step['attempt']==index+1,
            'NFL_PRIVATE_SCORING_TRANSFER_CONFLICT')
        begin,end=clock(step['started_at']),clock(step['received_at'])
        require(begin and end and previous<=begin<=end<=times[1], 'NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT')
        previous=end
        if index==0:
            require(step['request']==PRIMARY and begin==times[0], 'NFL_PRIVATE_SCORING_SOURCE_CONFLICT')
        else: validate_hop_descriptor(step['request'])
        if index==len(chain)-1:
            require(step['status']==200 and step['location_sha256'] is None and end==times[1], 'NFL_PRIVATE_SCORING_TRANSFER_CONFLICT')
        else:
            require(step['status']==302 and isinstance(step['location_sha256'],str) and re.fullmatch('[0-9a-f]{64}',step['location_sha256']), 'NFL_PRIVATE_SCORING_TRANSFER_CONFLICT')
    require(not re.search(rb'(?i)(api[_-]?key|access[_-]?token|authorization|client_secret|password)\s*["\x27:=]',raw), 'NFL_PRIVATE_CREDENTIAL_FORBIDDEN')
    try:
        reader=csv.reader(io.StringIO(raw.decode('utf-8'),newline=''));fields=next(reader,None)
        require(fields and len(fields)==len(set(fields))
            and {'game_id','season','gameday','home_team','away_team','home_score','away_score','result'}<=set(fields), 'NFL_PRIVATE_RESPONSE_SCHEMA')
        rows=[]
        for values in reader:
            if not values:continue # Same empty-line behavior as the preserved DictReader.
            require(len(values)==len(fields), 'NFL_PRIVATE_RESPONSE_SCHEMA')
            rows.append(dict(zip(fields,values)))
        require(rows, 'NFL_PRIVATE_RESPONSE_SCHEMA')
    except (UnicodeError,ValueError,TypeError,csv.Error): raise ValueError('NFL_PRIVATE_RESPONSE_SCHEMA') from None
    return raw,rows


def read(obj):
    return read_score(obj) if isinstance(obj,dict) and obj.get('version')==VERSION else old.read(obj)


def objects(items):
    require(isinstance(items,dict) and 0 < len(items) <= profile.MAX_OBJECTS and 'scores' in items, 'NFL_PRIVATE_CUSTODY_MISSING')
    result={}; total=0; identities=set()
    for key,obj in items.items():
        require((key=='scores')==(obj.get('version')==VERSION), 'NFL_PRIVATE_SCORING_PROFILE_CONFLICT')
        raw,value=read_score(obj) if key=='scores' else old.read(obj)
        require(obj['body_sha256'] not in identities, 'NFL_PRIVATE_DUPLICATE_RESPONSE')
        identities.add(obj['body_sha256']); total+=len(raw)
        require(total<=profile.MAX_TOTAL, 'NFL_PRIVATE_RESPONSE_SIZE')
        result[key]=value
    return result
