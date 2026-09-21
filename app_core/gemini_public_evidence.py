"""Public provenance summary, never a factual veto or wager authorization."""
import json
import re
from app_core.result_reconciliation import stamp


def evidence_summary(row):
    def decode(key, default):
        value = row.get(key)
        if isinstance(value, type(default)):
            return value
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, type(default)) else default
        except (ValueError, TypeError):
            return default
    context = decode('gemini_verified_context', {})
    cited = decode('gemini_supporting_evidence', [])
    reviewed = stamp(row.get('gemini_reviewed_at'))
    labels = {'probable_pitchers':'Probable pitchers','lineups':'Lineups','injuries':'Injuries','weather':'Weather'}
    supported=[]
    for key,label in labels.items():
        fact=context.get(key)
        if key not in cited or not isinstance(fact,dict) or reviewed is None:
            continue
        at=stamp(fact.get('recorded_at'))
        source=fact.get('source')
        # Only simple provider labels are public; URLs, credentials and raw facts
        # remain private. Supplied evidence is not independent web verification.
        if not isinstance(source,str) or not re.fullmatch(r'[A-Za-z0-9 ._-]{1,60}',source):
            continue
        if at is None or not 0 <= (reviewed-at).total_seconds() <= 3600 or fact.get('value') is None:
            continue
        supported.append(f'{label}: {source}, observed {at.isoformat()}')
    unknown=[label for key,label in labels.items() if not any(s.startswith(label+':') for s in supported)]
    return ('Cited supplied evidence: '+ '; '.join(supported)+'. ' if supported else 'No timestamped supplied evidence citations verified. ') + 'Not established by this review: '+', '.join(unknown)+'.' if unknown else 'Cited supplied evidence: '+'; '.join(supported)+'. Independent verification is not established.'
