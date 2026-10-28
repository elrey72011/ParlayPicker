"""Verify full immutable collection before filtering in its original order."""
from pathlib import Path
import hashlib,json,os
PLAN=json.loads((Path(__file__).parent/'selection.json').read_bytes())
CONTEXT=next(p for p in PLAN['contexts'] if p['name']==os.environ['CHROME_CONTEXT'])

def pytest_collection_modifyitems(session,config,items):
    nodes=[item.nodeid for item in items]
    digest=hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest()
    assert len(nodes)==len(set(nodes))==CONTEXT['collected_count'],'Unexpected collection count'
    assert digest==CONTEXT['collected_sha256'],'Unexpected collection identities/order'
    wanted=CONTEXT['expected_nodeids'];selected=[item for item in items if item.nodeid in set(wanted)]
    assert [item.nodeid for item in selected]==wanted,'Unexpected selection identities/order'
    removed=[item for item in items if item.nodeid not in set(wanted)]
    Path(os.environ['CHROME_SELECTION_RECEIPT']).write_text(json.dumps(dict(
        collected_count=len(nodes),collected_sha256=digest,selected_nodeids=wanted),indent=2)+'\n')
    config.hook.pytest_deselected(items=removed);items[:]=selected
