"""Actual retained candidate ordering, display and warm mutation rejection."""
from copy import deepcopy
from datetime import timedelta
import json
import pytest
from app_core import per_game_boards as boards
from test_ncaaf_staged_evidence import fixture, owner, NOW, synthetic


@pytest.mark.parametrize('mutation',['result_hash','original_packet'])
def test_private_board_revalidates_ordering_and_rejects_later_mutation(synthetic,monkeypatch,tmp_path,mutation):
    packet,row=fixture(synthetic,monkeypatch,tmp_path)
    before=deepcopy(packet)
    analysis=owner.admitted_analysis(packet,row)
    import test_source_contract_pipeline as exporting
    monkeypatch.setattr(exporting,'NOW',NOW)
    monkeypatch.setattr(exporting,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:NOW)
    saved=exporting.exported(monkeypatch,tmp_path/'export',analysis)
    original=boards._compatible_ncaaf_research;calls=[]
    def observed(source):
        calls.append(source.get('candidate_id'))
        return original(source)
    monkeypatch.setattr(boards,'_compatible_ncaaf_research',observed)
    def display(candidates):
        frame=boards.per_game_board(saved['card'],candidates,family='sides',novig_only=True,
                                   private_research=True,college_fallback=True)
        return json.loads(frame.iloc[0].research_display)
    first=display(saved['captured'])
    assert first['probability']==pytest.approx(.636830651175619)
    assert calls and len(calls)==len(set(calls))
    changed=saved['captured'].copy(deep=True)
    for index in changed.index:
        metadata=json.loads(changed.at[index,'ml_estimate_metadata'])
        if mutation=='result_hash':metadata['ncaaf_inputs']['sha256']='0'*64
        else:metadata['ncaaf_inputs']['payload']['original_packet']['sha256']='0'*64
        changed.at[index,'ml_estimate_metadata']=json.dumps(metadata)
    calls.clear()
    assert display(changed)['probability'] is None
    assert calls
    assert display(saved['captured'])['probability']==first['probability']
    assert packet==before
