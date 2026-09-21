from app_core.gemini_public_evidence import evidence_summary
from app_core.price_value_display import display


def test_evidence_requires_real_key_source_and_fresh_timestamp():
    row={'gemini_reviewed_at':'2026-09-21T12:00:00Z','gemini_supporting_evidence':['weather','invented'],
         'gemini_verified_context':{'weather':{'value':'Rain','source':'NOAA','recorded_at':'2026-09-21T11:45:00Z'}}}
    assert 'Weather: NOAA' in evidence_summary(row)
    row['gemini_verified_context']['weather']['recorded_at']='2026-09-20T12:00:00Z'
    assert 'No timestamped' in evidence_summary(row)
    row['gemini_verified_context']['weather']['source']='https://private?token=secret'
    assert 'secret' not in evidence_summary(row)


def test_negligible_value_is_display_only():
    assert display(.5001,100,.0002)['value_status']=='NEGLIGIBLE ESTIMATED VALUE'
    assert display(.5001,100,.0002)['estimated_expected_value']==.0002
    assert display(.6,100,.2)['value_status']=='POSITIVE ESTIMATED VALUE'
