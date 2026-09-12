"""Fail closed on incomplete game inputs before presenting recommendations."""
import math


def quality_reason(row):
    def value(name):
        result=str(row.get(name, '')).strip().lower()
        return '' if result in {'nan', 'none', '<na>'} else result
    if value('feature_stats_fallback') in {'true','1','1.0'} or value('stats_source') in {'fallback','failed'} or value('stats_resolution_status') == 'unresolved':
        return 'Incomplete team statistics; research only'
    if value('degraded_feature_subset_flag') in {'true','1','1.0'} or value('model_status') in {'statistical fallback','neutral fallback','model failure'}:
        return 'Degraded model inputs; research only'
    if value('stats_fallback_reason'):
        return 'Team statistics used fallback data; research only'
    if not value('stats_source') or value('stats_resolution_status') != 'resolved':
        return 'Team data quality not verified; refresh analysis'
    return ''


def positive_price_edge(probability, odds, ev):
    try:
        probability,odds,ev=map(float,(probability,odds,ev))
        if not all(map(math.isfinite,(probability,odds,ev))) or not 0<probability<1 or not 100<=abs(odds)<=10000:
            return False
        decimal=1+(odds/100 if odds>0 else 100/abs(odds))
        return ev>0 and probability*decimal-1>0
    except (TypeError,ValueError):
        return False
