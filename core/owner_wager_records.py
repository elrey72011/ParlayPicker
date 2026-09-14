"""Owner records and value rechecks. No external execution capability."""
from core.wager_decisions import decimal_price, finite


def placement_record(recommendation, *, sportsbook, line, odds, stake, bet_id, snapshot_id, team_ids):
    c=recommendation
    actual=decimal_price(odds);p=finite(c.get('conservative_probability'))
    same_line=finite(line)==finite(c.get('line'))
    ev=p*actual-1 if same_line and p is not None and actual is not None else None
    warning=('Line changed: original probability does not validate this wager.' if not same_line else
             'Value unavailable or nonpositive at the actual price.' if ev is None or ev<=0 else '')
    return {'status':'COMMITTED','bet_id':bet_id,'source_snapshot_id':snapshot_id,'sportsbook':sportsbook,
        'stake_dollars':stake,'recommendation_price_changed':odds!=c.get('odds'),'actual_conservative_ev':ev,
        'value_warning':warning,'legs':[{'sport':c['sport'],'game_id':c['game_id'],'team_ids':team_ids,
          'market':c['market_type'],'selection':c['selection'],'line':line,'odds':odds}]}
