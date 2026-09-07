import pytest
from app_core.mlb_pitcher_history import parse_boxscore, enrich


def box():
    return {"teams":{s:{"team":{"id":i},"players":{"p":{"person":{"id":i+10},"stats":{"pitching":{
        "gamesStarted":1,"outs":10,"earnedRuns":1,"strikeOuts":4,"baseOnBalls":1,"hits":2}}}}} for s,i in [("home",1),("away",2)]}}


def test_starter_identity_and_outs_not_decimal_innings():
    parsed=parse_boxscore(box(),{"game_id":1,"home_id":1,"away_id":2})
    assert parsed["starters"]=={"home":11,"away":12}
    games={i:{"game_id":i,"season":2023,"completed_at":f"2023-04-0{i}T21:00:00Z"} for i in range(1,5)}
    boxes={str(i):parsed for i in games}
    feature={"game_id":4,"season":2023,"cutoff":"2023-04-04T12:00:00Z"}
    rows,_=enrich([feature],games,boxes)
    assert rows[0]["home_starter_source_game_ids"]==[1,2,3]
    assert rows[0]["home_starter_era"]==pytest.approx(2.7)
    assert rows[0]["home_starter_whip"]==pytest.approx(.9)
    games[3]["completed_at"]="2023-04-05T00:00:00Z"
    assert not enrich([feature],games,boxes)[0]


def test_missing_starter_stats_and_identity_rejected():
    payload=box();payload["teams"]["home"]["players"]["p"]["stats"]["pitching"]["gamesStarted"]=0
    with pytest.raises(ValueError,match="ambiguous"):parse_boxscore(payload,{"game_id":1,"home_id":1,"away_id":2})
    payload=box();del payload["teams"]["home"]["players"]["p"]["stats"]["pitching"]["outs"]
    with pytest.raises(ValueError,match="invalid"):parse_boxscore(payload,{"game_id":1,"home_id":1,"away_id":2})
    with pytest.raises(ValueError,match="identity"):parse_boxscore(box(),{"game_id":1,"home_id":9,"away_id":2})


def test_missing_boxscore_and_cross_season_not_filled():
    f={"game_id":1,"season":2024,"cutoff":"2024-04-01T00:00:00Z"}
    assert enrich([f],{}, {})[1]=={"missing_boxscore":1}
    parsed=parse_boxscore(box(),{"game_id":1,"home_id":1,"away_id":2})
    assert not enrich([f],{1:{"season":2023,"completed_at":"2023-01-01T00:00:00Z"}},{"1":parsed})[0]


def test_development_comparison_and_partial_rejection():
    import json, hashlib
    from datetime import datetime, timedelta, timezone
    from scripts.evaluate_mlb_pitcher_development import evaluate
    state={"games":{}}
    pitchers={"boxes":{},"excluded":{}}
    for year in (2023,2024):
        for i in range(112):
            gid=year*1000+i
            start=datetime(year,4,1,tzinfo=timezone.utc)+timedelta(days=i)
            g={"game_id":gid,"season":year,"cutoff":start.isoformat(),
               "completed_at":(start+timedelta(hours=3)).isoformat(),"home_id":1,"away_id":2,
               "home_score":3+i%3,"away_score":1}
            state["games"][str(gid)]={"record":g}
            pitchers["boxes"][str(gid)]={"data":parse_boxscore(box(),g)}
    raw=json.dumps(state).encode();pitchers["source_hash"]=hashlib.sha256(raw).hexdigest()
    result=evaluate(raw,pitchers)
    assert result["train_games"]==result["development_games"]==102
    assert set(result["metrics"])=={"team_only","team_and_starter"}
    assert result["production_eligible"] is False
    del pitchers["boxes"]["2023000"]
    with pytest.raises(ValueError,match="Complete"):
        evaluate(raw,pitchers)
