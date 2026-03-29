from app_core.prediction_engine import _build_matchup_id
import hashlib

def test():
    home_team = 'Lakers'
    away_team = 'Warriors'
    game_date = '2026-03-30'
    market_type = 'spread_home'

    matchup_id = _build_matchup_id(home_team, away_team)

    seed_str = f"{matchup_id}|{game_date}|{market_type}"
    print(f"seed_str: {seed_str}")
    md5_hash = hashlib.md5(seed_str.encode()).hexdigest()
    epsilon = (int(md5_hash[:8], 16) / 0xFFFFFFFF) * 9e-7
    print(epsilon)

test()
