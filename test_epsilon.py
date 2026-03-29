import hashlib
for i in range(10):
    matchup_id = "test"
    game_date = "date"
    market_type = f"market_{i}"
    seed_str = f"{matchup_id}|{game_date}|{market_type}"
    md5_hash = hashlib.md5(seed_str.encode()).hexdigest()
    epsilon = (int(md5_hash[:8], 16) / 0xFFFFFFFF) * 9e-7
    print(f"seed: {seed_str}, epsilon: {epsilon}")
