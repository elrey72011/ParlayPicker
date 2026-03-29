import hashlib
row_id = 'NHL_15'
market_type = 'h2h_home'
game_date = '2026-03-30'
seed_str = f"{row_id}|{game_date}|{market_type}"
md5_hash = hashlib.md5(seed_str.encode()).hexdigest()
epsilon = (int(md5_hash[:8], 16) / 0xFFFFFFFF) * 9e-7
print(epsilon)
