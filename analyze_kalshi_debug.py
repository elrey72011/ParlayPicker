import json

with open('app_exports/parlay_debug_history - 2026-02-20T145530.372.json', 'r') as f:
    debug_history = json.load(f)

print(f"Total entries: {len(debug_history)}")

for i, entry in enumerate(debug_history):
    home = entry.get('Home', 'Unknown')
    away = entry.get('Away', 'Unknown')
    market = entry.get('Market', 'Unknown')
    print(f"{i}: {away} @ {home} ({market})")
    # Check if there are any kalshi specific fields that might contain candidate info
    # The prompt implies there should be candidate info.
    # Maybe it's in a nested field?
    # Let's print all keys for the first entry again
    if i == 0:
        print("Keys:", entry.keys())
