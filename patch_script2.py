import re

with open('app_core/kalshi_integrator.py', 'r') as f:
    content = f.read()

# Remove candidate/date_code stuff
content = re.sub(
    r'        game_date = pd\.to_datetime\(row\.get\("game_date"\), errors="coerce", utc=True\)\n.*?(?=        if not series:)',
    '',
    content,
    flags=re.DOTALL
)

# Insert the base date_code check removal part cleanly
content = content.replace(
    '        if not series:\n            out.at[idx, "kalshi_match_status"] = "miss"\n            out.at[idx, "kalshi_match_reason"] = "missing_series"\n            continue\n\n        base = f"{series}-{date_code}"\n        candidates = [f"{base}{away_code}{home_code}", f"{base}{home_code}{away_code}"]',
    '        if not series:\n            out.at[idx, "kalshi_match_status"] = "miss"\n            out.at[idx, "kalshi_match_reason"] = "missing_series"\n            continue'
)

# Remove the date_code filtering from series_markets extraction
content = content.replace(
    '            # If the date_code is present in the game row, filter by it roughly if possible\n            if date_code and date_code not in ticker and date_code not in ev_ticker:\n                continue',
    ''
)

with open('app_core/kalshi_integrator.py', 'w') as f:
    f.write(content)
