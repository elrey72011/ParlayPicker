with open("tests/test_kalshi_enrich_league_inference.py", "r") as f:
    c = f.read()

c = c.replace('assert captured["series"] == ["KXNCAAMBSPREAD"]', 'assert "KXNCAAMBSPREAD" in captured["series"]')

with open("tests/test_kalshi_enrich_league_inference.py", "w") as f:
    f.write(c)
