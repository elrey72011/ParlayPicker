import re

with open('tests/test_actionable_tuning.py', 'r') as f:
    content = f.read()

# Add missing spread_line and total_line columns to bypass Missing Line check
content = content.replace('"matchup_id": "game1",', '"matchup_id": "game1",\n        "spread_line": -5,\n        "total_line": 220,')

with open('tests/test_actionable_tuning.py', 'w') as f:
    f.write(content)
