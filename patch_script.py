import re

with open('app_core/kalshi_integrator.py', 'r') as f:
    content = f.read()

# 1. Update tolerances
content = re.sub(r'KALSHI_LINE_TOLERANCE_SPREAD = 2.5', 'KALSHI_LINE_TOLERANCE_SPREAD = 0.5', content)
content = re.sub(r'KALSHI_LINE_TOLERANCE_TOTAL = 20.0', 'KALSHI_LINE_TOLERANCE_TOTAL = 0.5', content)

# 2. Remove kalshi_tried_tickers from enrich_with_kalshi_markets
content = re.sub(r'        "kalshi_tried_tickers",\n', '', content)
content = re.sub(r'    out\["kalshi_match_status"\] = "miss"', '    out["kalshi_match_status"] = "miss"', content)
content = re.sub(r'        if col not in out\.columns:\n            out\[col\] = pd\.NA', '        if col not in out.columns:\n            out[col] = pd.NA', content)

# Remove the usages of kalshi_tried_tickers
content = re.sub(r'            out\.at\[idx, "kalshi_tried_tickers"\] = json\.dumps\(candidates\)\n', '', content)

with open('app_core/kalshi_integrator.py', 'w') as f:
    f.write(content)
