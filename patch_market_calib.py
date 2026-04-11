import re

with open('tests/test_market_calibration.py', 'r') as f:
    content = f.read()

# We need to set calibrated_probability higher so it passes the new probability floors for Totals (0.56)
content = content.replace('"calibrated_probability": [0.55, 0.55, 0.55, 0.55],', '"calibrated_probability": [0.55, 0.59, 0.55, 0.59],')

with open('tests/test_market_calibration.py', 'w') as f:
    f.write(content)
