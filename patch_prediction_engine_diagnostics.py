import re

file_path = 'app_core/prediction_engine.py'
with open(file_path, 'r') as f:
    content = f.read()

# We already have diagnostic logging around line 849. Let's find it and ensure it strictly checks for 0.5.
# Look for: "Total rows entering model:"
