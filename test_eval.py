import pandas as pd
import numpy as np

# Simulate what happens if ev is NA
df = pd.DataFrame({"expected_value": [0.43, pd.NA, np.nan, -0.1]})
for idx in df.index:
    ev = df.at[idx, "expected_value"]
    print(f"ev: {ev}, ev > 0.35: {ev > 0.35 if not pd.isna(ev) else 'NA'}, isna: {pd.isna(ev)}")
