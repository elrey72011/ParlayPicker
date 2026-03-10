import re

with open("core/streamlit_pipeline.py", "r") as f:
    content = f.read()

# Instead of hardcoding -110 in _apply_analysis_calculations,
# keep the pd.NA logic or just use what we have, but modify tests?
# The problem is _apply_analysis_calculations has out["odds_american"] = _numeric_series(out, "odds_american", -110.0)
# which overwrites odds_american even if they exist if it isn't parsed correctly? No, it only defaults to -110 if missing.
