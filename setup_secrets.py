from config import THE_ODDS_API_KEY
import os

with open(".streamlit/secrets.toml", "w") as f:
    f.write(f'ODDS_API_KEY = "{THE_ODDS_API_KEY}"\n')
