import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_apis")

import requests

try:
    resp = requests.get("https://api.collegefootballdata.com", timeout=5)
    logger.info("✅ CFBD API is reachable.")
except Exception as e:
    logger.error(f"❌ CFBD API not reachable. Error: {e}")

try:
    import nhlpy
    from nhlpy import NHLClient
    logger.info("✅ nhlpy loaded successfully.")
except ImportError as e:
    logger.error(f"❌ nhlpy NHLClient not installed. Error: {e}")

try:
    import cbbpy.mens_scraper as cbb_s
    logger.info("✅ cbbpy loaded successfully.")
except ImportError as e:
    logger.error(f"❌ CBBpy not installed. Error: {e}")
