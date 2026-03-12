import json
import logging
import os
import re
from typing import Optional
from pydantic import BaseModel, Field
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


logger = logging.getLogger(__name__)

# Constants for File Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
UNMATCHED_QUEUE_FILE = os.path.join(DATA_DIR, "unmatched_queue.json")
DYNAMIC_ALIASES_FILE = os.path.join(DATA_DIR, "dynamic_aliases.json")

# Pydantic Schema for strict JSON output from Gemini
class ResolvedEntities(BaseModel):
    mapped_home_team: Optional[str] = Field(description="The matching Kalshi event team representing the home team, or null if no match.")
    mapped_away_team: Optional[str] = Field(description="The matching Kalshi event team representing the away team, or null if no match.")

def _initialize_files():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(UNMATCHED_QUEUE_FILE):
        with open(UNMATCHED_QUEUE_FILE, "w") as f:
            json.dump([], f)
    if not os.path.exists(DYNAMIC_ALIASES_FILE):
        with open(DYNAMIC_ALIASES_FILE, "w") as f:
            json.dump({}, f)

def load_queue() -> list[dict]:
    try:
        with open(UNMATCHED_QUEUE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load queue: {e}")
        return []

def load_aliases() -> dict:
    try:
        with open(DYNAMIC_ALIASES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load aliases: {e}")
        return {}

def save_queue(queue: list[dict]):
    with open(UNMATCHED_QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

def save_aliases(aliases: dict):
    with open(DYNAMIC_ALIASES_FILE, "w") as f:
        json.dump(aliases, f, indent=2)

def get_gemini_client():
    if not GENAI_AVAILABLE:
        logger.error("google-genai is not installed.")
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables.")
        return None

    return genai.Client(api_key=api_key)

def extract_json_from_response(text: str) -> dict:
    """Helper to safely extract JSON if the model returns markdown codeblocks."""
    text = text.strip()
    # Try to find JSON block
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {text}")
        return {}


def resolve_queue():
    """
    Tier 5: Semantic Inference and Self-Healing Dictionary Feedback.
    Reads the unmatched_queue.json, uses Gemini to map them against Kalshi events,
    and updates dynamic_aliases.json.
    """
    _initialize_files()
    queue = load_queue()
    if not queue:
        logger.info("No unmatched records in queue.")
        return

    client = get_gemini_client()
    if not client:
        return

    aliases = load_aliases()
    processed_indices = []

    for i, item in enumerate(queue):
        home_team = item.get("home_team")
        away_team = item.get("away_team")
        league = item.get("league", "Unknown")
        candidates = item.get("candidates", [])

        if not home_team or not away_team or not candidates:
            processed_indices.append(i)
            continue

        candidate_strs = [f"Title: '{c.get('title')}', Subtitle: '{c.get('subtitle')}'" for c in candidates]
        candidates_text = "\n".join(candidate_strs)

        prompt = f"""
You are an expert sports data analyst. Map the following unstandardized prediction market event names to the exact standard home_team and away_team.

Standard Teams:
Home Team: {home_team}
Away Team: {away_team}
League: {league}

Candidate Kalshi Prediction Market Events (Choose the best match or return null):
{candidates_text}

Return ONLY a valid JSON object with the keys 'mapped_home_team' and 'mapped_away_team'.
The values should be the team names extracted from the matching Kalshi event.
If no logical match exists, return null for both.
"""

        try:
            logger.info(f"Resolving {away_team} @ {home_team} via Gemini...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResolvedEntities,
                    temperature=0.0
                )
            )

            result = extract_json_from_response(response.text)
            mapped_home = result.get('mapped_home_team')
            mapped_away = result.get('mapped_away_team')

            if mapped_home:
                aliases[home_team] = mapped_home
                logger.info(f"Mapped {home_team} -> {mapped_home}")
            if mapped_away:
                aliases[away_team] = mapped_away
                logger.info(f"Mapped {away_team} -> {mapped_away}")

            processed_indices.append(i)

        except APIError as e:
            logger.error(f"Gemini API Error for {home_team} vs {away_team}: {e}")
        except Exception as e:
            logger.error(f"Error processing {home_team} vs {away_team}: {e}")

    # Remove processed items from queue
    queue = [item for i, item in enumerate(queue) if i not in processed_indices]

    save_aliases(aliases)
    save_queue(queue)
    logger.info("Semantic inference complete. Self-healing dictionary updated.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resolve_queue()
