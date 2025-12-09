# app_core/llm_assistant.py

import os
import json
from typing import List, Dict, Any

from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini (placeholder engine until Vertex endpoint is fixed)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Add it to your environment or Streamlit secrets."
    )

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL_NAME = "gemini-1.5-flash"  # cheap/fast placeholder


class ContractRecommendation(BaseModel):
    ticker: str
    side: str          # "yes" or "no"
    bid_price: int
    reason: str
    confidence: int    # 0–100


def analyze_kalshi_context_with_llm(context_markdown: str) -> List[Dict[str, Any]]:
    """
    Take a markdown string describing Kalshi markets (like assistant.py builds),
    call Gemini once, and return a list of contract-level recommendations.

    Returns a list of dicts with keys:
        ticker, side, bid_price, reason, confidence
    """
    system_instructions = (
        "You are a prediction market assistant that evaluates current prices "
        "for event contracts on Kalshi.\n\n"
        "For each contract in the provided context, identify whether the 'yes' "
        "or 'no' side is underpriced and explain why.\n\n"
        "Output JSON ONLY with this structure:\n\n"
        "{\n"
        '  "contracts": [\n'
        "    {\n"
        '      "ticker": "...",\n'
        '      "side": "yes" or "no",\n'
        "      \"bid_price\": <int>,\n"
        "      \"reason\": \"...\",\n"
        "      \"confidence\": <int>\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
    )

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    resp = model.generate_content(
        [
            {"role": "system", "parts": [system_instructions]},
            {"role": "user", "parts": [context_markdown]},
        ]
    )

    raw_text = resp.text or ""

    # Try to extract pure JSON in case it's wrapped in ```json ``` blocks
    json_str = raw_text
    if "```" in raw_text:
        try:
            if "```json" in raw_text:
                json_block = raw_text.split("```json", 1)[1].split("```", 1)[0]
            else:
                json_block = raw_text.split("```", 1)[1].split("```", 1)[0]
            json_str = json_block.strip()
        except Exception:
            json_str = raw_text.strip()

    contracts: List[Dict[str, Any]] = []

    try:
        parsed = json.loads(json_str)
        raw_contracts = parsed.get("contracts", [])
        for c in raw_contracts:
            rec = ContractRecommendation(**c)
            contracts.append(rec.dict())
    except Exception:
        # As a last resort, try interpreting JSON as a list of contracts
        try:
            lst = json.loads(json_str)
            if isinstance(lst, list):
                for c in lst:
                    rec = ContractRecommendation(**c)
                    contracts.append(rec.dict())
        except Exception:
            # Give up – return empty
            contracts = []

    return contracts
