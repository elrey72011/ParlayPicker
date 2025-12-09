"""
app_core.llm_assistant

Lightweight LLM assistant wrapper for ParlayPicker.

Provides:
    - analyze_kalshi_context_with_llm(context_markdown: str) -> List[dict]

This is used as a SECOND-OPINION reasoning engine (e.g., with Google Gemini),
NOT as a replacement for Vertex AI. If Gemini or the API key is unavailable,
the function simply returns an empty list.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI SETUP
# -------------------------------------------------------------------

try:
    import google.generativeai as genai  # type: ignore

    _GEMINI_IMPORTED = True
except Exception as e:
    genai = None  # type: ignore
    _GEMINI_IMPORTED = False
    logger.warning(f"google-generativeai not available: {e}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

_GEMINI_CONFIGURED = False

if _GEMINI_IMPORTED and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _GEMINI_CONFIGURED = True
        logger.info(f"LLM assistant using Gemini model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini: {e}")
        _GEMINI_CONFIGURED = False
elif _GEMINI_IMPORTED and not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not set. LLM assistant will be disabled "
        "until a valid key is provided."
    )


# -------------------------------------------------------------------
# Pydantic schema for contract recommendations
# -------------------------------------------------------------------

class ContractRecommendation(BaseModel):
    ticker: str
    side: str          # "yes"/"no" or "home"/"away"
    bid_price: int     # 0–100 (cents)
    reason: str
    confidence: int    # 0–100


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def analyze_kalshi_context_with_llm(context_markdown: str) -> List[Dict[str, Any]]:
    """
    Use an LLM (Gemini) to analyze a Kalshi-style context string and return
    a list of contract recommendations.

    Args:
        context_markdown:
            A markdown string describing a single game's context:
              - away @ home
              - moneyline odds
              - spreads
              - Kalshi implied probability, label, ticker, etc.
            This is built by VertexMasterAnalyzer._build_kalshi_context_for_llm().

    Returns:
        List[dict]: zero or more items with keys:
            - ticker: str
            - side: str ("yes"/"no" or "home"/"away")
            - bid_price: int
            - reason: str
            - confidence: int

        If Gemini is not available or anything fails, returns [].
    """
    if not _GEMINI_CONFIGURED:
        # Soft failure – caller treats "no contracts" as "assistant unavailable"
        logger.debug("LLM assistant disabled (Gemini not configured).")
        return []

    if not context_markdown or not context_markdown.strip():
        return []

    system_instructions = (
        "You are a prediction market assistant that evaluates current prices "
        "for event contracts on Kalshi.\n\n"
        "You will receive a description of a single game, including:\n"
        "- League and teams (e.g., NBA, NFL)\n"
        "- Game time\n"
        "- Sportsbook moneyline and spread odds\n"
        "- Kalshi market ticker, label, and implied probability\n\n"
        "Your job is to identify one or more underpriced contracts and explain why.\n\n"
        "CRITICAL OUTPUT REQUIREMENTS:\n"
        "1. Output JSON ONLY (no commentary, no markdown).\n"
        "2. Use this exact structure:\n"
        "{\n"
        '  "contracts": [\n'
        "    {\n"
        '      "ticker": "<Kalshi ticker or descriptive id>",\n'
        '      "side": "yes" or "no" or "home" or "away",\n'
        "      \"bid_price\": <int 0-100>,\n"
        "      \"reason\": \"short explanation\",\n"
        "      \"confidence\": <int 0-100>\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "3. 'confidence' must be an integer 0–100.\n"
        "4. If you see no clear edge, you may return an empty list for 'contracts'.\n"
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        resp = model.generate_content(
            [
                {"role": "system", "parts": [system_instructions]},
                {"role": "user", "parts": [context_markdown]},
            ]
        )

        raw_text = resp.text or ""
    except Exception as e:
        logger.warning(f"Gemini request failed in LLM assistant: {e}")
        return []

    # ------------------------------------------------------------------
    # Extract JSON from model response
    # ------------------------------------------------------------------
    json_str = raw_text.strip()

    # Handle ```json ... ``` / ``` ... ``` fenced blocks if present
    if "```" in json_str:
        try:
            if "```json" in json_str:
                json_block = json_str.split("```json", 1)[1].split("```", 1)[0]
            else:
                json_block = json_str.split("```", 1)[1].split("```", 1)[0]
            json_str = json_block.strip()
        except Exception:
            # Fall back to entire text
            json_str = raw_text.strip()

    contracts: List[Dict[str, Any]] = []

    # First attempt: expect {"contracts": [...]}
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            raw_contracts = parsed.get("contracts", [])
            for c in raw_contracts:
                try:
                    rec = ContractRecommendation(**c)
                    contracts.append(rec.dict())
                except ValidationError as ve:
                    logger.debug(f"Skipping invalid contract in assistant output: {ve}")
    except Exception:
        # Second attempt: maybe the top-level JSON is a list
        try:
            parsed_list = json.loads(json_str)
            if isinstance(parsed_list, list):
                for c in parsed_list:
                    try:
                        rec = ContractRecommendation(**c)
                        contracts.append(rec.dict())
                    except ValidationError as ve:
                        logger.debug(f"Skipping invalid contract in assistant output: {ve}")
        except Exception as e2:
            logger.warning(
                f"Failed to parse LLM assistant JSON response. "
                f"Error: {e2}. Raw text: {raw_text[:400]}"
            )
            return []

    return contracts
