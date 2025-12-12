# redeploy: bump
"""
app_core.llm_assistant

Lightweight LLM assistant wrapper for ParlayPicker.

Provides:
    - analyze_kalshi_context_with_llm(context_markdown: str) -> List[dict]

This is used as a SECOND-OPINION reasoning engine (Gemini via Vertex AI),
NOT as a replacement for Vertex Master Analysis. If Gemini/Vertex isn't
available, the function returns an empty list.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI (VERTEX AI) SETUP
# -------------------------------------------------------------------

try:
    from vertexai.generative_models import GenerativeModel

    _GEMINI_AVAILABLE = True
except Exception as e:
    GenerativeModel = None  # type: ignore
    _GEMINI_AVAILABLE = False
    logger.warning(f"Vertex Gemini not available: {e}")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# NOTE:
# vertexai.init(project=..., location=...) should be called elsewhere in your app
# (you already do that in streamlit_app.py / gemini_integration.py).
# If it isn't, calls will fail and we’ll soft-return [].

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
    Use Gemini via Vertex AI to analyze a Kalshi-style context string and return
    contract recommendations.

    Returns [] if Gemini/Vertex is unavailable or any error occurs.
    """
    if not _GEMINI_AVAILABLE or GenerativeModel is None:
        logger.debug("LLM assistant disabled (Vertex Gemini not available).")
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
        '      "side": "yes" or "no" or "home" or "away",\n"
        '      "bid_price": <int 0-100>,\n'
        '      "reason": "short explanation",\n'
        '      "confidence": <int 0-100>\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "3. confidence must be an integer 0–100.\n"
        "4. If you see no clear edge, return an empty list for contracts.\n"
    )

    try:
        model = GenerativeModel(GEMINI_MODEL_NAME)

        resp = model.generate_content(
            [
                {"role": "system", "parts": [system_instructions]},
                {"role": "user", "parts": [context_markdown]},
            ]
        )

        raw_text = (getattr(resp, "text", None) or "").strip()
        if not raw_text:
            return []
    except Exception as e:
        logger.warning(f"Vertex Gemini request failed in LLM assistant: {e}")
        return []

    # ------------------------------------------------------------------
    # Extract JSON from model response
    # ------------------------------------------------------------------
    json_str = raw_text

    # Handle ```json ... ``` / ``` ... ``` fenced blocks if present
    if "```" in json_str:
        try:
            if "```json" in json_str:
                json_block = json_str.split("```json", 1)[1].split("```", 1)[0]
            else:
                json_block = json_str.split("```", 1)[1].split("```", 1)[0]
            json_str = json_block.strip()
        except Exception:
            json_str = raw_text

    contracts: List[Dict[str, Any]] = []

    # Attempt 1: expect {"contracts": [...]}
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            raw_contracts = parsed.get("contracts", [])
            for c in raw_contracts if isinstance(raw_contracts, list) else []:
                try:
                    rec = ContractRecommendation(**c)
                    contracts.append(rec.model_dump())
                except ValidationError as ve:
                    logger.debug(f"Skipping invalid contract in assistant output: {ve}")
            return contracts
    except Exception:
        pass

    # Attempt 2: maybe top-level JSON is a list
    try:
        parsed_list = json.loads(json_str)
        if isinstance(parsed_list, list):
            for c in parsed_list:
                try:
                    rec = ContractRecommendation(**c)
                    contracts.append(rec.model_dump())
                except ValidationError as ve:
                    logger.debug(f"Skipping invalid contract in assistant output: {ve}")
            return contracts
    except Exception as e2:
        logger.warning(
            "Failed to parse LLM assistant JSON response. "
            f"Error: {e2}. Raw text: {raw_text[:400]}"
        )
        return []

    return contracts
