"""
app_core.llm_assistant

Second-opinion LLM assistant wrapper for ParlayPicker (Gemini via Vertex AI).
Never blocks the app: returns [] if unavailable or any failure.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ---- Vertex Gemini availability ----
_GEMINI_AVAILABLE = False
GenerativeModel = None  # type: ignore

try:
    from vertexai.generative_models import GenerativeModel as _GenerativeModel  # type: ignore

    GenerativeModel = _GenerativeModel  # type: ignore
    _GEMINI_AVAILABLE = True
except Exception as e:
    logger.warning(f"Vertex Gemini not available: {e}")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")


class ContractRecommendation(BaseModel):
    ticker: str
    side: str
    bid_price: int
    reason: str
    confidence: int


def analyze_kalshi_context_with_llm(context_markdown: str) -> List[Dict[str, Any]]:
    if not _GEMINI_AVAILABLE or GenerativeModel is None:
        return []
    if not context_markdown or not context_markdown.strip():
        return []

    system_instructions = (
        "Output JSON ONLY. Schema:\n"
        '{ "contracts": [ { "ticker": "...", "side": "yes|no|home|away", '
        '"bid_price": 0, "reason": "...", "confidence": 0 } ] }\n'
        'If no edge: {"contracts": []}'
    )

    try:
        model = GenerativeModel(GEMINI_MODEL_NAME)
        resp = model.generate_content([system_instructions, context_markdown])
        raw_text = (getattr(resp, "text", "") or "").strip()
        if not raw_text:
            return []
    except Exception as e:
        logger.warning(f"Gemini request failed: {e}")
        return []

    # Strip fenced blocks if present
    json_str = raw_text
    if "```" in json_str:
        try:
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
            else:
                json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception:
            json_str = raw_text.strip()

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        logger.warning(f"Non-JSON assistant output: {e}. Raw: {raw_text[:300]}")
        return []

    raw_contracts = []
    if isinstance(parsed, dict):
        raw_contracts = parsed.get("contracts", [])
    elif isinstance(parsed, list):
        raw_contracts = parsed

    if not isinstance(raw_contracts, list):
        return []

    out: List[Dict[str, Any]] = []
    for c in raw_contracts:
        if not isinstance(c, dict):
            continue
        try:
            rec = ContractRecommendation(**c)
            out.append(rec.model_dump() if hasattr(rec, "model_dump") else rec.dict())
        except ValidationError:
            continue

    return out
