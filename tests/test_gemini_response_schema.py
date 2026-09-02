from __future__ import annotations

import json
from types import SimpleNamespace

from app_core import llm_assistant


class _FakeModels:
    def __init__(self) -> None:
        self.config = None
        self.contents = ""

    def generate_content(self, *, model, contents, config):
        self.config = config
        self.contents = contents
        payload = [{
            "game_id": "g1",
            "recommended_bet": "Home +1.5",
            "confidence": "MEDIUM",
            "explanation": "Positive exact-price expected value supports the side.",
            "risk_notes": "Normal market variance.",
            "flags": [],
        }]
        return SimpleNamespace(text=json.dumps(payload))


def test_batch_call_enforces_complete_json_schema_and_exact_price_rule(monkeypatch):
    models = _FakeModels()
    client = SimpleNamespace(models=models)
    monkeypatch.setattr(llm_assistant, "_GEMINI_AVAILABLE", True)
    monkeypatch.setattr(llm_assistant, "initialize_gemini", lambda: (client, None))
    monkeypatch.setattr(llm_assistant.time, "sleep", lambda _seconds: None)

    result = llm_assistant.generate_batch_confidence_explanation([{
        "game_id": "g1",
        "is_player_prop": True,
        "side_a": {
            "best_pick": "Home +1.5",
            "odds_american": -110,
            "expected_value": 0.05,
            "edge": 0.03,
        },
        "side_b": None,
    }])

    assert result["g1"]["risk_notes"] == "Normal market variance."
    assert models.config.response_mime_type == "application/json"
    schema = models.config.response_json_schema
    assert set(schema["items"]["required"]) == {
        "game_id",
        "recommended_bet",
        "confidence",
        "explanation",
        "risk_notes",
        "flags",
    }
    assert schema["items"]["additionalProperties"] is False
    assert "expected_value is the authoritative value at the offered odds" in models.contents
    assert "flags array (use [] when there are no flags)" in models.contents
