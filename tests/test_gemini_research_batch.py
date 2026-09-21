from types import SimpleNamespace

from app_core.gemini_research_batch import (
    build_inline_requests,
    submit_research_batch,
)


def test_research_batch_is_chunked_cost_controlled_and_non_live(monkeypatch):
    monkeypatch.delenv("PARLAYPICKER_GEMINI_RESEARCH_MODEL", raising=False)
    rows = [
        {
            "game_id": f"g{index}",
            "side_a": {"best_pick": "Under 8.5"},
            "verified_context": {},
            "missing_context": [
                "probable_pitchers", "lineups", "injuries", "weather"
            ],
        }
        for index in range(13)
    ]
    requests = build_inline_requests(rows)
    assert len(requests) == 2
    assert [request["config"]["response_json_schema"]["maxItems"] for request in requests] == [12, 1]
    assert all(request["model"] == "gemini-2.5-flash-lite" for request in requests)
    assert all(request["config"]["thinking_config"] == {"thinking_budget": 0} for request in requests)
    assert all(request["config"]["max_output_tokens"] == 8192 for request in requests)
    required = requests[0]["config"]["response_json_schema"]["items"]["required"]
    assert "supporting_evidence" in required
    assert "missing_information" in required

    captured = {}

    class Batches:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(name="batches/one", state="JOB_STATE_PENDING")

    job = submit_research_batch(
        rows,
        display_name="research-only",
        client=SimpleNamespace(batches=Batches()),
    )
    assert job.name == "batches/one"
    assert captured["model"] == "gemini-2.5-flash-lite"
    assert captured["config"]["display_name"] == "research-only"
    assert len(captured["src"]) == 2
