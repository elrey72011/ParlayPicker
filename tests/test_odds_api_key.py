import core.streamlit_pipeline as pipeline


class _MissingSecrets:
    def get(self, *_args, **_kwargs):
        raise FileNotFoundError("no secrets.toml")


def test_odds_api_key_prefers_environment_without_touching_streamlit(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "  env-key  ")
    monkeypatch.setattr(pipeline.st, "secrets", _MissingSecrets())

    assert pipeline._get_odds_api_key() == "env-key"


def test_odds_api_key_missing_streamlit_secrets_fails_closed(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(pipeline.st, "secrets", _MissingSecrets())

    assert pipeline._get_odds_api_key() == ""


def test_odds_api_key_uses_streamlit_secret_as_fallback(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(pipeline.st, "secrets", {"ODDS_API_KEY": "  secret-key  "})

    assert pipeline._get_odds_api_key() == "secret-key"
