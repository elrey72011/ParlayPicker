import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app_core.odds_api import TheOddsAPIClient

def test_odds_api_pagination(monkeypatch):
    client = TheOddsAPIClient(api_key="test_key")

    call_count = 0
    def mock_get(url, params, timeout):
        nonlocal call_count
        call_count += 1

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        if call_count == 1:
            # First page
            mock_resp.json.return_value = [{"id": "game1", "bookmakers": [{"key": "novig"}]}]
            mock_resp.headers = {"x-next-page": "token1"}
        elif call_count == 2:
            # Second page
            mock_resp.json.return_value = [{"id": "game2", "bookmakers": [{"key": "novig"}]}]
            mock_resp.headers = {"x-next-page": "token2"}
        else:
            # Last page
            mock_resp.json.return_value = [{"id": "game3", "bookmakers": [{"key": "novig"}]}]
            mock_resp.headers = {}

        return mock_resp

    with patch('requests.get', side_effect=mock_get):
        result = client.get_odds("basketball_ncaab")

    assert len(result) == 3
    assert result[0]["id"] == "game1"
    assert result[1]["id"] == "game2"
    assert result[2]["id"] == "game3"
