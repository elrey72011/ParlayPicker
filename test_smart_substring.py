from app_core.kalshi_integrator import _markets_by_team_text

def test_smart_substring_matching():
    # Test "Eastern Wa" matching with Kalshi's "Eastern Washington"
    markets = [{"title": "Eastern Washington vs Portland State", "subtitle": "Winner", "ticker": "1", "event_ticker": "1"}]
    res = _markets_by_team_text(markets, "Eastern Wa", "Portland St", "")
    assert len(res) == 1

    # Test "Miss Valley St" matching with Kalshi's "Mississippi Valley State"
    markets = [{"title": "Mississippi Valley State vs Jackson State", "subtitle": "Winner", "ticker": "2", "event_ticker": "2"}]
    res = _markets_by_team_text(markets, "Miss Valley St", "Jackson St", "")
    assert len(res) == 1

    # Test "Northern Co" matching with Kalshi's "Northern Colorado"
    markets = [{"title": "Northern Colorado vs Idaho State", "subtitle": "Winner", "ticker": "3", "event_ticker": "3"}]
    res = _markets_by_team_text(markets, "Northern Co", "Idaho State", "")
    assert len(res) == 1

    # Test 2-letter university prefix
    markets = [{"title": "NC State vs Duke", "subtitle": "Winner", "ticker": "4", "event_ticker": "4"}]
    res = _markets_by_team_text(markets, "NC State", "Duke", "")
    assert len(res) == 1

    print("All tests passed!")

if __name__ == "__main__":
    test_smart_substring_matching()