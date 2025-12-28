from app_core.team_name_matcher import TeamNameMatcher

def test_matching():
    home_norm = TeamNameMatcher.normalize("Army")
    home_api = "Army Black Knights"

    match = TeamNameMatcher.match_team(home_norm, [home_api], threshold=0.75)
    print(f"Match 'Army' vs 'Army Black Knights': {match}")
    assert match == "Army Black Knights"

    # Test "Oklahoma City" vs "Oklahoma City Thunder"
    match2 = TeamNameMatcher.match_team("oklahoma city", ["Oklahoma City Thunder"], threshold=0.75)
    print(f"Match 'OKC' vs 'OKC Thunder': {match2}")
    assert match2 == "Oklahoma City Thunder"

if __name__ == "__main__":
    try:
        test_matching()
        print("Verification PASSED")
    except Exception as e:
        print(f"Verification FAILED: {e}")
