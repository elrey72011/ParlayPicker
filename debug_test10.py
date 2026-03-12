import sys
sys.path.append('.')
from rapidfuzz import fuzz

concatenated_teams = "Los Angeles Lakers Boston Celtics"
combined_event_text = "Boston Celtics at Los Angeles Lakers: Spread LAL at BOS (Mar 10)"

score1 = fuzz.token_sort_ratio(concatenated_teams, combined_event_text)
score2 = fuzz.token_set_ratio(concatenated_teams, combined_event_text)
score3 = fuzz.partial_ratio(concatenated_teams, combined_event_text)
try:
    score4 = fuzz.partial_token_sort_ratio(concatenated_teams, combined_event_text)
except Exception:
    score4 = 0

print(f"Score: {max(score1, score2, score3, score4)}")
