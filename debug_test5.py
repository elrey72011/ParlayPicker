import sys
sys.path.append('.')
from rapidfuzz import fuzz

concatenated_teams = "Los Angeles Lakers Boston Celtics"
combined_event_text = "Boston at Los Angeles L: Spread LAL at BOS (Mar 10)"
score1 = fuzz.token_sort_ratio(concatenated_teams, combined_event_text)
score2 = fuzz.token_set_ratio(concatenated_teams, combined_event_text)

print(f"Score: {max(score1, score2)}")

concatenated_teams = "Pepperdine Saint Mary's"
combined_event_text = "Saint Mary's vs Pepperdine total over 140.5 PEP at SMC (Mar 10)"
score1 = fuzz.token_sort_ratio(concatenated_teams, combined_event_text)
score2 = fuzz.token_set_ratio(concatenated_teams, combined_event_text)

print(f"Score: {max(score1, score2)}")
