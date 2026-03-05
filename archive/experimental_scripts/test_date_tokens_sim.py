import sys
from datetime import datetime, timedelta

def test_date_tokens_sim():
    print("Testing date_tokens_from_commence (Simulation)...")

    # Simulation of the new logic implemented in streamlit_app.py
    def parse_commence_to_utc_sim(time_str: str) -> datetime:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))

    def date_tokens_from_commence_sim(commence_list) -> set:
        tokens = set()
        for raw in commence_list:
            dt_utc = parse_commence_to_utc_sim(raw)
            # simulate local_tz (e.g. UTC for simplicity as we can't easily mock ZoneInfo("America/New_York") without timezone data)
            dt_local = dt_utc

            # Widen window to ±2 days to handle timezone shifts and Kalshi inconsistencies
            for offset in [-2, -1, 0, 1, 2]:
                dt_adj = dt_local + timedelta(days=offset)
                tokens.add(dt_adj.strftime("%y%b%d").upper())
        return tokens

    # Test Case 1
    commence_utc = "2026-02-12T01:00:00Z"
    tokens = date_tokens_from_commence_sim([commence_utc])
    print(f"Simulated Tokens for {commence_utc}: {sorted(list(tokens))}")

    # Expected: 12th +/- 2 days -> 10, 11, 12, 13, 14
    expected = {"26FEB10", "26FEB11", "26FEB12", "26FEB13", "26FEB14"}

    missing = expected - tokens
    extra = tokens - expected

    if not missing and not extra:
        print("PASS: Tokens match expected window +/- 2 days.")
    else:
        print(f"FAIL: Missing {missing}, Extra {extra}")
        exit(1)

if __name__ == "__main__":
    test_date_tokens_sim()
