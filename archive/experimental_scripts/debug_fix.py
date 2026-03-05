import os

filepath = "streamlit_app.py"
with open(filepath, "r") as f:
    content = f.read()

spread_search_start = "spread_prob_final, spread_base_prob, spread_weights_used, spread_decision_driver, spread_warnings_new, spread_kalshi_prob_for_pick, spread_sentiment_debug = compute_final_probability("

idx = content.find(spread_search_start)
if idx != -1:
    print("Found start at:", idx)
    print("Next 500 chars:\n" + content[idx:idx+500])
else:
    print("Start not found")
