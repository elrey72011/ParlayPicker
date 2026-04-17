import pandas as pd
from app_core.results_ingestion import attach_results

def test_attach_results_pretty_exports():
    master_df = pd.DataFrame([
        {
            "League": "NBA",
            "Home Team": "Lakers",
            "Away Team": "Bulls",
            "Pick Taken": "Lakers -5.5",
            "Commence (UTC)": "2023-10-25T00:00:00Z"
        }
    ])

    # Ensure pipeline normalized columns map over as well
    master_df['league'] = master_df['League']
    master_df['home_team'] = master_df['Home Team']
    master_df['away_team'] = master_df['Away Team']
    master_df['best_pick'] = master_df['Pick Taken']

    results_df = pd.DataFrame([
        {
            "League": "NBA",
            "Home": "Lakers",
            "Away": "Bulls",
            "HomeScore": 110,
            "AwayScore": 100,
            "Date": "2023-10-25T00:00:00Z"
        }
    ])

    df = attach_results(master_df, results_df)
    assert float(df["actual_home_score"].iloc[0]) == 110.0
    assert df["Pick_Outcome"].iloc[0] == "WIN"
