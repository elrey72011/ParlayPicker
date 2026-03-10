import asyncio
from core.streamlit_pipeline import run_analysis_pipeline

async def test_run():
    # Use mock CSV files or real data if available
    import pandas as pd
    try:
        # Load a small mock dataframe instead of full pipeline UI
        mock_data = pd.DataFrame({
            "game_id": ["TEST-GAME-1"],
            "league": ["NBA"],
            "home_team": ["LAL"],
            "away_team": ["BOS"],
            "game_date": ["2024-05-15"],
            "market_type": ["spread_home"],
            "spread_line": [-5.5]
        })

        from app_core.kalshi_integrator import enrich_with_kalshi_markets
        result_df = enrich_with_kalshi_markets(mock_data)

        print("Success! Enriched DF rows:", len(result_df))
        print("Columns:", list(result_df.columns))
    except Exception as e:
        print("Error during test run:", e)

if __name__ == "__main__":
    asyncio.run(test_run())
