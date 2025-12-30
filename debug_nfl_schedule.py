import nfl_data_py as nfl
import logging

logging.basicConfig(level=logging.INFO)

try:
    print("Fetching NFL schedule for 2023...")
    df = nfl.import_schedules([2023])
    print("Columns:", df.columns)
    print("Head:", df.head())
    print(f"Rows: {len(df)}")
except Exception as e:
    print(f"Error: {e}")
