import re

file_path = 'core/streamlit_pipeline.py'
with open(file_path, 'r') as f:
    content = f.read()

# Already contains:
# status_counts = best["Pick_Status"].value_counts().to_dict()
# logger.info("--- STATUS BRANCH VALIDATION ---")
# for status in ["Actionable", "Below Threshold", "Fallback / Low Confidence", "No Play", "Missing Line"]:
#     if status in status_counts:
#         logger.info(f"Validated branch: {status} ({status_counts[status]} rows)")
#     else:
#         logger.info(f"Branch not exercised: {status} (0 rows)")

print("Checking if NHL shorthand mapping success log is present in feature_processing.py")
