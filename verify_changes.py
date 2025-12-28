import sys
import logging
from app_core.llm_assistant import MODEL_FALLBACKS

# Verify Model Fallbacks
expected_fallbacks = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
if MODEL_FALLBACKS == expected_fallbacks:
    print("SUCCESS: MODEL_FALLBACKS updated correctly.")
else:
    print(f"FAILURE: MODEL_FALLBACKS mismatch. Found: {MODEL_FALLBACKS}")
    sys.exit(1)

# Verify List Location in streamlit_app.py
with open("streamlit_app.py", "r") as f:
    lines = f.readlines()

tab_master_line = -1
required_cols_line = -1
export_cols_line = -1

for i, line in enumerate(lines):
    if "with tab_master:" in line:
        tab_master_line = i
    if "required_display_cols =" in line and "Moved required_display_cols" not in line: # Avoid comment
        # We need to make sure we find the definition inside the tab block, not the commented out old one
        # The old one at the top was commented out by me, or should have been deleted?
        # Actually, in my diff, I deleted the global one and added it inside tab_master.
        # But wait, my diff failed. I applied it manually?
        # Ah, the diff tool said "Invalid merge diff" but then "Edit applied successfully" in the second attempt?
        # Let's just find the active definition.
        if i > tab_master_line and tab_master_line != -1:
             required_cols_line = i
             break

if tab_master_line != -1 and required_cols_line != -1 and required_cols_line > tab_master_line:
    print(f"SUCCESS: required_display_cols found inside tab_master block (Line {required_cols_line+1}).")
else:
    print(f"FAILURE: required_display_cols not found in correct location. tab_master at {tab_master_line+1}, list at {required_cols_line+1}")
    # Don't fail the script yet, let's debug manually if this fails
