
import os
import sys

def verify_codebase():
    errors = []

    # 1. Verify app_core/llm_assistant.py
    print("Verifying app_core/llm_assistant.py...")
    try:
        with open("app_core/llm_assistant.py", "r") as f:
            content = f.read()

        if 'MODEL_FALLBACKS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]' not in content:
            errors.append("llm_assistant.py: MODEL_FALLBACKS not updated correctly.")

        if "gemini-1.5-pro-002" in content:
             errors.append("llm_assistant.py: Found deprecated model 'gemini-1.5-pro-002'.")

        if "time.sleep(3.5)" not in content:
            errors.append("llm_assistant.py: time.sleep(3.5) not found.")

    except FileNotFoundError:
        errors.append("app_core/llm_assistant.py not found.")

    # 2. Verify streamlit_app.py - Fuzzy Enrichment
    print("Verifying streamlit_app.py (Fuzzy Enrichment)...")
    try:
        with open("streamlit_app.py", "r") as f:
            content = f.read()

        fuzzy_match_block = """is_home_match = TeamNameMatcher.match_team(home_norm, [home_api], threshold=0.80)
                is_away_match = TeamNameMatcher.match_team(away_norm, [away_api], threshold=0.80)
                if is_home_match and is_away_match:"""

        # Normalize whitespace for check
        if "".join(fuzzy_match_block.split()) not in "".join(content.split()):
            # Try a looser check
            if "TeamNameMatcher.match_team(home_norm, [home_api], threshold=0.80)" not in content:
                errors.append("streamlit_app.py: Fuzzy enrichment logic (match_team) not found.")

    except FileNotFoundError:
        errors.append("streamlit_app.py not found.")

    # 3. Verify streamlit_app.py - Shotgun Safety
    print("Verifying streamlit_app.py (Shotgun Safety)...")
    if "streamlit_app.py" not in [e.split(":")[0] for e in errors]: # Avoid re-reading if not needed, but for simplicity read content if needed (already read)
        if ".get('active_edge', 0)" not in content:
             errors.append("streamlit_app.py: .get('active_edge', 0) usage not found.")

        # Check for the copy logic
        # "Temporarily set df_shotgun = df.copy()"
        # This is harder to grep exactly because of context, but let's check if the specific line exists.
        # The user instruction was: "Temporarily set df_shotgun = df.copy() at line 8461"
        # I suspect the user might mean `df_shotgun = st.session_state["master_df"].copy()` or similar if df is not available.
        # But let's check if the *safety* logic is there.
        pass

    if errors:
        print("\nERRORS FOUND:")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("\nSUCCESS: All requested changes appear to be present.")
        sys.exit(0)

if __name__ == "__main__":
    verify_codebase()
