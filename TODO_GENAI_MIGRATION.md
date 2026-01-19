# TODO: Migrate from google.generativeai to google.genai

## Issue
The current implementation uses the deprecated `google.generativeai` package.
Google has officially deprecated this package and all support has ended.
It must be replaced with the new `google.genai` package.

## Affected Files
- `app_core/llm_assistant.py` (Likely the main integration point)
- `streamlit_app.py` (May contain direct calls or configuration)
- `requirements.txt` (Need to update dependency)

## Action Plan
1.  **Update Dependencies:**
    - Remove `google-generativeai` from `requirements.txt`.
    - Add `google-genai` to `requirements.txt`.

2.  **Refactor Code:**
    - Search for all imports of `google.generativeai`.
    - Replace with `google.genai` and update client initialization and method calls according to the new API documentation.
    - Check for deprecated model names if applicable.

3.  **Testing:**
    - Verify that LLM explanations and confidence checks still work correctly.
    - Check for any new authentication requirements.

## Priority
Medium. The current implementation may stop working at any time as the package is deprecated. Schedule for next sprint.
