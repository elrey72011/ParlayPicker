# ParlayPicker App Issues - 2026-01-21 Exports Analysis

## Summary
- Timestamp: 2026-01-21 20:38-20:39 UTC run (67 games: NBA/NCAAB/NHL).
- Files: logs (file:94), master_df_raw (file:95, 315 cols), picks CSV (file:93, 67 rows), debug JSON (file:96), TheOver (file:97).
- Generated: parlaypicker_all_picks_20260121_2038.csv, master_df_raw (46).csv.[file:93][file:95]

## Critical Issues
1. **Model Placeholders (High Priority)**: 0.623 output triggers fallbacks ("Model file may be corrupted"). Affects ~20+ NCAAB games. See logs: ERRORappcore.predictionengineCRITICAL.[file:94]
2. **DataFrame Fragmentation**: PerformanceWarning in streamlit_app.py:11606 during merges. Slows exports.[file:94]

## Recommendations
- Use Jules prompt above for code fixes.
- Validate/retrain XGBoost models per league.
- Monitor NCAAB stats matching (fallbacks for MISSING teams).[file:94]

## Evidence Excerpts
- Logs: "Single prediction is placeholder value 0.623034656047821!" (multiple).[file:94]
- Raw DF: "warnings Fallback Placeholder Detected" in decision traces.[file:95]
