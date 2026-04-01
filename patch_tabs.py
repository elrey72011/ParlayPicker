import re

with open('streamlit_app.py', 'r') as f:
    content = f.read()

# We want to move the tab definition up before the `if analysis_df is None or analysis_df.empty:` block.
tabs_def = '    tab1, tab2, tab3, tab_performance, tab4, tab5, tab6, tab7 = st.tabs(["Odds", "Analysis", "Best Picks", "Performance Recap", "Parlays", "Portfolio", "Debug", "Strategy Lab"])'

# find where tabs_def is currently
old_tabs_idx = content.find(tabs_def)

# Let's just replace the `if analysis_df is None or analysis_df.empty: return`
# with the tab definitions, and then wrap the metrics in a check.

replacement = """    tab1, tab2, tab3, tab_performance, tab4, tab5, tab6, tab7 = st.tabs(["Odds", "Analysis", "Best Picks", "Performance Recap", "Parlays", "Portfolio", "Debug", "Strategy Lab"])

    with tab_performance:
        from app_core.performance_pipeline import run_performance_pipeline
        from app.ui.results_dashboard import render_results_dashboard
        perf_df = run_performance_pipeline()
        render_results_dashboard(perf_df)

    if analysis_df is None or analysis_df.empty:
        st.info("Configure filters in the sidebar and click **Run Master Analysis**.")
        return
"""

content = content.replace("""    if analysis_df is None or analysis_df.empty:
        st.info("Configure filters in the sidebar and click **Run Master Analysis**.")
        return""", replacement)

# Now remove the old tab definitions and the old `with tab_performance:` block down below
old_perf_block = """    with tab_performance:
        from app_core.performance_pipeline import run_performance_pipeline
        from app.ui.results_dashboard import render_results_dashboard
        perf_df = run_performance_pipeline()
        render_results_dashboard(perf_df)"""

content = content.replace(tabs_def, "")
content = content.replace(old_perf_block, "")

with open('streamlit_app.py', 'w') as f:
    f.write(content)
