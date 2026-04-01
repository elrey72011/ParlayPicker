import re
with open('streamlit_app.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "if analysis_df is None or analysis_df.empty:" in line:
        new_lines.extend([
            '    tab1, tab2, tab3, tab_performance, tab4, tab5, tab6, tab7 = st.tabs(["Odds", "Analysis", "Best Picks", "Performance Recap", "Parlays", "Portfolio", "Debug", "Strategy Lab"])\n',
            '\n',
            '    with tab_performance:\n',
            '        from app_core.performance_pipeline import run_performance_pipeline\n',
            '        from app.ui.results_dashboard import render_results_dashboard\n',
            '        perf_df = run_performance_pipeline()\n',
            '        render_results_dashboard(perf_df)\n',
            '\n',
            '    if analysis_df is None or analysis_df.empty:\n'
        ])
    elif "st.info(\"Configure filters in the sidebar and click **Run Master Analysis**.\")" in line and 'tab1' in ''.join(new_lines[-10:]):
        new_lines.append(line)
    elif "return" in line and 'tab1' in ''.join(new_lines[-10:]):
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('streamlit_app.py', 'w') as f:
    f.writelines(new_lines)
