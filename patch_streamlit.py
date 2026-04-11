import re

with open('streamlit_app.py', 'r') as f:
    content = f.read()

old_cal_diag = """
                st.markdown("#### New Calibration Diagnostics")
                cal_col1, cal_col2 = st.columns(2)

                cal_col1.write("**Actionable by League:**")
                league_counts = diagnostics.get("actionable_counts_by_league", {})
                for l, count in league_counts.items():
                    cal_col1.write(f"- {l}: {count}")

                cal_col1.write(f"**NHL Totals Actionable:** {diagnostics.get('nhl_totals_actionable', 0)}")
                cal_col1.write(f"**Totals Below Prob Floor:** {diagnostics.get('actionable_totals_below_floor', 0)}")

                cal_col2.write("**Spread Divergence Override:**")
                cal_col2.write(f"- Downgraded: {diagnostics.get('spreads_downgraded_by_divergence', 0)}")
                cal_col2.write(f"- Rescued: {diagnostics.get('spreads_rescued_by_divergence', 0)}")
"""

new_cal_diag = """
                st.markdown("#### New Calibration Diagnostics")
                cal_col1, cal_col2 = st.columns(2)

                cal_col1.write("**Actionable by League:**")
                league_counts = diagnostics.get("actionable_counts_by_league", {})
                for l, count in league_counts.items():
                    cal_col1.write(f"- {l}: {count}")

                cal_col1.write(f"**NHL Totals Actionable:** {diagnostics.get('nhl_totals_actionable', 0)}")
                cal_col1.write(f"**Totals Below Prob Floor:** {diagnostics.get('actionable_totals_below_floor', 0)}")

                cal_col2.write("**Spread Divergence Override:**")
                cal_col2.write(f"- Downgraded: {diagnostics.get('spreads_downgraded_by_divergence', 0)}")
                cal_col2.write(f"- Rescued: {diagnostics.get('spreads_rescued_by_divergence', 0)}")

                st.markdown("#### Consensus & Side Floor Diagnostics")
                con_col1, con_col2 = st.columns(2)

                con_col1.write("**Actionable by Consensus:**")
                consensus_counts = diagnostics.get("actionable_counts_by_consensus", {})
                for c_type, count in consensus_counts.items():
                    con_col1.write(f"- {c_type}: {count}")
                con_col1.write(f"**Final Actionable Count:** {diagnostics.get('final_actionable_count', 0)}")

                con_col2.write("**Downgraded Rows:**")
                con_col2.write(f"- Neutral Thresholds: {diagnostics.get('neutral_downgrades', 0)}")
                con_col2.write(f"- Disagrees Thresholds: {diagnostics.get('disagrees_downgrades', 0)}")
                con_col2.write(f"- Side Prob Floor: {diagnostics.get('side_prob_floor_downgrades', 0)}")
"""
content = content.replace(old_cal_diag, new_cal_diag)

with open('streamlit_app.py', 'w') as f:
    f.write(content)
