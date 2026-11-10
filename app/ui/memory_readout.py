"""Memory panel: how much the app process is using and which Refresh stage grew it."""
import pandas as pd
import streamlit as st

from app_core import memory_readout


def render_memory_readout():
    with st.expander("Memory", expanded=False):
        found = memory_readout.readout()
        if found["rss_mb"] is None:
            st.caption("Memory cannot be read on this platform.")
            return
        peak = "" if found["peak_mb"] is None else f", peak {found['peak_mb']:,} MB"
        st.write(f"Using {found['rss_mb']:,} MB now{peak}. Streamlit Community Cloud restricts an app at about "
                 f"{found['limit_mb']:,} MB ({found['share_of_limit']:.0%} used).")
        if found["near_limit"]:
            st.warning("Memory is within 20% of the limit. Close other tabs of this app, and reboot it from "
                       "Streamlit's Manage app menu if it keeps climbing.")
        rise = found["largest_rise"]
        if rise and rise["rose_mb"] > 0:
            st.caption(f"Largest rise so far: {rise['rose_mb']:,} MB during \"{rise['stage']}\" ({rise['at']} UTC).")
        if found["notes"]:
            st.dataframe(pd.DataFrame(found["notes"]).reindex(columns=("at", "stage", "rss_mb", "rose_mb", "peak_mb")),
                         hide_index=True)
        else:
            st.caption("No Refresh has run in this process yet; stages are noted as a Refresh runs.")
        st.caption("Process-wide: every open tab shares this memory. It measures the process only and reads no data.")
