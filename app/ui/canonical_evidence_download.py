"""Owner-only canonical download; called after the existing token check."""
import streamlit as st
from app_core import canonical_download


def render(setting):
    with st.expander("Private canonical evidence download", expanded=False):
        st.caption("Prepare a private snapshot of retained canonical research evidence. No analysis refresh, storage restore or source request is needed. This download supplies no wagering authority.")
        path = canonical_download.effective_path()
        binding = str(path.resolve())
        key = "private_canonical_download"
        previous = st.session_state.get(key)
        if previous and previous["binding"] != binding:
            st.session_state.pop(key, None)
        if st.button("Prepare canonical evidence download", key="canonical_download_prepare"):
            st.session_state.pop(key, None)
            try:
                # Detect configured owner/provider credentials without serializing
                # any configuration or session state into the artifact.
                names = ("PARLAYPICKER_PUBLISH_TOKEN", "ODDS_API_KEY", "CFBD_API_KEY",
                         "API_SPORTS_KEY", "GEMINI_API_KEY", "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT",
                         "PARLAYPICKER_NETLIFY_TOKEN", "PARLAYPICKER_SFTP_PASSWORD")
                forbidden = tuple(setting(n) for n in names)
                # Streamlit can promote root secrets to the environment lazily.
                # Resolve the same producer path after that configuration load.
                path = canonical_download.effective_path()
                binding = str(path.resolve())
                raw, manifest = canonical_download.build_download(path, forbidden_values=forbidden)
                st.session_state[key] = dict(binding=binding, raw=raw, manifest=manifest)
            except canonical_download.DownloadUnavailable as exc:
                st.error("Canonical evidence download unavailable: "+str(exc)+". Hosted remote contents remain unknown; no store was initialized or restored.")
        prepared = st.session_state.get(key)
        if prepared:
            manifest = prepared["manifest"]
            st.caption("Snapshot prepared at "+manifest["prepared_at"]+". Later commits are not part of this snapshot.")
            st.download_button("Download private canonical evidence ZIP", prepared["raw"],
                "private-prospective-evidence-"+manifest["snapshot"]["sha256"]+".zip",
                "application/zip", key="canonical_download_zip", on_click="ignore")
