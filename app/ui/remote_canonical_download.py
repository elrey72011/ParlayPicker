"""Existing-owner-gated remote object download, distinct from local SQLite."""
import hashlib
import streamlit as st
from app_core import remote_canonical_download as remote


def render(setting):
    with st.expander("Private remote canonical JSON download", expanded=False):
        st.caption("Read existing canonical JSON objects from the configured Shared Drive. This is separate from the local SQLite snapshot. No store is created or restored; no analysis or provider request is run.")
        # Resolve root secrets before binding. No configuration enters the ZIP.
        folder = str(setting("PARLAYPICKER_DRIVE_FOLDER_ID")).strip()
        account = str(setting("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT"))
        owner = str(setting("PARLAYPICKER_PUBLISH_TOKEN"))
        binding = hashlib.sha256((folder+"\0"+account+"\0"+owner).encode()).hexdigest()
        key = "private_remote_canonical_download"
        previous = st.session_state.get(key)
        if previous and previous["binding"] != binding:
            st.session_state.pop(key, None)
        st.caption("Limits: 100 listing pages, 100,000 metadata entries, 50,000 canonical paths, 128 MiB of media and 5 minutes. Capped or interrupted downloads are explicitly partial. A paginated listing is not an atomic remote snapshot.")
        if st.button("Prepare remote canonical JSON download", key="remote_canonical_prepare"):
            st.session_state.pop(key, None)
            names = ("PARLAYPICKER_PUBLISH_TOKEN", "ODDS_API_KEY", "CFBD_API_KEY",
                     "API_SPORTS_KEY", "GEMINI_API_KEY", "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT",
                     "PARLAYPICKER_NETLIFY_TOKEN", "PARLAYPICKER_SFTP_PASSWORD")
            forbidden = tuple(setting(name) for name in names)
            try:
                # Also invalidate when lazy configuration loading changes scope.
                folder = str(setting("PARLAYPICKER_DRIVE_FOLDER_ID")).strip()
                account = str(setting("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT"))
                owner = str(setting("PARLAYPICKER_PUBLISH_TOKEN"))
                binding = hashlib.sha256((folder+"\0"+account+"\0"+owner).encode()).hexdigest()
                with st.spinner("Reading existing remote canonical objects within export limits…"):
                    raw, manifest = remote.build_download(folder, forbidden_values=forbidden)
                st.session_state[key] = dict(binding=binding, raw=raw, manifest=manifest)
            except remote.ExportUnavailable as exc:
                st.error("Remote canonical JSON download unavailable: "+str(exc)+". No store was initialized or restored.")
        prepared = st.session_state.get(key)
        if prepared:
            manifest = prepared["manifest"]
            if not manifest["export_complete"]:
                st.warning("PARTIAL remote export: "+str(manifest["inventory"]["incomplete_reason"] or manifest["stop_reason"])+". Unlisted or unread objects and their dependencies remain UNKNOWN.")
            counts = manifest["counts"]
            st.caption(str(counts["exported_paths"])+" verified canonical paths; "+
                       str(counts["exported_remote_files"])+" remote file identities. Counts do not establish an authentic complete research chain or wagering authority.")
            st.download_button("Download private remote canonical JSON ZIP", prepared["raw"],
                "private-remote-canonical-"+hashlib.sha256(prepared["raw"]).hexdigest()+".zip",
                "application/zip", key="remote_canonical_zip", on_click="ignore")
