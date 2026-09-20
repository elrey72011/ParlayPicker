"""Current and saved run diagnostics, without changing decisions."""
import json

import pandas as pd
import streamlit as st

from core.run_readiness import build_readiness, game_table, render_readiness


def render_readiness_dashboard(audit=None, final=None, diagnostics=None):
    # Release legacy full raw-feed backup objects retained by older sessions.
    st.session_state.pop("mlb_receipt_store_downloads", None)
    with st.expander("Run Readiness Report", expanded=False):
        st.caption("Evidence readiness and wager approval are separate. This report does not change picks or thresholds.")
        if st.button("Prepare research performance report", key="research_performance_prepare"):
            try:
                from core.research_performance import rebuild
                from app_core.prediction_evidence import database_path
                st.session_state["research_performance_report"] = rebuild(database_path())
            except Exception:
                st.session_state.pop("research_performance_report", None)
                st.error("Saved research evidence could not be verified. No ranking or wager settings changed.")
        research = st.session_state.get("research_performance_report")
        if research is not None:
            st.caption(f"Research performance: {research['eligible_games']} eligible settled games. Descriptive only; ranking and wager validation remain separate. Uses locally restored evidence.")
            st.download_button("Download research performance report", json.dumps(research, indent=2),
                               "research-performance.json", "application/json", key="research_performance_download")
        if st.button("Prepare football evidence inventory", key="football_inventory_prepare"):
            st.session_state.pop("football_inventory", None)
            try:
                from core.football_inventory import rebuild
                from app_core.prediction_evidence import database_path
                st.session_state["football_inventory"] = rebuild(database_path())
            except Exception:
                st.error("Football evidence inventory failed verification. No records or wager settings changed.")
        inventory = st.session_state.get("football_inventory")
        if inventory is not None:
            st.caption("First saved pregame selections from locally restored evidence. Research counts do not authorize model training or wagers.")
            st.download_button("Download football evidence inventory", json.dumps(inventory, indent=2),
                               file_name="football-evidence-inventory.json", mime="application/json")
        source = st.selectbox("Readiness source", ["Current run", "Saved snapshot"], key="readiness_source")
        if source == "Saved snapshot":
            if st.button("Load saved snapshots", key="readiness_load"):
                from app_core.prediction_evidence import load_snapshots
                try:
                    st.session_state["readiness_snapshots"] = load_snapshots()
                except Exception:
                    st.error("Saved evidence could not be loaded. Check Prediction Evidence Status.")
                    st.session_state.pop("readiness_snapshots", None)
            saved = st.session_state.get("readiness_snapshots", [])
            if not saved:
                st.info("Load saved snapshots to inspect an earlier run.")
                return
            by_id = {sid: (a, f) for sid, a, f in saved}
            sid = st.selectbox("Snapshot", list(reversed(by_id)), key="readiness_snapshot")
            audit, final = by_id[sid]
            diagnostics = None  # Current run warnings must not describe an older run.
        receipt_health = (diagnostics or {}).get("mlb_receipt_health", {})
        if receipt_health:
            st.caption(f"MLB research receipts: {receipt_health.get('receipts_created', 0)} created; {receipt_health.get('receipts_skipped', 0)} skipped. No training or wager activation.")
            st.download_button("Download MLB Receipt Health", json.dumps(receipt_health, indent=2),
                               file_name="mlb-receipt-health.json", mime="application/json")
        if source == "Current run":
            if st.button("Catch up MLB receipt history", key="mlb_receipt_history_catchup"):
                from app_core.mlb_receipt_remote import catch_up_history
                try:
                    with st.spinner("Collecting up to 100 historical feeds and verifying Drive backup..."):
                        st.session_state["mlb_receipt_catchup_health"] = catch_up_history()
                    st.session_state.pop("mlb_receipt_audit_downloads_v2", None)
                except Exception as exc:
                    st.session_state["mlb_receipt_catchup_health"] = {"failed_stage": "history_catchup", "error_type": type(exc).__name__, "remote_backup_verified": False}
            catchup = st.session_state.get("mlb_receipt_catchup_health")
            if catchup:
                st.caption("Latest receipt-only catch-up; does not refresh displayed picks or run Gemini.")
                st.json(catchup)
                st.download_button("Download MLB catch-up health", json.dumps(catchup, indent=2), file_name="mlb-receipt-catchup-health.json", mime="application/json")
            with st.expander("Restore or sync MLB receipt backup"):
                uploaded = st.file_uploader("Original MLB receipt backup JSON", type=["json"], key="mlb_receipt_restore_file")
                if st.button("Restore receipt backup and verify Drive copy", disabled=uploaded is None, key="mlb_receipt_restore"):
                    from app_core.mlb_receipt_remote import restore, connection, recover, backup, restore_diagnostic
                    stage = "connect"
                    try:
                        client, folder = connection()
                        stage = "recover_drive"
                        recover(client)
                        stage = "decode_uploaded_backup"
                        bundle = json.loads(uploaded.getvalue())
                        stage = "restore_uploaded_backup"
                        restored = restore(bundle)
                        st.session_state.pop("mlb_receipt_audit_downloads_v2", None)
                        stage = "backup_and_verify_drive"
                        result = backup(client, folder)
                        st.success(f"Restored {restored} records. Drive backup read-back verified.")
                        st.session_state.pop("mlb_receipt_audit_downloads_v2", None)
                    except Exception as exc:
                        import logging
                        detail = restore_diagnostic(stage, exc)
                        logging.getLogger(__name__).error("MLB receipt restore failure: %s", detail)
                        st.error(f"Receipt operation failed at {stage}: {detail['reason']} ({detail['error_type']}). No conflicting records were overwritten. Remote durability is not confirmed.")
                        st.download_button("Download receipt restore error", json.dumps(detail, indent=2),
                                           file_name="mlb-receipt-restore-error.json", mime="application/json")
            if st.button("Prepare MLB receipt store audit and settled records", key="mlb_receipt_store_audit"):
                from app_core.mlb_receipt_audit import audit_downloads
                st.session_state.pop("mlb_receipt_audit_downloads_v2", None)
                try:
                    st.session_state["mlb_receipt_audit_downloads_v2"] = audit_downloads()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("MLB receipt audit preparation failed")
                    st.error("Receipt store audit failed. No records were changed; inspect the store before training.")
            downloads = st.session_state.get("mlb_receipt_audit_downloads_v2")
            if downloads:
                st.caption("Re-prepare after collection or reconciliation. These downloads contain the audit and settled receipts, not the full raw-feed backup. Drive backup status is reported separately in receipt health.")
                for title, filename, value in downloads:
                    st.download_button("Download " + title, value,
                                       file_name=filename, mime="application/json")
        if audit is None or audit.empty:
            st.info("No candidate evidence is available for this run. Run Game Analysis or select a saved snapshot.")
            return
        report = build_readiness(audit, final, diagnostics=diagnostics)
        football = report.get("football_coverage", {})
        if football.get("sports"):
            with st.expander("Football model and input coverage"):
                st.caption("TheOver is optional. Limited-evidence estimates are research selections; model validation remains required.")
                for sport, coverage in football["sports"].items():
                    st.write(f"{sport}: {coverage['games']} games")
                    st.dataframe(coverage["rows"], hide_index=True)
                st.download_button("Download football coverage", json.dumps(football, indent=2),
                                   file_name="football-coverage.json", mime="application/json")
        rejected = (diagnostics or {}).get("preselection_rejected_candidates", [])
        if rejected:
            st.write(f"Market candidates excluded before ranking: {len(rejected)}")
            st.caption("These rows did not reach the ranked candidate export. Reasons distinguish price, line, identity and main-market policy exclusions.")
            st.download_button("Download excluded market candidates", json.dumps(rejected, indent=2),
                               file_name="excluded-market-candidates.json", mime="application/json")
        counts = report["counts"]
        st.write(f"Games: {counts['games']} · Evidence ready for grading: {counts['ready_for_grading']} · Approved wagers: {counts['approved_wagers']}")
        st.caption(f"Quote age warning: {report['quote_warning_minutes']} minutes at capture, for diagnostics only. Feature freshness is unavailable without a source timestamp.")
        table = game_table(report)
        visible = ["league", "matchup", "selected_pick", "readiness", "wager_decision", "displayed_probability",
                   "production_probability", "independent_model_probability", "verified_quote_candidates", "candidate_count",
                   "evidence_blockers", "data_warnings", "wager_reasons"]
        st.dataframe(table[visible], hide_index=True)
        for warning in report["run_warnings"]:
            st.warning(warning)
        st.caption("Quote verified means the provider quote matched. Line eligible separately reflects final line rejection. Push-capable lines require verified probability semantics for validation.")
        candidates = pd.DataFrame(report["candidates"])
        candidates["issues"] = candidates["issues"].map(lambda values: "; ".join(values))
        st.dataframe(candidates[["matchup_id", "pick", "selected", "quote_verified", "line_eligible", "quoted_line", "settlement_rule", "probability_semantics", "quote_age_minutes_at_capture", "odds_source", "issues"]], hide_index=True)
        st.download_button("Download Readiness Report", render_readiness(report), file_name="run-readiness.md", mime="text/markdown")
        st.download_button("Download Readiness Metrics", json.dumps(report, indent=2, allow_nan=False), file_name="run-readiness.json", mime="application/json")
        st.download_button("Download Game Readiness CSV", table.to_csv(index=False), file_name="game-readiness.csv", mime="text/csv")
        st.download_button("Download Candidate Readiness CSV", candidates.to_csv(index=False), file_name="candidate-readiness.csv", mime="text/csv")
