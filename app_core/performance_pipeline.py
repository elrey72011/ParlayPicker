import pandas as pd
import logging
from app_core.export_utils import find_yesterdays_export
from app_core.results_fetcher import fetch_yesterdays_results
from app_core.results_ingestion import attach_results

logger = logging.getLogger(__name__)

_no_export_logged = False


def _league_column(frame: pd.DataFrame) -> str | None:
    return next((column for column in ("league", "League") if column in frame.columns), None)


def _settled_mask(frame: pd.DataFrame) -> pd.Series:
    outcome = frame.get("Pick_Outcome", pd.Series("N/A", index=frame.index))
    return outcome.fillna("N/A").astype(str).str.upper().isin(["WIN", "LOSS", "PUSH"])


def _annotate_grading_status(
    frame: pd.DataFrame,
    unsupported_leagues: set[str],
) -> pd.DataFrame:
    """Make incomplete grading explicit and machine-readable per exported row."""

    out = frame.copy()
    league_col = _league_column(out)
    leagues = (
        out[league_col].fillna("").astype(str).str.upper().str.strip()
        if league_col
        else pd.Series("", index=out.index)
    )
    settled = _settled_mask(out)
    scores_attached = (
        pd.to_numeric(
            out.get("actual_home_score", pd.Series(pd.NA, index=out.index)),
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            out.get("actual_away_score", pd.Series(pd.NA, index=out.index)),
            errors="coerce",
        ).notna()
    )
    unsupported = leagues.isin(unsupported_leagues)

    out["grading_status"] = "PENDING"
    out["grading_issue"] = "final_score_not_available_or_team_unmatched"
    out.loc[scores_attached & ~settled, "grading_status"] = "UNRESOLVED PICK"
    out.loc[scores_attached & ~settled, "grading_issue"] = "score_attached_pick_unresolved"
    out.loc[unsupported, "grading_status"] = "UNSUPPORTED LEAGUE"
    out.loc[unsupported, "grading_issue"] = "no_results_provider_configured"
    out.loc[settled, "grading_status"] = "GRADED"
    out.loc[settled, "grading_issue"] = ""
    return out


def grade_picks_with_live_results(picks_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch, attach, and backfill final scores for every league in a mixed slate."""

    if picks_df is None or picks_df.empty:
        return picks_df

    league_col = _league_column(picks_df)
    if league_col is None:
        logger.error("Cannot grade picks without a league/League column.")
        return _annotate_grading_status(picks_df, set())

    leagues = picks_df[league_col].dropna().astype(str).unique().tolist()
    results_df = fetch_yesterdays_results(leagues, attempts=2)
    unsupported = set(results_df.attrs.get("unsupported_leagues", []))
    enriched = attach_results(picks_df, results_df)

    unresolved = ~_settled_mask(enriched)
    unresolved_leagues = (
        enriched.loc[unresolved, league_col]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
        .tolist()
    )
    retry_leagues = [league for league in unresolved_leagues if league not in unsupported]

    # One targeted backfill pass catches transient provider failures and mixed
    # slates where one league was unavailable during the first request.
    if retry_leagues:
        retry_df = fetch_yesterdays_results(retry_leagues, attempts=1)
        unsupported.update(retry_df.attrs.get("unsupported_leagues", []))
        if not retry_df.empty:
            combined = pd.concat([results_df, retry_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["league", "home_team", "away_team", "date"],
                keep="last",
            )
            enriched = attach_results(picks_df, combined)

    enriched = _annotate_grading_status(enriched, unsupported)
    settled_count = int(_settled_mask(enriched).sum())
    unresolved_count = int(len(enriched) - settled_count)
    enriched.attrs["requested_leagues"] = [str(league).upper() for league in leagues]
    enriched.attrs["unsupported_leagues"] = sorted(unsupported)
    enriched.attrs["settled_count"] = settled_count
    enriched.attrs["unresolved_count"] = unresolved_count

    try:
        import streamlit as st

        st.session_state["unsupported_result_leagues"] = set(unsupported)
    except Exception:
        pass

    if unsupported:
        logger.warning("Grading skipped unsupported leagues: %s", sorted(unsupported))
    if unresolved_count:
        logger.warning(
            "Performance grading remains incomplete after backfill: %s/%s rows unresolved.",
            unresolved_count,
            len(enriched),
        )
    try:
        from app_core.prediction_evidence import record_scores, write_validation_reports
        revisions = record_scores(enriched.loc[_settled_mask(enriched)])
        enriched.attrs["prediction_score_revisions_saved"] = revisions
        if revisions:
            enriched.attrs["prediction_validation_reports"] = write_validation_reports()
    except Exception as exc:
        enriched.attrs["prediction_evidence_error"] = str(exc)
        logger.warning("Final scores could not be saved to prediction evidence: %s", exc)
    return enriched

def run_performance_pipeline() -> pd.DataFrame | None:
    """
    Executes the pipeline to fetch yesterday's export, fetch final scores
    for the leagues in that export, and attach the scores to calculate outcomes.
    Returns the enriched DataFrame or None if no export was found.
    """
    global _no_export_logged
    export_path = find_yesterdays_export()
    # Saved decisions work without a manual CSV download and preserve run identity.
    try:
        from app_core.prediction_evidence import load_snapshots
        yesterday = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        saved = []
        for _, _, frame in load_snapshots():
            start = pd.to_datetime(frame["game_start_utc"], errors="coerce", utc=True)
            saved.append(frame[start.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").eq(yesterday)])
        saved = pd.concat(saved, ignore_index=True) if saved else pd.DataFrame()
        if not saved.empty:
            return grade_picks_with_live_results(saved)
    except Exception as exc:
        logger.warning("Could not load saved prediction decisions: %s", exc)
    if not export_path:
         if not _no_export_logged:
             logger.info("No best picks export found for yesterday (only logging once per run).")
             _no_export_logged = True
         return None

    try:
         picks_df = pd.read_csv(export_path)
    except Exception as e:
         logger.error(f"Failed to read {export_path}: {e}")
         return None

    if picks_df.empty:
         return picks_df

    return grade_picks_with_live_results(picks_df)
