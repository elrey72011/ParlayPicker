from datetime import date, timedelta

import pandas as pd
import streamlit as st
from app_core.results_ingestion import attach_results
from app_core.performance_pipeline import grade_picks_with_live_results
from app_core.candidate_recap import (
    grade_candidate_audit,
    merge_candidate_ledgers,
    summarize_candidate_performance,
)


def determine_display_outcome(row):
    # 1. Respect explicit manual dropdown overrides
    if 'Outcome' in row and pd.notna(row['Outcome']) and str(row['Outcome']) != 'N/A':
        return row['Outcome']

    pick = str(row.get('best_pick', '')).lower().strip()
    if not pick:
         return 'N/A'

    # Unresolved-line placeholders (e.g. "Total line unresolved") carry no
    # gradeable side/line — they must stay N/A, never WIN/LOSS. The 10 Jun
    # recap graded one as LOSS, polluting downstream calibration data.
    if 'unresolved' in pick:
         return 'N/A'

    # 2. Auto-grade based on scores if they exist in the grid
    h_score = row.get('actual_home_score')
    a_score = row.get('actual_away_score')

    if pd.notna(h_score) and pd.notna(a_score) and str(h_score).strip() != '' and str(a_score).strip() != '':
        try:
            h = float(h_score)
            a = float(a_score)

            # A 0-0 final does not exist in MLB/NBA/NHL — it means the game was
            # postponed or not played. Grading it WIN/LOSS poisons the
            # calibration data (6 Jun NYY/BOS and 11 Jun CWS/ATL were both
            # graded LOSS at 0-0). Void instead.
            if h == 0 and a == 0:
                return 'N/A'

            # Evaluate TOTALS (Over/Under)
            if 'over' in pick or 'under' in pick:
                import re
                m = re.search(r'(over|under)\s*(\d+\.?\d*)', pick, re.IGNORECASE)
                if m:
                    side = m.group(1).lower()
                    line = float(m.group(2))
                    total = h + a
                    if side == 'over':
                        return 'WIN' if total > line else ('LOSS' if total < line else 'PUSH')
                    elif side == 'under':
                        return 'WIN' if total < line else ('LOSS' if total > line else 'PUSH')

            # Evaluate SPREADS (+/-)
            elif '+' in pick or '-' in pick:
                import re
                from difflib import SequenceMatcher
                m = re.search(r'([+-]\d+\.?\d*)\s*(?:\(.*\))?$', pick)
                if m:
                    line = float(m.group(1))
                    pick_team = pick[:m.start()].strip()
                    # Safely get team names checking both 'home_team'/'away_team' and 'Home'/'Away'
                    home_team = str(row.get('home_team', row.get('Home', ''))).lower()
                    away_team = str(row.get('away_team', row.get('Away', ''))).lower()

                    home_ratio = SequenceMatcher(None, pick_team, home_team).ratio() if home_team else 0.0
                    away_ratio = SequenceMatcher(None, pick_team, away_team).ratio() if away_team else 0.0

                    is_home = (home_ratio > away_ratio) and home_ratio > 0.6
                    is_away = (away_ratio > home_ratio) and away_ratio > 0.6

                    if is_home:
                        margin = h - a
                        return 'WIN' if margin + line > 0 else ('LOSS' if margin + line < 0 else 'PUSH')
                    elif is_away:
                        margin = a - h
                        return 'WIN' if margin + line > 0 else ('LOSS' if margin + line < 0 else 'PUSH')

            # Evaluate MONEYLINE
            else:
                from difflib import SequenceMatcher
                pick_team = pick.replace(' ml', '').strip()
                # Safely get team names checking both 'home_team'/'away_team' and 'Home'/'Away'
                home_team = str(row.get('home_team', row.get('Home', ''))).lower()
                away_team = str(row.get('away_team', row.get('Away', ''))).lower()

                home_ratio = SequenceMatcher(None, pick_team, home_team).ratio() if home_team else 0.0
                away_ratio = SequenceMatcher(None, pick_team, away_team).ratio() if away_team else 0.0

                is_home = (home_ratio > away_ratio) and home_ratio > 0.6
                is_away = (away_ratio > home_ratio) and away_ratio > 0.6

                if is_home:
                    return 'WIN' if h > a else ('LOSS' if h < a else 'PUSH')
                elif is_away:
                    return 'WIN' if a > h else ('LOSS' if a < h else 'PUSH')

        except Exception:
            pass # If math fails, drop down to the fallback

    # 3. Fallback to ingestion logic (if no manual scores are entered yet)
    if 'Pick_Outcome' in row and pd.notna(row['Pick_Outcome']) and str(row['Pick_Outcome']) != 'N/A':
        return row['Pick_Outcome']

    if 'over' in pick or 'under' in pick:
         return row.get('total_result', 'N/A')
    elif '+' in pick or '-' in pick:
         return row.get('spread_result', 'N/A')
    else:
         return row.get('ml_result', 'N/A')


def funded_prop_rows(card: pd.DataFrame) -> pd.DataFrame:
    """Return only player props the production card actually funded."""

    if card is None or card.empty:
        return pd.DataFrame(columns=[] if card is None else card.columns)

    out = card.copy()
    if "Stake_Status" in out.columns:
        stake_status = out["Stake_Status"].fillna("").astype(str).str.strip()
        if stake_status.ne("").any():
            return out[stake_status.str.casefold().eq("funded")].copy()

    pick_status = out.get("Pick_Status", pd.Series("", index=out.index))
    status_mask = pick_status.fillna("").astype(str).str.strip().str.casefold().eq("actionable")
    stake = pd.to_numeric(out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    return out[status_mask & stake.gt(0)].copy()


def summarize_prop_results(results: pd.DataFrame) -> dict:
    """Summarize funded prop results without counting unresolved rows."""

    empty = {
        "wins": 0, "losses": 0, "pushes": 0, "unresolved": 0,
        "staked": 0.0, "pnl": 0.0, "roi": 0.0, "win_rate": 0.0,
    }
    if results is None or results.empty or "result" not in results.columns:
        return empty

    outcome = results["result"].fillna("").astype(str).str.upper()
    settled_mask = outcome.isin(["WIN", "LOSS", "PUSH"])
    settled = results[settled_mask]
    settled_outcome = outcome[settled_mask]
    stake = pd.to_numeric(settled.get("stake", pd.Series(0.0, index=settled.index)), errors="coerce").fillna(0.0)
    profit = pd.to_numeric(settled.get("profit", pd.Series(0.0, index=settled.index)), errors="coerce").fillna(0.0)

    wins = int(settled_outcome.eq("WIN").sum())
    losses = int(settled_outcome.eq("LOSS").sum())
    pushes = int(settled_outcome.eq("PUSH").sum())
    decisions = wins + losses
    staked = float(stake.sum())
    pnl = float(profit.sum())
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "unresolved": int((~settled_mask).sum()),
        "staked": staked,
        "pnl": pnl,
        "roi": (pnl / staked) if staked else 0.0,
        "win_rate": (wins / decisions) if decisions else 0.0,
    }


def _render_prop_results_recap() -> None:
    """Upload, grade, and recap the funded MLB player-prop card."""

    st.markdown("#### Player Prop Performance")
    uploaded = st.file_uploader(
        "Upload Yesterday's MLB Player Props Export",
        type=["csv"],
        key="perf_props_uploader",
    )
    if uploaded is None:
        st.caption("Upload the player-props export to grade only the bets marked Funded.")
        return

    file_identifier = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("perf_props_file_identifier") != file_identifier:
        st.session_state["perf_props_file_identifier"] = file_identifier
        st.session_state.pop("perf_props_results", None)

    try:
        uploaded.seek(0)
        card = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Error reading player-props export: {exc}")
        return

    funded = funded_prop_rows(card)
    st.caption(f"{len(funded)} funded prop(s) will be graded; unfunded research rows are excluded.")
    game_date = st.date_input(
        "Player-prop slate date",
        value=date.today() - timedelta(days=1),
        key="perf_props_date",
    )

    if st.button("Fetch MLB Player Prop Results", key="perf_props_fetch", disabled=funded.empty):
        with st.spinner("Fetching MLB player results and grading the funded card..."):
            try:
                from scripts.grade_props import (
                    _build_name_to_id,
                    fetch_actual_batter,
                    fetch_actual_ks,
                    grade_card,
                )

                date_text = game_date.isoformat()
                season = game_date.year
                name_to_id = _build_name_to_id(date_text, funded)
                rows = grade_card(
                    funded,
                    date_text,
                    name_to_id,
                    lambda player_id, participant_type: (
                        fetch_actual_batter(player_id, date_text, season)
                        if participant_type == "batter"
                        else fetch_actual_ks(player_id, date_text, season)
                    ),
                )
                st.session_state["perf_props_results"] = pd.DataFrame(rows)
            except Exception as exc:
                st.error(f"Error fetching or grading player props: {exc}")

    results = st.session_state.get("perf_props_results")
    if not isinstance(results, pd.DataFrame) or results.empty:
        if funded.empty:
            st.info("This export contains no funded player props.")
        return

    summary = summarize_prop_results(results)
    cols = st.columns(4)
    cols[0].metric("Funded Prop Record", f"{summary['wins']}-{summary['losses']}-{summary['pushes']}")
    cols[1].metric("Funded Prop Win Rate", f"{summary['win_rate']:.1%}")
    cols[2].metric("Funded Prop P&L", f"${summary['pnl']:+.2f}")
    cols[3].metric("Funded Prop ROI", f"{summary['roi']:+.1%}", f"${summary['staked']:.2f} staked")

    if summary["unresolved"]:
        st.warning(
            f"{summary['unresolved']} funded prop(s) could not be resolved from MLB StatsAPI "
            "and are excluded from the record and ROI."
        )

    if "market_type" in results.columns and "result" in results.columns:
        market_recap = pd.crosstab(
            results["market_type"].fillna("unknown"),
            results["result"].fillna("UNRESOLVED"),
        ).reset_index()
        st.markdown("##### Results by Prop Market")
        st.dataframe(market_recap, width="stretch", hide_index=True)

    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Download Player Prop Performance Recap",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="player_prop_performance_recap.csv",
        mime="text/csv",
        key="perf_props_download",
    )


def _production_wager_mask(frame: pd.DataFrame) -> pd.Series:
    """True only where the exported card assigned a positive production stake."""

    if frame is None or frame.empty:
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)

    stake_columns = [
        name
        for name in (
            "Play_Stake",
            "production_bet_amount",
            "Kelly_Bet_Size",
            "recommended_bet",
            "Suggested_Stake",
        )
        if name in frame.columns
    ]
    if not stake_columns:
        return pd.Series(False, index=frame.index, dtype=bool)

    amounts = pd.concat(
        [pd.to_numeric(frame[name], errors="coerce").fillna(0.0) for name in stake_columns],
        axis=1,
    )
    return amounts.max(axis=1).gt(0.0)


def _precision_card_mask(frame: pd.DataFrame) -> pd.Series:
    """True only for rows explicitly exported on the precision shortlist."""

    if frame is None or frame.empty or "Precision_Card" not in frame.columns:
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    values = frame["Precision_Card"]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").fillna("").str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes", "y"})


def _format_candidate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for column in ("Hit Rate", "Avg Probability", "Avg EV"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.1%}"
            )
    return out


def _render_candidate_results_recap(
    scored_picks: pd.DataFrame,
    uploaded_candidate_audit,
    uploaded_candidate_ledger,
) -> None:
    """Grade the full candidate set and expose cumulative rank/family evidence."""

    st.markdown("#### Candidate Ranking Backtest")
    if uploaded_candidate_audit is None:
        st.caption(
            "Upload yesterday's Candidate Selection Audit to grade every side and total, "
            "not only the one-row-per-game selection."
        )
        return

    try:
        uploaded_candidate_audit.seek(0)
        candidate_audit = pd.read_csv(uploaded_candidate_audit)
        current_graded = grade_candidate_audit(candidate_audit, scored_picks)
    except Exception as exc:
        st.error(f"Could not grade the candidate audit: {exc}")
        return

    prior_ledger = None
    if uploaded_candidate_ledger is not None:
        try:
            uploaded_candidate_ledger.seek(0)
            prior_ledger = pd.read_csv(uploaded_candidate_ledger)
        except Exception as exc:
            st.error(f"Could not read the prior candidate ledger: {exc}")
            return

    ledger = merge_candidate_ledgers(current_graded, prior_ledger)
    current_settled = int(current_graded.get("candidate_graded", pd.Series(dtype=bool)).sum())
    total_settled = int(ledger.get("candidate_graded", pd.Series(dtype=bool)).sum())
    current_keys = set(
        current_graded.get("candidate_ledger_key", pd.Series(dtype=str))
        .dropna()
        .astype(str)
    )
    prior_keys: set[str] = set()
    if prior_ledger is not None and not prior_ledger.empty:
        normalized_prior = merge_candidate_ledgers(pd.DataFrame(), prior_ledger)
        prior_keys = set(
            normalized_prior.get("candidate_ledger_key", pd.Series(dtype=str))
            .dropna()
            .astype(str)
        )
    prior_history_rows = len(prior_keys - current_keys)
    ledger_is_cumulative = prior_history_rows > 0

    if ledger_is_cumulative:
        st.caption(
            f"Graded {current_settled}/{len(current_graded)} candidates from this slate; "
            f"the cumulative ledger contains {total_settled}/{len(ledger)} settled candidates "
            f"including {prior_history_rows} prior unique rows."
        )
    else:
        st.warning(
            "No earlier unique candidate history was supplied. This download contains the "
            "current slate only and must not be treated as cumulative calibration evidence. "
            "Upload the previously downloaded candidate_results_ledger.csv on the next run."
        )
        st.caption(
            f"Graded {current_settled}/{len(current_graded)} current-slate candidates; "
            f"{total_settled}/{len(ledger)} rows are settled."
        )

    if total_settled:
        summaries = summarize_candidate_performance(ledger)
        left, right = st.columns(2)
        with left:
            st.markdown("##### By Overall Candidate Rank")
            st.dataframe(_format_candidate_summary(summaries["rank"]), width="stretch", hide_index=True)
        with right:
            st.markdown("##### By Market Family")
            st.dataframe(
                _format_candidate_summary(summaries["market_family"]),
                width="stretch",
                hide_index=True,
            )
        st.markdown("##### By Rank Within Each Market Family")
        st.dataframe(
            _format_candidate_summary(summaries["family_rank"]),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "These are diagnostics, not an automatic ranking rewrite. Change the selector only "
            "after the cumulative ledger shows a stable out-of-sample advantage."
        )
    else:
        st.info(
            "No candidate rows have final scores yet. Attach or edit the game scores, then "
            "the full candidate set will grade automatically."
        )

    dl_left, dl_right = st.columns(2)
    ledger_download_label = (
        "Download Updated Cumulative Candidate Results Ledger"
        if ledger_is_cumulative
        else "Download Current-Slate Candidate Results Ledger"
    )
    dl_left.download_button(
        "Download This Slate's Graded Candidates",
        data=current_graded.to_csv(index=False, encoding="utf-8-sig"),
        file_name="graded_candidate_audit.csv",
        mime="text/csv",
        key="download_current_candidate_grades",
    )
    dl_right.download_button(
        ledger_download_label,
        data=ledger.to_csv(index=False, encoding="utf-8-sig"),
        file_name="candidate_results_ledger.csv",
        mime="text/csv",
        key="download_candidate_results_ledger",
    )


def render_results_dashboard(picks_df: pd.DataFrame) -> None:
    # 1. File Uploader for Yesterday's Picks at the very top
    uploaded_picks_file = st.file_uploader("Upload Yesterday's Best Picks Export", type=["csv"], key="perf_picks_uploader")

    uploaded_candidate_audit = st.file_uploader(
        "Upload Yesterday's Candidate Selection Audit (optional)",
        type=["csv"],
        key="perf_candidate_audit_uploader",
    )
    uploaded_candidate_ledger = st.file_uploader(
        "Upload Prior Candidate Results Ledger (optional)",
        type=["csv"],
        key="perf_candidate_ledger_uploader",
    )

    st.subheader("Prior Day Performance")

    _render_prop_results_recap()
    st.divider()
    st.markdown("#### Game Performance")

    restricted_leagues = st.session_state.get("restricted_leagues", set())
    if restricted_leagues:
        league_str = ", ".join(sorted(list(restricted_leagues)))
        st.warning(f"Results for [{league_str}] are currently unavailable due to API plan restrictions. These picks must be verified manually.")

    unsupported_leagues = st.session_state.get("unsupported_result_leagues", set())
    if unsupported_leagues:
        league_str = ", ".join(sorted(str(league) for league in unsupported_leagues))
        st.error(
            f"No automatic results provider is configured for: {league_str}. "
            "Those rows remain ungraded and are excluded from the win rate."
        )

    # Re-evaluate the source data based on the explicit uploader first
    new_upload_detected = False
    if uploaded_picks_file is not None:
        file_identifier = f"{uploaded_picks_file.name}_{uploaded_picks_file.size}"
        if st.session_state.get("perf_picks_file_identifier") != file_identifier:
            new_upload_detected = True
            st.session_state["perf_picks_file_identifier"] = file_identifier

        try:
            # We must reset the file pointer to 0 because the file might have been read before
            uploaded_picks_file.seek(0)
            picks_df = pd.read_csv(uploaded_picks_file)

            # Map common pretty/export columns to canonical names
            if "League" in picks_df.columns and "league" not in picks_df.columns:
                picks_df["league"] = picks_df["League"]
            if "Home Team" in picks_df.columns and "home_team" not in picks_df.columns:
                picks_df["home_team"] = picks_df["Home Team"]
            elif "Home" in picks_df.columns and "home_team" not in picks_df.columns:
                picks_df["home_team"] = picks_df["Home"]
            if "Away Team" in picks_df.columns and "away_team" not in picks_df.columns:
                picks_df["away_team"] = picks_df["Away Team"]
            elif "Away" in picks_df.columns and "away_team" not in picks_df.columns:
                picks_df["away_team"] = picks_df["Away"]

            if "Pick Taken" in picks_df.columns and "best_pick" not in picks_df.columns:
                picks_df["best_pick"] = picks_df["Pick Taken"]
            elif "Best Pick" in picks_df.columns and "best_pick" not in picks_df.columns:
                picks_df["best_pick"] = picks_df["Best Pick"]

            st.success("Successfully loaded uploaded picks.")
        except Exception as e:
            st.error(f"Error reading uploaded picks file: {e}")
            picks_df = None

    if picks_df is None or picks_df.empty:
        st.info("No picks data available for the previous day. Please upload a file above.")
        return

    # Initialize editable picks in session state if not present, OR if a new file was uploaded
    if "perf_edited_picks" not in st.session_state or new_upload_detected:
        st.session_state["perf_edited_picks"] = picks_df.copy()

    # Get the latest dataframe state (could be edited previously)
    current_df = st.session_state["perf_edited_picks"]

    # 2. Toggle to fetch scores from API
    fetch_api = st.toggle("Fetch Scores from API", value=False, key="perf_api_toggle")

    # 3. File Uploader for Manual Results
    uploaded_manual_results = st.file_uploader("Upload Manual Results CSV (If API restricted)", type=["csv"], key="perf_manual_results_uploader")

    has_results = False

    if uploaded_manual_results is not None:
        try:
            uploaded_manual_results.seek(0)
            manual_results_df = pd.read_csv(uploaded_manual_results)
            current_df = attach_results(current_df, manual_results_df)
            st.session_state["perf_edited_picks"] = current_df
            st.success("Successfully applied manual results.")
            has_results = True
        except Exception as e:
            st.error(f"Error reading manual results file: {e}")
    elif fetch_api:
        with st.spinner("Fetching yesterday's results from API..."):
            leagues_in_picks = current_df['league'].unique().tolist() if 'league' in current_df.columns else []
            # Filter out restricted leagues
            restricted = st.session_state.get("restricted_leagues", set())
            allowed_leagues = [l for l in leagues_in_picks if l not in restricted]

            if allowed_leagues:
                try:
                    current_df = grade_picks_with_live_results(current_df)
                    st.session_state["perf_edited_picks"] = current_df
                    settled = int(
                        current_df.get("Pick_Outcome", pd.Series("N/A", index=current_df.index))
                        .fillna("N/A")
                        .astype(str)
                        .str.upper()
                        .isin(["WIN", "LOSS", "PUSH"])
                        .sum()
                    )
                    if settled:
                        st.success(
                            f"Fetched and applied API results: {settled}/{len(current_df)} rows settled."
                        )
                        has_results = True
                    if settled < len(current_df):
                        st.warning(
                            f"{len(current_df) - settled} row(s) remain ungraded after retry. "
                            "Use Refresh / Backfill after late games become final."
                        )
                except Exception as e:
                    st.error(f"Error fetching results from API: {e}")
            else:
                st.warning("No allowed leagues found in picks to fetch results for (or all are restricted).")

    # We consider it having results if either Pick_Outcome is present or actual_home_score is present
    if not has_results and ('Pick_Outcome' not in current_df.columns and 'actual_home_score' not in current_df.columns):
         st.warning("Results have not been attached to these picks yet. Please upload manual results, fetch from API, or edit manually below.")

    # Process all rows for the display table first so we have the Outcome logic for the top metrics
    display_df = current_df.copy()

    # Ensure actual scores exist as columns for the editor
    for col in ['actual_home_score', 'actual_away_score', 'Outcome']:
         if col not in display_df.columns:
              display_df[col] = pd.NA

    display_df['Outcome'] = display_df.apply(determine_display_outcome, axis=1)

    unresolved_mask = ~display_df["Outcome"].fillna("N/A").astype(str).str.upper().isin(
        ["WIN", "LOSS", "PUSH"]
    )
    if unresolved_mask.any():
        unresolved_league_col = next(
            (column for column in ("league", "League") if column in display_df.columns),
            None,
        )
        unresolved_leagues = (
            sorted(
                display_df.loc[unresolved_mask, unresolved_league_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if unresolved_league_col
            else []
        )
        league_detail = f" ({', '.join(unresolved_leagues)})" if unresolved_leagues else ""
        st.warning(
            f"Grading is incomplete: {int(unresolved_mask.sum())} row(s){league_detail} are N/A. "
            "They are excluded from every win-rate denominator below."
        )

    def calculate_metrics(df):
        wins = len(df[df['Outcome'] == 'WIN'])
        losses = len(df[df['Outcome'] == 'LOSS'])
        pushes = len(df[df['Outcome'] == 'PUSH'])

        total_decisions = wins + losses
        win_rate = (wins / total_decisions) if total_decisions > 0 else 0.0

        # Calculate Net Profit based on odds_american assuming 1 unit bet
        net_profit = 0.0
        for idx, row in df.iterrows():
            outcome = row['Outcome']
            if outcome == 'WIN':
                 odds = pd.to_numeric(row.get('odds_american'), errors='coerce')
                 if pd.isna(odds):
                      odds = -110 # default

                 if abs(odds) < 1.0:
                      odds = -110

                 # Calculate profit for 1 unit (e.g. $100 bet)
                 if odds < 0:
                      profit = 1.0 / (abs(odds) / 100.0)
                 else:
                      profit = odds / 100.0
                 net_profit += profit
            elif outcome == 'LOSS':
                 net_profit -= 1.0

        return wins, losses, pushes, win_rate, net_profit

    production_mask = _production_wager_mask(display_df)
    production_df = display_df[production_mask].copy()

    st.markdown("#### Production-Approved Wager Performance")
    if production_df.empty:
        st.info(
            "No rows carried a positive app-approved stake on this slate. That means the "
            "production record is 0 wagers; the full-board results below are diagnostic coverage."
        )
    else:
        wins_prod, losses_prod, pushes_prod, win_rate_prod, net_profit_prod = calculate_metrics(production_df)
        settled_prod = wins_prod + losses_prod + pushes_prod
        prod_cols = st.columns(4)
        prod_cols[0].metric("Settled Record", f"{wins_prod}-{losses_prod}-{pushes_prod}")
        prod_cols[1].metric("Production Win Rate", f"{win_rate_prod:.1%}")
        prod_cols[2].metric("Flat-Bet P&L (Units)", f"{net_profit_prod:+.2f}")
        prod_cols[3].metric("Rows Settled", f"{settled_prod}/{len(production_df)}")

    precision_df = display_df[_precision_card_mask(display_df)].copy()
    st.markdown("#### Precision Shortlist Performance")
    if precision_df.empty:
        st.info(
            "This recap contains no precision-shortlist rows. Export the current top-two "
            "precision card to begin tracking the accuracy-first pilot."
        )
    else:
        (
            wins_precision,
            losses_precision,
            pushes_precision,
            win_rate_precision,
            net_profit_precision,
        ) = calculate_metrics(precision_df)
        settled_precision = wins_precision + losses_precision + pushes_precision
        precision_cols = st.columns(4)
        precision_cols[0].metric(
            "Settled Record",
            f"{wins_precision}-{losses_precision}-{pushes_precision}",
        )
        precision_cols[1].metric("Precision Win Rate", f"{win_rate_precision:.1%}")
        precision_cols[2].metric(
            "Flat-Bet P&L (Units)", f"{net_profit_precision:+.2f}"
        )
        precision_cols[3].metric(
            "Rows Settled", f"{settled_precision}/{len(precision_df)}"
        )
    st.caption(
        "The precision shortlist is an accuracy-first pilot with a 75% monitoring target, "
        "not a guaranteed result or automatic wager. Bettable plus a positive exported "
        "stake remains the only wagering authority."
    )

    wins_all, losses_all, pushes_all, win_rate_all, net_profit_all = calculate_metrics(display_df)
    settled_all = wins_all + losses_all + pushes_all
    st.markdown("#### Coverage Board Performance (Diagnostic)")
    _all_cols = st.columns(4)
    _all_cols[0].metric("Settled Record", f"{wins_all}-{losses_all}-{pushes_all}")
    _all_cols[1].metric("Coverage Win Rate", f"{win_rate_all:.1%}")
    _all_cols[2].metric("Hypothetical Flat-Bet P&L (Units)", f"{net_profit_all:+.2f}")
    _all_cols[3].metric("Rows Settled", f"{settled_all}/{len(display_df)}")
    st.caption(
        "This grades the best available direction for every game, including PASS rows. "
        "It is coverage analysis—not approval to wager every row. N/A, postponed, started, "
        "and unresolved-line rows do not count as settled decisions."
    )

    from app_core.strategy_lab_realized import summarize_recap_tiers
    _tier_src = display_df.rename(columns={"Pick_Status": "Status"}) if "Pick_Status" in display_df.columns else display_df
    tiers = summarize_recap_tiers(_tier_src)
    st.markdown("#### Performance by Model Tier (Diagnostic)")
    _tcols = st.columns(3)
    for _col, (_, _row) in zip(_tcols, tiers.iterrows()):
        _col.metric(
            _row["Tier"],
            f"{_row['Hit Rate']:.1%}",
            f"{int(_row['Wins'])}-{int(_row['Losses'])} ({int(_row['Total'])})",
        )
    st.caption(
        "Tier results help diagnose calibration and selection quality. A tier label alone "
        "does not prove a wager was approved; the positive exported stake is authoritative."
    )

    _render_candidate_results_recap(
        display_df,
        uploaded_candidate_audit,
        uploaded_candidate_ledger,
    )

    st.divider()

    # Ensure team columns map correctly if the export used capitalized 'Home'/'Away'
    if 'Home' in display_df.columns and 'home_team' not in display_df.columns:
        display_df = display_df.rename(columns={'Home': 'home_team'})
    if 'Away' in display_df.columns and 'away_team' not in display_df.columns:
        display_df = display_df.rename(columns={'Away': 'away_team'})
    
    # Select columns to show. export_run_id is carried through so a downloaded
    # recap stays traceable to the pipeline run that produced the card —
    # scripts/grade_slate.py warns when a recap's run id doesn't match the
    # export being graded (11 Jun: a recap built from a stale morning card
    # graded lines the final card never played).
    cols_to_show = [
        'league', 'home_team', 'away_team', 'best_pick',
        'actual_home_score', 'actual_away_score', 'Outcome',
        'grading_status', 'grading_issue', 'Precision_Card', 'Precision_Rank',
        'Precision_Card_Instruction', 'Pick_Status', 'export_run_id',
    ]
    # Filter to only existing columns
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]

    # Rename columns for presentation
    rename_map = {
         'league': 'League',
         'home_team': 'Home',
         'away_team': 'Away',
         'best_pick': 'Pick Taken',
         'grading_status': 'Grading Status',
         'grading_issue': 'Grading Issue',
         'Precision_Card': 'Precision Shortlist',
         'Precision_Rank': 'Precision Rank',
         'Precision_Card_Instruction': 'Precision Instruction',
         'Pick_Status': 'Status'
    }

    table_df = display_df[cols_to_show].rename(columns=rename_map)

    # Render st.data_editor
    edited_df = st.data_editor(
        table_df,
        width="stretch",
        disabled=[
            "League", "Home", "Away", "Pick Taken", "Grading Status",
            "Grading Issue", "Precision Shortlist", "Precision Rank",
            "Precision Instruction", "Status", "export_run_id",
        ],
        column_config={
            "actual_home_score": st.column_config.NumberColumn(
                "Home Score",
                min_value=0,
                step=1,
                format="%d",
                required=False
            ),
            "actual_away_score": st.column_config.NumberColumn(
                "Away Score",
                min_value=0,
                step=1,
                format="%d",
                required=False
            ),
            "Outcome": st.column_config.SelectboxColumn(
                "Outcome",
                options=["WIN", "LOSS", "PUSH", "N/A"],
                required=True
            )
        },
        key="perf_data_editor"
    )

    # Check for changes and update session state
    if not edited_df.equals(table_df):
        # We need to map edited columns back to our session state current_df
        # Create mapping of presentation names to original names
        reverse_rename_map = {v: k for k, v in rename_map.items()}
        for col in edited_df.columns:
            orig_col = reverse_rename_map.get(col, col)
            current_df[orig_col] = edited_df[col]

        # Save back to session state
        st.session_state["perf_edited_picks"] = current_df

        # Trigger a rerun so metrics update immediately
        st.rerun()
