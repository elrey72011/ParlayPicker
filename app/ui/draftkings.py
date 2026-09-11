"""DraftKings salary-file builders, independent of player-prop analysis."""
import pandas as pd
import streamlit as st


def render_draftkings_builders(publication_dfs):
    st.subheader("DraftKings Lineup Builder")
    st.caption("Upload a Classic salary CSV below. Running Player Props is not required.")
    with st.expander("DraftKings NFL Classic — Top 5 Complete Lineups"):
        st.caption(
            "Upload the official player-pool CSV from the exact DraftKings Classic contest. "
            "The optimizer builds five complete nine-player lineups under the $50,000 cap: "
            "QB, two RBs, three WRs, TE, RB/WR/TE FLEX, and DST. Every lineup includes a "
            "same-team WR/TE with its quarterback and excludes offensive players facing the "
            "selected DST. Uploaded projections are preferred; DraftKings average fantasy "
            "points per game is used only as a clearly labeled fallback."
        )
        _dk_nfl_exclude_questionable = st.checkbox(
            "Exclude NFL players marked Questionable or Doubtful",
            value=True,
            key="draftkings_nfl_exclude_questionable",
        )
        _dk_nfl_min_changes = st.slider(
            "Minimum player changes between NFL lineups",
            min_value=1,
            max_value=4,
            value=2,
            key="draftkings_nfl_min_lineup_changes",
        )
        _dk_classic_upload = st.file_uploader(
            "Upload DraftKings NFL Classic salary CSV",
            type=["csv"],
            key="draftkings_nfl_classic_salary_csv",
        )
        _dk_nfl_projection_upload = st.file_uploader(
            "Optional: upload NFL forward projections CSV",
            type=["csv"],
            key="draftkings_nfl_classic_projection_csv",
            help="Include player ID or Name plus a Projected Points column.",
        )
        if _dk_classic_upload is not None:
            try:
                from app_core.draftkings_classic import (
                    attach_draftkings_projections,
                    build_draftkings_classic_lineups,
                    export_draftkings_classic_position_csv,
                    parse_draftkings_classic_salary_csv,
                )

                _dk_classic_pool = parse_draftkings_classic_salary_csv(
                    _dk_classic_upload,
                    exclude_questionable=_dk_nfl_exclude_questionable,
                )
                if _dk_nfl_projection_upload is not None:
                    _dk_classic_pool = attach_draftkings_projections(
                        _dk_classic_pool,
                        _dk_nfl_projection_upload,
                    )
                    st.success(
                        "Forward projections matched "
                        f"{_dk_classic_pool.attrs.get('projection_match_count', 0)} "
                        "eligible NFL players."
                    )
                st.caption(
                    f"Loaded {_dk_classic_upload.name}: "
                    f"{len(_dk_classic_pool)} eligible players across "
                    f"{_dk_classic_pool['Game Info'].nunique()} games."
                )
                _dk_nfl_projection_matches = int(
                    _dk_classic_pool["ProjectionSource"]
                    .eq("uploaded_projection")
                    .sum()
                )
                if _dk_nfl_projection_matches == 0:
                    st.warning(
                        "No forward-looking Projected Points column was found. These "
                        "lineups are research-only rankings based on historical DraftKings "
                        "average fantasy points per game."
                    )
                elif _dk_nfl_projection_matches < len(_dk_classic_pool):
                    st.warning(
                        f"Forward projections matched {_dk_nfl_projection_matches} of "
                        f"{len(_dk_classic_pool)} eligible players. Unmatched players "
                        "still use historical DraftKings average fantasy points."
                    )
                _dk_classic_lineups = build_draftkings_classic_lineups(
                    _dk_classic_pool,
                    top_n=5,
                    min_unique_players_between_lineups=_dk_nfl_min_changes,
                )
                if _dk_classic_lineups.empty:
                    st.warning(
                        "No valid NFL Classic lineup could be built from this file. Confirm "
                        "the contest includes every required position, team/game data, "
                        "projections or FPPG, and a feasible $50,000 roster."
                    )
                else:
                    _dk_nfl_lineup_export = _dk_classic_lineups.drop(
                        columns=["Lineup Key"],
                        errors="ignore",
                    )
                    publication_dfs["NFL"] = _dk_nfl_lineup_export.copy()
                    st.dataframe(
                        _dk_nfl_lineup_export,
                        width="stretch",
                        hide_index=True,
                    )
                    st.download_button(
                        "Export NFL Lineup Review + Diagnostics",
                        _dk_nfl_lineup_export.to_csv(
                            index=False,
                            encoding="utf-8-sig",
                        ),
                        "draftkings_nfl_classic_top5_lineups.csv",
                        mime="text/csv",
                        key="download_draftkings_nfl_classic_top5_lineups",
                    )
                    st.download_button(
                        "Export NFL Position-Only CSV",
                        export_draftkings_classic_position_csv(
                            _dk_classic_lineups,
                            sport="NFL",
                        ),
                        "draftkings_nfl_classic_position_lineups.csv",
                        mime="text/csv",
                        key="download_draftkings_nfl_position_lineups",
                        help=(
                            "Contains only DraftKings Name + ID values under the Classic "
                            "position columns. Add contest-entry metadata from the exact "
                            "DraftKings template when required."
                        ),
                    )
            except Exception as exc:
                st.error(f"Unable to read the DraftKings salary CSV: {exc}")
    with st.expander("DraftKings MLB Classic — Top 5 Complete Lineups"):
        st.caption(
            "Upload the official player-pool CSV from the exact DraftKings MLB Classic "
            "contest. The optimizer builds five complete 10-player lineups under the "
            "$50,000 cap: two pitchers, C, 1B, 2B, 3B, SS, and three outfielders. "
            "Multi-position eligibility is honored, hitters are capped at five per team, "
            "and hitters facing a selected pitcher are excluded. Uploaded projections are "
            "preferred; DraftKings average fantasy points per game is used only as a "
            "clearly labeled fallback."
        )
        _dk_mlb_require_confirmed = st.checkbox(
            "Require every MLB hitter to have a confirmed batting-order position",
            value=True,
            key="draftkings_mlb_require_confirmed_hitters",
        )
        _dk_mlb_min_changes = st.slider(
            "Minimum player changes between MLB lineups",
            min_value=1,
            max_value=4,
            value=2,
            key="draftkings_mlb_min_lineup_changes",
        )
        _dk_mlb_classic_upload = st.file_uploader(
            "Upload DraftKings MLB Classic salary CSV",
            type=["csv"],
            key="draftkings_mlb_classic_salary_csv",
        )
        _dk_mlb_projection_upload = st.file_uploader(
            "Optional: upload MLB forward projections CSV",
            type=["csv"],
            key="draftkings_mlb_classic_projection_csv",
            help="Include player ID or Name plus a Projected Points column.",
        )
        if _dk_mlb_classic_upload is not None:
            try:
                from app_core.draftkings_classic import (
                    attach_draftkings_projections,
                    build_draftkings_mlb_classic_lineups,
                    export_draftkings_classic_position_csv,
                    parse_draftkings_mlb_classic_salary_csv,
                )

                _dk_mlb_classic_pool = parse_draftkings_mlb_classic_salary_csv(
                    _dk_mlb_classic_upload
                )
                if _dk_mlb_projection_upload is not None:
                    _dk_mlb_classic_pool = attach_draftkings_projections(
                        _dk_mlb_classic_pool,
                        _dk_mlb_projection_upload,
                    )
                    st.success(
                        "Forward projections matched "
                        f"{_dk_mlb_classic_pool.attrs.get('projection_match_count', 0)} "
                        "eligible MLB players."
                    )
                st.caption(
                    f"Loaded {_dk_mlb_classic_upload.name}: "
                    f"{len(_dk_mlb_classic_pool)} eligible players across "
                    f"{_dk_mlb_classic_pool['Game Info'].nunique()} games."
                )
                _dk_mlb_projection_matches = int(
                    _dk_mlb_classic_pool["ProjectionSource"]
                    .eq("uploaded_projection")
                    .sum()
                )
                if _dk_mlb_projection_matches == 0:
                    st.warning(
                        "No forward-looking Projected Points column was found. These "
                        "lineups are research-only rankings based on historical DraftKings "
                        "average fantasy points per game."
                    )
                elif _dk_mlb_projection_matches < len(_dk_mlb_classic_pool):
                    st.warning(
                        f"Forward projections matched {_dk_mlb_projection_matches} of "
                        f"{len(_dk_mlb_classic_pool)} eligible players. Unmatched players "
                        "still use historical DraftKings average fantasy points."
                    )
                _dk_mlb_classic_lineups = build_draftkings_mlb_classic_lineups(
                    _dk_mlb_classic_pool,
                    top_n=5,
                    require_confirmed_hitters=_dk_mlb_require_confirmed,
                    min_unique_players_between_lineups=_dk_mlb_min_changes,
                )
                if _dk_mlb_classic_lineups.empty:
                    st.warning(
                        "No valid MLB Classic lineup could be built from this file. "
                        "Confirm the contest contains two pitchers, every required hitter "
                        "position, team/game data, projections or FPPG, and a feasible "
                        "$50,000 roster. When confirmed-hitter safety is enabled, wait "
                        "until DraftKings has populated all required batting-order positions."
                    )
                else:
                    _dk_mlb_lineup_export = _dk_mlb_classic_lineups.drop(
                        columns=["Lineup Key"],
                        errors="ignore",
                    )
                    publication_dfs["MLB"] = _dk_mlb_lineup_export.copy()
                    st.dataframe(
                        _dk_mlb_lineup_export,
                        width="stretch",
                        hide_index=True,
                    )
                    st.download_button(
                        "Export MLB Lineup Review + Diagnostics",
                        _dk_mlb_lineup_export.to_csv(
                            index=False,
                            encoding="utf-8-sig",
                        ),
                        "draftkings_mlb_classic_top5_lineups.csv",
                        mime="text/csv",
                        key="download_draftkings_mlb_classic_top5_lineups",
                    )
                    st.download_button(
                        "Export MLB Position-Only CSV",
                        export_draftkings_classic_position_csv(
                            _dk_mlb_classic_lineups,
                            sport="MLB",
                        ),
                        "draftkings_mlb_classic_position_lineups.csv",
                        mime="text/csv",
                        key="download_draftkings_mlb_position_lineups",
                        help=(
                            "Contains only DraftKings Name + ID values under the Classic "
                            "position columns. Add contest-entry metadata from the exact "
                            "DraftKings template when required."
                        ),
                    )
            except Exception as exc:
                st.error(f"Unable to build MLB Classic lineups: {exc}")
