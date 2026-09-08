# Automated research capture and grading

The **Research capture and grading** GitHub Actions workflow runs every 15 minutes (minutes 7, 22, 37, 52 UTC), independently of Streamlit. It is disabled until the repository variable below is set. GitHub schedules are best-effort and can be delayed; no exact start-time coverage is guaranteed.

## One-time setup after merge

1. In Streamlit, freeze each league's models and run its Drive restore/backup button successfully. The scheduler never fits or re-freezes a model. Existing runtime hashes are unchanged by this PR.
2. In GitHub repository Settings â†’ Secrets and variables â†’ Actions, add secrets `PARLAYPICKER_DRIVE_FOLDER_ID`, `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT`, `CFBD_API_KEY`, and `ODDS_API_KEY`. Use the existing Shared Drive folder and service account. The service-account value must be the raw complete JSON, without Streamlit's surrounding triple quotes. Streamlit secrets are not automatically shared with GitHub. Do not put credentials in commits or workflow inputs.
3. Set repository variable `RESEARCH_SPORTS` to `MLB,NCAAF` (or just one league). CFBD/Odds keys are needed only for NCAAF. Set `RESEARCH_SCHEDULER_ENABLED` to `true`.
4. In Actions, select **Research capture and grading â†’ Run workflow**. Inspect its run summary and confirm it succeeds. For failures, check provider keys, Drive access, and that a compatible frozen cohort was backed up. Configure GitHub Actions failure notifications in your own account if desired; the code does not send email or other messages.
5. In Streamlit, use the league's restore/backup button to retrieve scheduled captures and grades. Download the prospective report and records for verification. Disable the scheduler by setting the enable variable to `false`.

## Behavior and limits

Each league restores and verifies Drive evidence before work. A restore failure prevents its captures. New events must start within two hours and remain pregame at capture. Existing captured games in the current cohort are skipped; underlying reports still preserve the first entry if manual and scheduled attempts overlap. Expected ineligibility is reported without inventing inputs. Other validation/provider failures mark the run failed.

MLB attempts at most six new games per cycle (three data requests each plus the upcoming schedule), and grades at most six ended games (up to two requests each). Failed capture and unfinished grade attempts rotate so they do not permanently block later games. NCAAF refreshes up to six CFBD requests, resumes its input checkpoint across runs, and refreshes inputs before their 24-hour limit. One Odds API request is made only when new schedule candidates have eligible prior features and are in range, with at most 24 candidate games per cycle. Existing NCAAF eligibility and quote-age rules remain authoritative; some/all candidates may be excluded. Grading uses up to six CFBD requests. Odds API billing may count multiple market credits per request. Remote storage requests are additional and grow with history.

Evidence is synced after capture/grade work and in cleanup. Scheduler state, including NCAAF inputs, attempt cursors and safe status, is immutable and read-back verified under `parlaypicker/research-scheduler-v1/` in Drive. Each league retains its existing evidence prefix. The workflow has one concurrency group and does not cancel a running cycle. Manual Streamlit actions are not locked by GitHub; restore afterward to reconcile evidence. Forced runner termination can still interrupt a pending backup, so successful run status matters.

Failures produce a nonzero exit and a credential-free Actions summary. No raw provider exceptions or secret values are logged. The workflow does not publish evidence artifacts to GitHub, change wagers, guarantee wins, or automate closing-line capture. Repeated errors require operator attention; do not re-freeze merely because a run is pending or no games qualify.
