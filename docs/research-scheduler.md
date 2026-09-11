# Automated research capture and grading

The **Research capture and grading** GitHub Actions workflow runs every 30 minutes in the **11:45 a.m.–2:30 a.m. America/New_York** operating window, independently of Streamlit. Regular starts are 11:45 a.m., 12:15 p.m., then every 30 minutes through 2:15 a.m. the following morning (30 starts daily). The 2:30 a.m. minute is the final allowed minute for delayed/manual starts, not an additional regular run. UTC triggers cover both daylight-saving offsets; an Eastern gate skips off-hours before dependencies or API work. It is disabled until the repository variable below is set. GitHub schedules are best-effort and can be delayed; no exact start-time coverage is guaranteed.

## One-time setup after merge

1. In Streamlit, freeze the MLB/NCAAF models and run its Drive restore/backup button successfully. The scheduler never fits or re-freezes a model. NFL market tracking needs no frozen model; it does not produce model predictions.
2. In GitHub repository Settings â†’ Secrets and variables â†’ Actions, add secrets `PARLAYPICKER_DRIVE_FOLDER_ID`, `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT`, `CFBD_API_KEY`, and `ODDS_API_KEY`. Use the existing Shared Drive folder and service account. The service-account value must be the raw complete JSON, without Streamlit's surrounding triple quotes. Streamlit secrets are not automatically shared with GitHub. Do not put credentials in commits or workflow inputs.
3. Set repository variable `RESEARCH_SPORTS` to `MLB,NCAAF,NFL` (or a subset of these leagues). CFBD is needed for NCAAF; the Odds key is shared by NCAAF and NFL. Set `RESEARCH_SCHEDULER_ENABLED` to `true`.
4. In Actions, select **Research capture and grading â†’ Run workflow**. Inspect its run summary and confirm it succeeds. For failures, check provider keys, Drive access, and that a compatible frozen cohort was backed up. Configure GitHub Actions failure notifications in your own account if desired; the code does not send email or other messages.
5. In Streamlit, use the league's restore/backup button to retrieve scheduled captures and grades. Download the prospective report and records for verification. Disable the scheduler by setting the enable variable to `false`.

## Behavior and limits

Each league restores and verifies Drive evidence before work. MLB/NCAAF use frozen models; NFL uses [market tracking](nfl-market-tracking.md). A restore failure prevents its captures. New events must start within two hours and remain pregame at capture. Existing captured games in the current cohort are skipped; underlying reports still preserve the first entry if manual and scheduled attempts overlap. Expected ineligibility is reported without inventing inputs. Other validation/provider failures mark the run failed.

MLB attempts at most six new games per cycle (three data requests each plus the upcoming schedule), and grades at most six ended games (up to two requests each). Failed capture and unfinished grade attempts rotate so they do not permanently block later games. NCAAF refreshes up to six CFBD requests, resumes its input checkpoint across runs, and refreshes inputs before their 24-hour limit. One Odds API request is made only when new schedule candidates have eligible prior features and are in range, with at most 24 candidate games per cycle. Existing NCAAF eligibility and quote-age rules remain authoritative; some/all candidates may be excluded. Grading uses up to six CFBD requests. Odds API billing may count multiple market credits per request. Remote storage requests are additional and grow with history.

Evidence is synced after capture/grade work and in cleanup. Scheduler state, including NCAAF inputs, attempt cursors and safe status, is immutable and read-back verified under `parlaypicker/research-scheduler-v1/` in Drive. Each league retains its existing evidence prefix. The workflow has one concurrency group and does not cancel a running cycle. Manual Streamlit actions are not locked by GitHub; restore afterward to reconcile evidence. Forced runner termination can still interrupt a pending backup, so successful run status matters.

Failures produce a nonzero exit and a credential-free Actions summary. No raw provider exceptions or secret values are logged. The workflow does not publish evidence artifacts to GitHub, change wagers, guarantee wins, or automate closing-line capture. Repeated errors require operator attention; do not re-freeze merely because a run is pending or no games qualify.


## Paid API protection

The scheduler caps CFBD at **25 requests per UTC day and 500 per rolling 31 UTC days**, and The Odds API at **200 credits per UTC day and 7,500 per rolling 31 UTC days**. The rolling window is deliberately conservative across monthly billing/reset dates. Limits are defined in `app_core/research_api_budget.py`.

This allocates at most half of the stated 1,000-CFBD / 15,000-Odds allowances to new automation usage. It does not measure earlier usage, Streamlit/manual requests, or other clients; therefore it cannot guarantee the account's remaining balance. Review provider dashboards for those totals. Budget accounting begins when this version is deployed and retains history across run restarts and model freezes. Existing usage is not retroactively counted.

Each request reserves and read-back verifies its budget in Drive before contacting the provider. Failed calls/timeouts keep their reservations. NFL and NCAAF share the same ODDS ledger. Current odds requests reserve three credits for three markets in one US region. NFL final-score requests with `daysFrom=3` reserve two credits; NFL event discovery reserves zero. No refunds are made even if the provider reports fewer credits. Inspect `x-requests-last` for unexpected higher costs and stop further paid requests if the pricing contract changes. Unsupported endpoints/market combinations fail closed. This is conservative allocation, not an exact billing dashboard.

When a daily or rolling cap is reached, paid requests pause automatically; usage and pause reasons appear in the Actions summary and saved scheduler state. They resume only as budget becomes available. No cap is reset by rerunning a job or re-freezing a model. The existing Actions concurrency group serializes reservations; do not run separate scheduler processes against the same Drive state. Manual Streamlit buttons are outside this budget.

The CLI refuses off-hours work, and paid requests check the operating window individually. No new MLB capture/grading batch starts after the cutoff; already-running MLB requests and storage verification may finish. Limits do not meter the public MLB API, Drive or Actions runner time. No closing-line automation is added.

The daily UTC cap can pause captures or grading before all games are processed; a schedule is not a promise of complete coverage. Daily and rolling limits protect automation usage even if all 30 cycles have work.

The 30-start count describes ordinary days; daylight-saving transition nights can have a different number of wall-clock slots, and GitHub may delay or skip scheduled triggers. Caps remain independent of cadence. Odds credit semantics: https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs .

## NFL enablement and verification

After merging NFL support, include `NFL` in `RESEARCH_SPORTS`. A manual workflow run can override the sports list (for example `NFL`) without changing scheduled configuration. The same concurrency lock, operating window, storage verification and shared paid budget apply. There are no new secrets.

NFL only collects market observations and final-score comparisons; it never invokes or trains an independent prediction model. Its Streamlit panel restores and exports evidence without paid API calls. Scores are available for a short recent window; interruptions over three days can leave unresolved games. Existing 200-credit daily protection remains, even though the shared rolling ceiling rises to 7,500. Provider account balances and manual usage remain outside this ledger.

## Automatic public-results grading

Set the GitHub Actions **variable** `PARLAYPICKER_NETLIFY_SITE_ID` to the same project/site ID configured in Streamlit. The existing Drive credentials and enabled research workflow are reused. No Netlify token is required. Without this variable, research continues and the summary reports public grading as not configured.

Each operating-window cycle restores confirmed publications, imported research recaps, score revisions and grading status from Drive. It checks only pending entries in `RESEARCH_SPORTS`, between three hours and 30 days after scheduled start. At most two league/date batches are fetched per cycle, with at least one hour between attempts for each batch. The existing ESPN scoreboard reader supplies final scores; this adds no CFBD or The Odds API requests. Drive reads and writes are additional and grow with history. A successful check can leave games pending when final scores or an unambiguous match are unavailable.

New score revisions and grading status are immutable and verified in Drive. Already settled entries are not automatically rechecked; use manual grading for score corrections or older unresolved games. Imported recaps remain separately labeled research, with no claim of verified pregame publication. Provider/storage errors fail the workflow and expose only exception classes in its summary.

In Streamlit, open **Workspace > Preview & Publish > Public results history**, enter the publishing token and select **Restore public history from Drive**. The panel shows the latest saved automatic grading status, last successful cycle and pending category count. A successful cycle means the scheduled check completed, not that every game settled. Build a fresh preview and publish it manually to update Netlify. The scheduled job never deploys the public site. Verify one workflow run and restore its status after enabling this setting.


## Schedule and final-score resilience

The MLB scheduler requests only the Eastern dates intersecting its two-hour capture window, rather than a full season of hydrated schedule data. Schedule reads retry timeouts and connection failures at most twice, with short backoff. Exhausted retries still fail visibly. This does not change frozen model files or require refreezing.

NFL score grading requires the same provider event ID and exact home/away teams. Kickoff revisions of at most 15 minutes are accepted only when the original capture predates both scheduled times, both times are in the past, and the final-score timestamp and score pair validate. Scores retain the original start and a separate reported start for audit. Larger schedule shifts and identity changes remain rejected. Rejection diagnostics contain safe reason codes and timing fields; invalid results are never marked as wins or losses.
