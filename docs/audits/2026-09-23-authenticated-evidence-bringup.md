# Authenticated evidence bring-up: current blocker and owner settings

The first post-merge `main` research workflow run was [35951189887](https://github.com/elrey72011/ParlayPicker/actions/runs/35951189887) at commit `eab9ce44ebfe09d61b053e610c92326369af1b42`. GitHub reported **Invalid workflow file** at line 30: `Unrecognized named-value: 'runner'` in `${{ runner.temp }}`. The run had no jobs. This was a workflow parsing failure, so it says nothing about whether repository secrets exist or whether providers or Drive are reachable.

The workflow fix moves `RESEARCH_AUDIT_PATH` to the step where `runner.temp` is valid. A separate configuration preflight now reports only setting names and presence, and stops research capture before provider or Drive calls if a required setting is absent. GitHub's unauthenticated Actions variables endpoint requires authentication, and the available GitHub connector has no secret/variable listing or workflow dispatch action. Current repository setting presence is therefore **unverified**, rather than presumed missing. The local workstation has none of the credentials; that is not evidence about Actions.

The last inspected scheduled research run before PR #2334, [35940380421](https://github.com/elrey72011/ParlayPicker/actions/runs/35940380421), did start both jobs successfully. Its Actions log showed all four required secret environment fields masked (`***`), so those settings appeared present **at that earlier run** and `RESEARCH_SCHEDULER_ENABLED` permitted execution then. This does not prove they are present and valid now or that the new six-sport path can reach their providers.

## Exact owner configuration

Use **Repository → Settings → Secrets and variables → Actions**: [Secrets](https://github.com/elrey72011/ParlayPicker/settings/secrets/actions) and [Variables](https://github.com/elrey72011/ParlayPicker/settings/variables/actions). The workflow declares no GitHub environment, so repository-level settings (or organization secrets explicitly shared with this repository) are needed. Do not paste values into an issue, PR, or log.

| Type | Name | Required value or format | Why |
| --- | --- | --- | --- |
| Variable | `RESEARCH_SCHEDULER_ENABLED` | Literal `true` (lowercase) | Enables the research and activation-evidence jobs. Missing or any other value skips them. |
| Secret | `PARLAYPICKER_DRIVE_FOLDER_ID` | ID segment of an active **Google Workspace Shared Drive folder**, not its URL | Canonical/native evidence restore and immutable backup. |
| Secret | `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT` | Complete original Google service-account JSON object, including `type: service_account`, `client_email`, `private_key`, and `token_uri` | Authenticated Drive access. Supply raw JSON as the GitHub secret value; grant that service account access to the Shared Drive folder and enable the Drive API. |
| Secret | `ODDS_API_KEY` | The Odds API key with event, spread/total quote, and score access for the requested sports | NFL, NCAAF, NBA, NCAAB, and NHL provider capture. The workflow also maps it to `THE_ODDS_API_KEY`. |
| Secret | `CFBD_API_KEY` | CollegeFootballData API key with game access | NCAAF event/team identity and results. |
| Variable | `PARLAYPICKER_NETLIFY_SITE_ID` | Existing Netlify site ID, if public-results grading is desired | Optional for the six-sport research cycle; absence is reported as `not_configured`. |
| Variable | `RESEARCH_SPORTS` | Optional `NFL,NCAAF,NBA,NCAAB,MLB,NHL` | Defaults to all six. A manual workflow-dispatch `sports` input overrides it. |

The exact six-sport run requires the enabled variable and four secrets above. Given the earlier successful scheduled job, do not replace any existing value merely because the post-merge parser failed. The preflight checks current presence without displaying values; successful presence does not prove provider entitlement, API quota, Drive folder permission, or remote integrity. Those require the live cycle.

## Run and evidence status

After the fix is on the selected branch, use **Actions → Research capture and grading → Run workflow** with `sports` set to `NFL,NCAAF,NBA,NCAAB,MLB,NHL`, during the workflow's Eastern operating window (11:45 a.m.–2:30 a.m.). Inspect the configuration-preflight job first. If it passes, the research job should restore remote stores, freeze plans in the canonical database, capture/reconcile, back up, and upload `six-sport-research-cycle-audit`. A first cycle remains unverified until the audit and remote read-back succeed.

As of this report, there is no authenticated six-sport run ID, remote inventory/hash, canonical plan artifact hash, real reconciliation count, model fit, or certified-close source verification to report. Local zero counts are not proof of empty remote stores. All markets remain `UNVALIDATED`, ineligible for production, and at zero stake; no wager was placed.
