# Daily publishing workflow

1. Click **Refresh picks** to run game analysis. Saved history refreshes automatically when the authenticated publishing panel opens.
2. Open **Workspace → Preview & Publish** and enter the publishing token if needed. History loads once per session; the preview builds and updates automatically.
3. Select the picks you want and click **Lock selected picks**. The original line and odds are saved to Drive, then the reviewed board (including selected props and DFS) is published to Namecheap and checked over HTTPS. Existing locks cannot be replaced. All eligible options are selected initially; deselect games you want to leave until later.
4. For other changes, review the preview and click **Publish board**. Destination verification, Drive backup, upload, HTTPS verification and confirmation happen within this action.
5. Click **Update results and publish** to reload history, grade pending games and MLB props, and publish updated results. MLB collection remains bounded to ten final boxscores per date per action. Missing or ambiguous data stays unresolved; use the advanced grading controls for review/correction rechecks.

Fresh saved props are included initially; the owner can change the checkbox. Generate DK lineups in Full Pick Board and select the slate, name and lock time in the publishing panel. A single Classic slate is supported per publication. Game analysis and Player Props still run separately.

Locked picks display whether publication has been verified. The website section also indicates whether the current preview is published. The app does not automatically retry uncertain uploads: use Check public status before explicitly allowing another attempt. Successful locks remain saved even if publishing fails.

Downloads, local output, historical imports and individual grading are available in collapsed sections. The How Picks Work folder is unchanged. The combined publishing actions target the configured Namecheap/SFTP host; legacy Netlify controls remain manual.

## Timing and eligibility

New game locks require fresh game analysis and a supported price timestamp or
labeled ESPN observation, each within 30 minutes, on the game's Eastern date and
before its start. **Refresh picks** renews game analysis; **Run Player Props**
updates props separately. **Refresh preview** rebuilds the saved board without
fetching new quotes. Lock games before optional prop/DFS work if time is short.

ESPN college research lines show **Observed at ... via ESPN (sportsbook update
time unknown)**. This is the snapshot capture time, not a DraftKings update time.
Existing locks preserve their original observation and price. The
**Why games cannot be locked** section explains exclusions and offers a CSV.
See [lock eligibility](lock-eligibility-audit.md) for the full policy.

To grade yesterday's results, open this panel and use **Update results and
publish** without first running new picks or props. The public record begins
September 11, 2026 (Eastern); earlier history remains archived. Results separate
locked overall picks, published overall/sides/totals/parlays, and the first
qualifying published Top 10 cohort. These groups overlap.
