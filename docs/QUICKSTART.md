# Quick start

## Local setup

Use Python 3.12, matching CI. From the repository root:

```sh
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS/Linux
source .venv/bin/activate
```

Then install the application's dependency set and launch the active entry point:

```sh
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Supply provider credentials through environment variables or
`.streamlit/secrets.toml`. Use [Drive setup](evidence-storage-setup.md) and
[Namecheap setup](namecheap-publishing.md) for history and publication settings.
Existing installations should preserve their current history namespace and
credentials. No new model training is required just to open the app.

## First analysis and publication

1. Choose the sport and bankroll, then click **Refresh picks**.
2. Review the saved game selections. Use **Run Player Props** separately when
   player props are needed; this does not refresh the game prices.
3. Open **Workspace → Preview & Publish** and enter the publishing token.
4. Select the eligible games you want under **Lock Overall Best Picks** and click
   **Lock selected picks**. Lock promptly after game analysis; do not wait for
   optional prop or DFS work if the game quotes are nearing expiration.
5. Include fresh saved props or generated DFS lineups and use **Publish board**
   for subsequent updates. Open the public website to inspect the result.

Game prices or labeled ESPN observations and game analysis must be within 30
minutes for a new lock, and the game must not have started. Existing locks stay
saved when you refresh. **Why games cannot be locked** explains each exclusion.
Refreshing the preview does not refresh prices or observation timestamps.

For grading, use **Update results and publish** in the publishing panel; new
analysis is not a prerequisite. See [the daily workflow](publishing-workflow.md)
for props, DFS, pending results, and publication verification.

## Development checks

```sh
python -m pip install pytest
python -m pytest -q
```

See [CI test execution](ci-test-execution.md) for selecting tests and running
isolated shards. Historical enhanced-build instructions are archived references;
`streamlit_app.py` is the current application.
