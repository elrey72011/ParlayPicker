# ParlayPicker

ParlayPicker is a Streamlit workspace for sports analysis, saved research picks,
player props, DraftKings Classic lineups, and graded public results. The owner
chooses which picks to lock and publishes a static website to Namecheap/SFTP.
Research selections and estimated probabilities are distinct from approved
wagers and observed results.

## Run locally

Use Python 3.12 to match GitHub CI. Create and activate a virtual environment,
then run:

```sh
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Configure provider credentials in environment variables or the local
`.streamlit/secrets.toml`; never put credentials in tracked files. See
[the quick start](docs/QUICKSTART.md), [Drive evidence setup](docs/evidence-storage-setup.md),
and [Namecheap publication setup](docs/namecheap-publishing.md).

## Daily workflow

1. Click **Refresh picks** for game analysis. **Run Player Props** is a separate
   action with its own saved results and timestamp.
2. Open **Workspace → Preview & Publish** and enter the publishing token.
   Saved history loads automatically once per session.
3. Choose the games under **Lock Overall Best Picks**, then click
   **Lock selected picks**. This saves the original selections to Drive and
   publishes the reviewed board. Deselect any eligible options you want to leave
   for later. Existing locks retain their original picks and prices.
4. Include saved props or a generated DraftKings slate as needed. Use
   **Publish board** for changes that do not create new locks.
5. To grade outstanding results, click **Update results and publish**. This can
   be done without running new game analysis. Unresolved results remain pending
   or need review.

Game locks require the game date in Eastern time, a game that has not started,
and analysis plus a supported quote/observation within the 30-minute window.
**Refresh preview** does not fetch fresh odds. ESPN college research picks may
use **Observed at ... via ESPN**, which measures when the snapshot was fetched;
the sportsbook's own update time remains unknown. These picks remain research
selections, not wager approvals.

See [the complete publishing workflow](docs/publishing-workflow.md),
[lock eligibility explanations](docs/lock-eligibility-audit.md),
[separate game and prop analysis](docs/separate-analysis.md), and
[player-prop grading](docs/public-prop-results.md).

## Website and results

The public site includes Picks, Player Props, Parlays, Results, DraftKings DFS,
and How Picks Work. Locked selections retain their saved line, price, source,
and timing. The Results page separates the locked overall record from first
published Overall Best Picks, Sides, Totals, and Parlays. Categories overlap and
must not be added together. The first qualifying published Top 10 cohort of a
day is tracked separately; later updates do not rewrite that cohort.

Win percentages exclude pushes, pending results, and unresolved cases. Locking
is a record of an owner-selected pick; it does not place a bet or establish that
the selection passed the wager-approval checks.

## Code map

| Path | Responsibility |
| --- | --- |
| `streamlit_app.py` | Active Streamlit entry point |
| `app/ui/` | Owner controls, publishing, history, and lock panels |
| `core/streamlit_pipeline.py` | Shared game-analysis orchestration |
| `app_core/` | Provider adapters, modeling, evidence, props, and public records |
| `publishing/board.html` | Static public website template |
| `scripts/` | Maintained command-line tools and scheduled workflow entry points |
| `tests/` | Automated pytest suite used by CI |
| `data/`, `models/` | Curated datasets, calibration inputs, and model artifacts |
| `archive/experimental_scripts/` | Historical experiments; not the active application |

Gemini is an optional bounded secondary reviewer. Enabling it does not replace
quote verification or the deterministic wager checks. See
[Gemini review](docs/gemini-review.md), [selector validation](docs/selector-validation.md),
[qualified-pick evaluation](docs/qualified-picks-evaluation.md), and
[scheduled research/grading](docs/research-scheduler.md).

## Tests and maintenance

```sh
python -m pip install pytest
python -m pytest -q
```

Default discovery targets `tests/`, matching the CI suite. Root-level diagnostics
are manual tools and may perform provider calls or write data; run only the
specific tool you intend. For timing reports and isolated CI shards, see
[CI test execution](docs/ci-test-execution.md).

Local `output/`, `outputs/`, audit captures, and runtime parlay logs are ignored
by Git. Curated data, model artifacts, calibration files, and historical records
are retained. See [repository maintenance](docs/repository-maintenance.md) before
removing research outputs or changing dependencies and shared pipeline code.
