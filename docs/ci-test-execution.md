# CI test execution

The CI workflow keeps `production-safety` as a focused check and `full-suite` as the required gate for all application tests. The complete suite runs in two separate GitHub jobs, each with its own checkout and runner filesystem. `scripts/run_ci_tests.py` partitions pytest's collected files deterministically and retains collection order within each file. Every collected test belongs to exactly one shard. A failing, empty, or unsuccessful shard prevents the full-suite gate from passing.

New commits cancel superseded runs on the same pull request. Pushes to main retain separate runs. The scheduled research/grade workflow is independent and unchanged.

Each test job prints the 30 slowest test phases. JUnit XML reports include individual test durations and remain downloadable from the workflow artifacts for seven days, including after a test failure. Dependency installation already uses the pip download cache.

## Local verification

Install the same application dependencies as CI, plus pytest:

```sh
python -m pip install -r requirements.txt pytest
python -m pytest -q
# One selected module:
python -m pytest -q tests/test_locked_picks.py
```

`pytest.ini` sets default discovery to `tests/`, matching CI. Root-level manual
diagnostics and archived experiments are not collected by default. An explicit
file path can still be supplied when intentional. The obsolete `run_tests.py`
dummy and unittest-based `run_all_tests.py` have been removed; neither ran the
maintained pytest suite correctly.

To reproduce the CI partitions:

```sh
python scripts/run_ci_tests.py --shard 1 --shards 2 -q tests --durations=30 --junitxml=test-results/full-suite-1.xml
python scripts/run_ci_tests.py --shard 2 --shards 2 -q tests --durations=30 --junitxml=test-results/full-suite-2.xml
```

Run local shards sequentially unless they have separate working directories; some application tests use repository-relative paths. GitHub shards have separate runners.

The September 12 baseline ran 1,764 tests in 237 seconds on GitHub; repository checkout took only one second. Local profiling found distributed test cost rather than one long-running test. Warnings remain visible: DataFrame fragmentation and deprecation warnings require separate behavior-preserving fixes, not blanket suppression. Repo history or saved research does not need deletion to obtain this CI improvement.
