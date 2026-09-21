# MLB development freeze

Frozen at 2026-09-21T16:45:35.297252+00:00 UTC.

45 independent settled games per family from September 17–20; complementary selections are deduplicated before fitting. Separate fixed ridge classifiers fitted for spreads and totals using the existing trainer configuration. No evaluation outcomes were used.

Artifact directory: `output\mlb-development-freeze\ea0b18f0b6ac475a0a378b4a62a32a453ab33b8590fa0dd3292125b9b25bf997`. The directory contains the immutable plan, fitted development estimators and exact source dataset. Plan hashes bind the data, trainer source, configuration and estimator bytes.

Validation: September 22–24 Eastern. Holdout: September 25–27 Eastern. Each family/period requires at least 20 independent event/line units. Insufficient evidence means incomplete, with no automatic window extension or promotion. Preserve this artifact and capture pregame evidence throughout both windows.

This is a development-only artifact, deliberately incompatible with the runtime manifest. Future evaluation must use these exact estimator bytes, establish availability before each evaluated receipt, verify chronological provenance, and compare paired baselines. No calibration, validation success, live installation or wager authority is claimed.
