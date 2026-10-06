# Private canonical evidence download

This bounded owner tool exports an existing canonical store. It does not change
inference, research eligibility, source acceptance, qualification or wagering.
NFL remains deferred; the MLB, NHL and NCAAF contracts remain unchanged.

In **Publish board**, enter the existing publishing token, open **Private canonical
evidence download**, and select **Prepare canonical evidence download**, then
**Download private canonical evidence ZIP**. This is available even with no loaded
game analysis. Preparing reads local storage only; it never restores remote
objects, initializes a missing store, runs analysis or contacts providers.

The resolver uses `app_core.prediction_evidence.database_path()` and replaces
`evidence.sqlite3` with `prospective-evidence.sqlite3`. The effective directory is
the process environment's `PARLAYPICKER_EVIDENCE_DIR`, or the application root's
`data/prediction_evidence`. Root-level Streamlit secrets become environment
variables under Streamlit; this exporter does not create a separate resolver.
An explicitly empty directory value resolves to the process working directory;
an absent variable uses the application-root default. Relative configured paths
also resolve against the producer's working directory. The manifest distinguishes
the presence of the setting rather than treating an empty value as absent.
The generic `evidence.sqlite3` stores prediction snapshots; `nhl-market.sqlite3`
is a separate native store. Their populations cannot establish canonical cover
artifact availability. Canonical remote objects use
`parlaypicker/canonical-prospective-v1/`; no remote contents are read by this tool.

The download contains only `prospective-evidence.sqlite3` and `manifest.json`.
`sqlite3.Connection.backup` opens the source with `mode=ro`, retaining committed
WAL contents and excluding uncommitted transactions. It copies to a temporary
destination, closes it, then inventories the destination with an immutable
reader. It never checkpoints, vacuums, restores or migrates the source. It
preserves original record fields and blob bytes, not the source's physical page
layout or WAL file. A missing local file reports remote availability UNKNOWN.

The manifest contains snapshot SHA-256/size, actual schema and schema hash,
contract/user versions, per-table counts, absent contract tables and available
model/dependency identifiers and clocks. Raw dependencies remain in the private
database. External artifact contents are not included and remain UNKNOWN.
References are capped at 10,000 with an explicit truncation flag; the database
still contains all records. Schema tables must belong to the canonical contract
and have its exact primary keys. Unknown tables fail closed.

Limits are 128 MiB and 30 seconds. Busy, corrupt, unsupported, oversized or timed
out stores return fixed diagnostic codes and no partial download. Recognized
credential fields, credential URLs/private-key markers and configured credential
values cause refusal rather than historical redaction. Configuration, credentials
and session state are never packaged. This is an owner-private artifact; do not
attach its database or manifest to GitHub or put it in a public board package.

Preparation is explicit. Session downloads are invalidated when the effective
directory changes. A prepared snapshot retains its preparation timestamp and
does not claim to contain later commits. Downloading uses `on_click="ignore"`.
The existing constant-time publishing-token check precedes this UI. No public
rendering or publication path calls the exporter.

Regression execution uses labelled synthetic databases and blocks network access.
Tests cover committed WAL, uncommitted exclusion, byte preservation, snapshot
integrity, concurrent writer isolation, model/dependency references, owner gating
with no loaded analysis, scope invalidation, corrupt/missing/unsupported stores,
credential refusal and bounded limits. Authentic retained packets are inspected
statically only; no historical numeric reconstruction is authorized here.

Engineering completeness does not establish authentic compatible inputs,
point-in-time/out-of-sample lineage, permitted use, settlement applicability or
scientific acceptance. Unsupported or unqualified selections remain PASS at
zero stake. A separately authorized deployment is required before this new UI
exists on the hosted producer.
