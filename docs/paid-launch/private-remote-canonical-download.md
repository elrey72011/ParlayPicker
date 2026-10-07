# Private remote canonical JSON download

This owner-only download reads existing objects from the configured
`PARLAYPICKER_DRIVE_FOLDER_ID` Shared Drive folder. It includes only
`parlaypicker/canonical-prospective-v1/` objects. The existing **Private canonical
evidence download** remains the local SQLite backup, including committed WAL.
Neither download supplies scientific acceptance or wagering authority.

Once this revision is separately merged and deployed by the owner:

1. Open **Publish board** and enter the existing **Publishing token**.
2. Expand **Private remote canonical JSON download**.
3. Click **Prepare remote canonical JSON download**.
4. Read its complete/partial status and click **Download private remote canonical
   JSON ZIP**.

For the next authentic-input inspection, select **prospective_reconciled_source**
in **Canonical table to download**, leave **Continue after verified canonical path
(optional)** blank, then prepare and download. This retrieves original reconciled
source records rather than restarting alphabetically at football coverage. Their
original source table, source record ID, raw bytes and hash identify retained
captures/model/feature dependencies; their existence does not establish a complete
prediction chain. NFL development remains deferred.

If the requested range is capped, copy `continuation.next_start_after` from its
manifest (also shown in the private panel) into the continuation field and prepare
the next download with the same table. Original ZIPs remain separate evidence;
no downloaded objects are hydrated or silently combined. A fresh full-folder
listing runs for each request. No cursor is offered after an incomplete listing
or before an object is verified. New remotely added objects before a cursor are
outside that explicitly requested range; cross-download atomic consistency is
not asserted. The original all-table/no-cursor download remains available.

`inventory.canonical_paths_by_table` records counts from the traversed metadata;
`request_scope` records the exact table/range, selected and excluded path counts;
`selection_complete` describes only that request. `export_complete` remains false
for any scoped or continued download. Omitted FK targets remain
`NOT_INCLUDED_REMOTE_UNKNOWN`, even after a requested table is fully retrieved.
Selection changes invalidate prepared ZIPs under the existing owner gate.

No refresh, provider request, upload, database initialization or restoration occurs.
Original remote media bytes remain unchanged at their logical paths in the ZIP;
`manifest.json` records the configured folder, opaque authenticated inventory
scope, traversal ID, file IDs, paths, checksums, row counts, distinct event/game
references, duplicates, unresolved dependencies and explicit resource limits.
Remote file identities with identical bytes share one ZIP member and remain
individually listed. Conflicting bytes, invalid canonical identities, unsupported
columns, source/payload hash corruption and credentials reject the whole export.
Records are never redacted to force export success.

The hard ceilings are 100 pages, 100,000 metadata entries, 50,000 logical paths,
128 MiB of received media, 60,000,000 bytes per object, 100,000 requests and 300
seconds. Streaming limit detection can receive one final chunk of at most 64 KiB;
it records all received bytes and excludes that incomplete object. Each request's
timeout is bounded by remaining time, with at most one read timeout outstanding.
The ZIP/manifest are bounded by these object/media ceilings and retained privately
in the authenticated owner's session, without a public-output or disk-cache path.

Pagination traverses the entire configured folder and filters locally; Drive
search indexing is not evidence of absence. Every retrieved duplicate and provider
checksum is verified. Listing, request, media, byte, object or time limits produce
explicitly partial exports. Missing canonical FK targets are reported as missing
from a complete export, or unknown outside a partial export. Optional unrecorded
prediction bindings and external feature/artifact/source availability remain
explicit. Pagination completeness is not an atomic snapshot, scientific chain
completeness or proof of permitted use. No historical probability is executed.

The next milestone is static inspection of one authentic event's original quote,
available-at features, consumed predictor/artifact/runtime and recorded prediction,
then its export/display boundaries. Counts and synthetic regressions cannot supply
these facts. NFL development remains deferred and unqualified selections PASS at
zero stake. Original evidence, frozen requirements and workflows remain unchanged.
