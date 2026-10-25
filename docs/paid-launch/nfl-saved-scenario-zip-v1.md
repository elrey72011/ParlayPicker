# Private saved NFL scenario ZIP replay

The owner authorized a separate, bounded UI/reader change from cleared main
`c273e3fa072cd6f161fb02abede159593ad07ce6` (reviewed tree
`1a5c6db4b8291542f96635af5379e9b2ad2aa8c6`). Its existing automatic checks
finished without retries: 23 passed, 20 artifact archives verified, 13 manifests
and 8,986 identities reconciled (8,984 passed and two inherited skips).

The hosted store can contain no authenticated NFL results. ZIP replay is a
separate session-only source, not a database importer or inference-input upload.
The existing default-off viewer and token gate remain. Choose **Upload saved
scenario ZIPs** before checking **Show saved NFL scenarios**; this branch never
resolves or opens SQLite. Existing private-database browsing keeps its released
readers and private-download behavior.

## Supported completed-result provenance

The original ZIP export has exactly `manifest.json`, `private-receipt.json` and
`scenario-result.json`. Both retained originals were independently compared with
the preserved local receipts, input packets, result bodies and archive digests.
The released reader accepted them under the recorded CPython 3.12.14 / NumPy
2.3.5 / pandas 3.0.1 Windows runtime, and rejected them with
`NFL_SCENARIO_RUNTIME_CONFLICT` under the hosted pinned NumPy 2.2.6 / pandas
2.2.3 packages. The live reader/model are unchanged.

The versioned static adapter therefore uses an independently reviewed trust
catalog, whose bytes are pinned by the reader. It registers only the two supplied
completed-result payload identities, their original input/receipt identities,
and the exact recorded reader/model/runtime/source-binding structures. The
catalog contains public model code/parameters and cryptographic references;
it contains no private source body, feature values, probabilities or approvals.
The complete input/result hashes bind the previously validated histories,
permissions, event mapping, six findings, feature ordering and inference clocks.
No new result can register itself by recomputing ZIP/internal hashes. Unknown
result identities or altered producer bindings are rejected, with a bounded
error code and field path. Adding another retained execution requires a separate
review and catalog change; this is not a general self-registration facility.

The adapter validates saved execution provenance, not the viewing runtime as a
new inference runtime. It does not rewrite runtime fields, modify authentication,
create approvals, execute embedded code, re-sign results, call a predictor,
recompute a prediction, or relax prospective execution/quote rules. It is not a
digital signature or proof of predictive accuracy. Existing uncalibrated private
research / PASS / zero-stake qualifications remain.

## Memory, authentication and limits

Uploads are token-protected, per-session Streamlit file inputs. No body or
acceptance verdict uses a shared/global cache. Authorization loss removes all
saved-viewer state. Clear rotates both uploader and selector identities; changing
files or selection invalidates the display. Load and subsequent display perform
fresh authentication. Input staging and the native Run action remain separate.

Limits are four files, 4 MiB compressed per file, 8 MiB compressed per selection,
three exact root members, 8 MiB declared/actual total inflation and 6 MiB for the
result member (16 KiB each for receipt and manifest). Central-directory counts
are checked before ZIP parsing; inflation is bounded, including a 64:1 ratio
ceiling. Duplicate/unexpected/unsafe members, encryption, symlinks, unsupported
compression/schema, malformed/duplicate-key JSON, conflicting receipts and
hash/provenance mismatches are rejected. Content is never extracted to disk.

Only saved fields are displayed: event, selected team/handicap, native cover
probability, selected-team margin projection before handicap, residual scale,
original inference timestamp, separate viewing time and input/result hashes.
Post-kickoff viewing remains static replay, with no quote, EV, recommendation,
current-price claim or execution/wagering authority. Nothing enters public
publishing or ordinary offer-backed pick rows.

## Authorized validation and release boundary

Use authentic files only in the private Windows environment, through the real
browser/file chooser with all predictor calls, provider connections and SQLite
access forbidden. Preserve all original hashes and clocks. Keep authentic bodies,
archives, screenshots and private evidence out of Git and hosted artifacts.

Linux CI uses synthetic exports with the same recorded producer/runtime/source
structures. Its explicit `SyntheticPolicy` supplies independently constructed
fixture anchors and can admit only `SYNTHETIC` packets; it cannot admit retained
originals. No authentication function is patched. The production UI never takes
a policy from uploads, environment variables or URLs. AppTest supplies only the
file-input boundary synthetically and is reported separately from actual browser
verification. Tests cover both results, switching, clearing/re-upload, session
isolation, wrong tokens, inactive paths, original remote-export controls, malformed
archives, receipt substitution and unsupported bindings. Original remote-export
assertions, SQLite prohibition, three-second timeout, model/replay sources,
workflows and capacity limits remain unchanged.

The owner amended the separate-private-Linux requirement: install no WSL, Docker,
QEMU or VM. Windows authentic tests plus Linux synthetic tests do **not** establish
authentic hosted compatibility. That specific verification remains **PENDING**
until owner-approved merge, release clearance, deployment and the final owner-only
upload. No merge or deployment is authorized here.

After combined correctness and existing capacity gates pass, publish one separate
draft PR and allow ONE complete final-head CI cycle. No automatic retries or
extra hosted diagnostics. Record the exact scope through the existing source
binding/implementation/policy-only seal controls. No provider acquisition,
prediction, credentials, Refresh, model changes, migration, training, scheduling
or operational changes. Preserve Monday's NFL preparation, Delaware–WKU and all
prior evidence; Chrome file permissions remain unchanged.

## Owner hosted upload after merge, clearance and deployment

1. Open Workspace → Preview & Publish and enter the existing owner token.
2. Expand Private NFL sporting scenario. Choose **Upload saved scenario ZIPs**,
   then check **Show saved NFL scenarios**. Do not enable/run a new scenario.
3. In **Saved NFL scenario ZIP exports (multiple files)** upload the preserved
   `KC-away-plus2p5.private-scenario.zip` and `SEA-home-minus2p5.private-scenario.zip`.
4. Wait for upload completion, select both names under **Uploaded ZIPs to view**
   (or Select all), then click **Load uploaded saved results**.
5. Confirm the expected payload hashes supplied by the owner, original inference
   timestamps and probabilities against the private local verification report.
   Switch to each side and Load;
   the old display must clear on selection changes. Clear must remove uploads,
   labels and results. Re-upload/select/Load both again.
6. Confirm no inference, persistence, provider request or normal Refresh occurred.
   Record this deployed hosted check separately; until it succeeds, keep authentic
   hosted verification pending. A failure must report the exact rejected field,
   never be resolved by rewriting the original or weakening provenance checks.
