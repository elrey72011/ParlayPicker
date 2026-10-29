# Owner-only, connection-free TLS trust preflight

Authorized successor of cleared main c548ddb2693b64c94942ff315eed93e3fcb2f56e,
tree e9c9ab85b66576cb9a04daa83ed956af69955b26. The observed hosted failure is
TLS certificate verification, not a demonstrated missing CA bundle. Preserve
the original SPENT/UNKNOWN and both SPENT/FAILED operations. This inspection
neither spends nor replenishes a database allowance.

The existing owner-only **Read running NFL build information (no connection)**
button now includes TLS observations. No new app, upload, arbitrary path, command,
DNS, socket, provider, database, prediction or trust-setting change is introduced.
Normal renders do not inspect; only the existing explicit owner action does.
Authorization loss clears the existing private build receipt. Existing connection
diagnostic, storage, model, transport, deadlines and both false switches are unchanged.

## What is observed

The helper snapshots this process's bounded procfs mappings before its explicit
Python SSL inspection. It reports only recognized libpq/libssl/libcrypto basenames,
mapped device/inode identities, and the psycopg implementation. It associates the
loaded psycopg C/binary extension with its libpq, SSL and crypto objects through
bounded ELF DT_NEEDED reads. Multiple/missing matches remain UNKNOWN; the presence
of a library elsewhere in the process is not proof it serves psycopg.

The selected crypto library is revalidated against procfs and accessed with
RTLD_NOLOAD only. Fixed, zero-argument OpenSSL trust-path getters and the fixed
version getter inspect the already-loaded implementation; no new library is
searched or loaded. Unsupported symbols/platforms stay UNKNOWN. Python SSL's version
and defaults are separately labelled and never substituted for driver findings.
Pure-Python psycopg still reports its implementation and mapped library inventory;
an unproven driver association remains UNKNOWN.

Only the fixed public candidate /etc/ssl/certs/ca-certificates.crt is opened as a
bundle. It must be a regular, non-symlink file at most4MiB; identity/size are checked
again after opening. Output includes existence, readability, public-file SHA-256,
size and the explicit parser's certificate/CA counts. The hash is emitted only after
successful CA-bundle parsing, not for malformed or private-looking input. Missing, unreadable, oversized,
changed or invalid files have distinct statuses. Parsing uses a new Python SSL
context with only these bytes; it does not authenticate their OS provenance, prove
driver selection, check a remote chain or establish that Neon will validate.

Procfs is limited to2MiB/32 recognized objects; each mapped ELF read is at most8MiB,
with inode/device and regular-file checks. Only already-mapped recognized libraries
and loaded extension anchors can supply those internal file paths. No path input
or path-valued setting is opened as a file. Compiled OpenSSL default locations under
conventional public /etc, /usr or /opt trust paths receive metadata-only checks;
there is no content reader for arbitrary defaults. Unknown paths are redacted.

Allowlisted environment fields are presence booleans for SSL_CERT_FILE, SSL_CERT_DIR,
OPENSSL_CONF, OPENSSL_MODULES and PGSSLROOTCERT. Only fixed public path classes,
system or OTHER_REDACTED/UNSET are emitted for trust selectors. No environment dump,
hostname, connection string, certificate contents, raw exception or secret hash.
The bundle hash identifies public CA-file bytes only. Settings are never mutated.

## Qualification and authenticated history

Local tests block DNS, sockets, HTTP, database factories, predictions and acquisition.
Synthetic public CA fixtures cover parsing, missing/unreadable data, file limits,
redaction, ELF association and NOLOAD getters. UI tests exercise the real owner
controls, explicit inspection, rerenders, spent-guard preservation and authorization
loss. The existing preflight also authenticates this helper's loaded source.
Native Linux tests execute the actual mapped-library, driver-getter and fixed-bundle
path in the single authorized PR cycle; Windows results alone are not Linux proof.

The exact successor scope preserves all earlier approval constants, policies and
beforeimages. The existing historical capacity input view receives authenticated
pre-TLS source bytes for changed historical inputs; original test bodies, counts
and frozen expectations are unchanged. Current native reads stay current. The
timing profile, thirteen shards,45-minute limits and120-second reserve remain
unchanged; collection/capacity must account for added tests using the existing costs.
One conditional final-head cycle only; no retries or additional diagnostic jobs.

## Post-merge handoff — not deployment authority

Owner reviews and merges the PR. Reconcile automatic main checks, tested ancestry,
reviewed tree, twenty archives, thirteen manifests, exact identities and capacity.
Only after clearance and owner deployment authority: load that exact merge in the
existing app, with at most a separately authorized controlled restart if needed.
No temporary direct-main hook or separate app. Preserve all secrets and Neon Free.

In Workspace -> Preview & Publish, enter the owner token privately. Select
Show private NFL storage diagnostic; leave guided forecasts off. Click only
Read running NFL build information (no connection). Do NOT click the database
diagnostic, Refresh, acquisition or calculation. Compare observed checkout/tree,
process identity and all scoped loaded-code findings to the newly cleared source.
Retain the actual TLS subreport with its observation clock. Current hosted CA/SSL
facts remain UNKNOWN until this deployed inspection is actually performed.

## Conditional trust decision and rollback — not applied

Consider SSL_CERT_FILE only if actual driver findings demonstrate support for that
selector, its default/override observations support an incorrect or unavailable
trust source, and the fixed public bundle is independently qualified, readable
and parseable. Bundle readability alone is insufficient. Do not infer that an
expired/misissued server chain, hostname issue or every TLS failure is a CA-path bug.
If evidence is inconclusive, retain UNKNOWN and make no trust change.

The candidate exact setting, ONLY after a separate concrete owner setup decision,
is root-level Streamlit secret:

    SSL_CERT_FILE = "/etc/ssl/certs/ca-certificates.crt"

Keep the private database URL unchanged, including sslmode=verify-full and
sslrootcert=system. PostgreSQL documents that system loads the SSL implementation's
roots, whose OpenSSL locations may be influenced by SSL_CERT_FILE/SSL_CERT_DIR.
The setting is process-wide: other clients using the same environment and OpenSSL
default trust selection may also be affected. Clients with an explicit CA argument
can behave differently. It is not a setting isolated to this diagnostic or database.
Do not overwrite an existing override without privately preserving its prior state
and qualifying affected clients offline. Merely saving a secret does not prove the
running process adopted it; follow separately approved restart/build verification.

Rollback: if SSL_CERT_FILE was previously absent, remove only that added entry;
otherwise restore its privately preserved previous value. Preserve every other
secret and both false switches. Reload/restart only within new explicit authority
and verify the actual environment selection through the connection-free preflight.
No database invocation is part of setting/rollback verification. Any subsequent
Neon connection needs a NEW one-use authorization linked to all spent predecessors.

Primary references:
- https://www.postgresql.org/docs/17/libpq-connect.html#LIBPQ-CONNECT-SSLROOTCERT
- https://neon.com/docs/connect/connect-securely
- https://raw.githubusercontent.com/openssl/openssl/openssl-3.0.18/crypto/x509/x509_def.c
- https://man7.org/linux/man-pages/man3/dlopen.3.html
- https://docs.python.org/3/library/ctypes.html

No trust changes, purchases, provider acquisition, authentic inference or writes
are authorized by this engineering work. Saved forecasts and NCAAF evidence remain
unchanged. Hosted TLS diagnosis is separate from persistence and forecast acceptance.
