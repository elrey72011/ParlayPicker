# Bounded, unaccepted NFL odds acquisition

This adjacent lane closes an intentional unsupported scope. The fixed NFL CSV
runner rejects an odds request with `NFL_ACQUIRE_ENDPOINT_NOT_APPROVED`; the
NCAAF transport rejects it with `NCAAF_CUSTODY_SCOPE_CONFLICT` before credential
entry. Those readers, plans, tests and limits remain unchanged. The completed
private scenario, its result and all authentic evidence/journals remain local
and unchanged. No authentic inference or request
is part of this implementation.

## Exact request and limits

`nfl-odds-acquisition-plan-v1` permits only one HTTPS GET to
`https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds`, with exactly:

```
regions=us_ex
bookmakers=novig
markets=spreads
oddsFormat=american
dateFormat=iso
commenceTimeFrom=<explicit UTC instant ending Z>
commenceTimeTo=<explicit UTC instant ending Z>
```

The [official v4 guide](https://the-odds-api.com/liveapi/guides/v4/index.html)
documents the sport endpoint, spreads, American/ISO formats and inclusive UTC
start filters; eventIds is optional. Explicit bookmakers take priority over
regions. The [official bookmaker list](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html)
lists Novig under `us_ex`. One market and one bookmaker cost at most one included
credit; the guide describes empty responses as zero-credit. Neither publication
nor the existence of a bookmaker key proves an applicable operator contract.
Documentation was checked read-only on October 12, 2026; no sports API request
was made.

Exactly one durably journaled attempt; no retry, redirect, pagination, discovery
expansion or fallback. Wire and complete decoded body are each bounded at
512 KiB. Connect/read/shared monotonic deadlines are 10/10/60 seconds. One
original object and at most 2 MiB serialized outcome; at most 128 KiB plan,
configuration and journal, 64 headers/16 KiB headers. No Odds request beyond one
included credit, zero CFBD requests and $0 additional spending. Complete JSON
bytes are retained without filtering or reserialization. Compression expansion,
incomplete bodies and oversized bodies fail; no truncated body is called complete.

## Before and after acquisition

Before dispatch: a verified named target/orientation/kickoff and schedule
reference; advance private acquisition/retention permission; exact owner
collection authorization; an owner-reviewed controlled configuration; an
existing verified encrypted, protected, non-reparse custody directory; and
hash-referenced account evidence with at least one remaining included credit,
known reset/validity clocks and no paid overage. The plan window must be pregame.
Unknown account balances cannot be supplied by the monthly allowance alone.

No provider event ID, future response hash, line, price, listing/rules admission
or completed offer acceptance is required before this unaccepted acquisition.
Robert's recorded advance review is permission for this operation only. Public
[Odds API terms](https://the-odds-api.com/terms-and-conditions.html), edition
August 31, 2026, allow retention and research while restricting standalone raw
data redistribution and requiring private API keys. Account-specific restrictions
and continuing access still require the actual owner's review. No provider
inquiry is sent and no review is manufactured by a control or test.

`nfl-odds-controlled-bindings-v1` contains the exact plan hash, permission,
authorization, custody/storage receipt, account entitlement reference, target
review and actual owner configuration review. The latter hashes all other
entries. Installation is process-local via a ContextVar; production acceptance
catalogs stay empty. Separate exact configuration application and collection
approval remain owner decisions. This PR creates neither authentic receipt.

Immediately before dispatch the attempt is flushed/fsynced to an exclusively
created journal. Authorization and deadline are rechecked after that durable
spend. An interruption can leave a journal without an outcome; that plan is
still spent and cannot resume. A completed valid response with one exact named
target, correct orientation/instant and Novig spreads stops at
`CAPTURED_UNREVIEWED`. Matching is exact, without alias expansion. Missing,
ambiguous, wrong-sport/orientation/kickoff or absent-spread findings are
`INCOMPLETE` with stable reasons; valid original bytes remain retained even when
matching rejects them. Full response custody includes any other returned games.
Malformed or credential-bearing bodies are rejected, not repaired or redacted.

Receipts preserve request/complete-body/first-observation clocks separately from
provider `markets.last_update`. Observation is not publisher publication or
quote renewal. Diagnostics retain only existing allowlisted phase/category,
OS/TLS codes, actually observed HTTP status and elapsed time; unknowns stay
UNKNOWN. Headers, secret-bearing URLs and exception messages are not retained.
Read-back verifies body/bundle/journal hashes and spent request identity.

There is no model call, analysis fetch, SQLite initialization, Drive operation,
offer admission, public export, qualification or stake creation. Outcomes retain
PASS and zero stake. Later verified crosswalk/offer, applicable Novig
listing/product/jurisdiction/period/OT/effective rule edition/FVS evidence and
substantive exact owner review are separate prerequisites for a fresh prospective
offer-backed private inference. Acquisition does not borrow the scenario review.

## Planning and Windows operator template

The committed example is SYNTHETIC/PROPOSAL, has unresolved prerequisites and
cannot execute. Its 2030 descriptors/window are illustrative, not an event
claim or an approval window. Generate a new actual plan only after owner review.
Default planning reads only the local JSON and loads no credentials, inspects
no store, issues no requests and writes no files:

```powershell
python -B scripts/nfl_odds_acquisition.py docs/paid-launch/nfl-odds-example-plan-v1.json
```

Use Robert's normal, non-AppContainer Windows Terminal with the bundled
PowerShell, not the offline pytest process. The following prepared template is
not an execution authorization. Paths and hashes for a future exact plan,
authorization and exclusively applied configuration must be finalized after
separate owner approvals. No secrets belong in those files or command arguments.

```powershell
$taskPython = 'C:\Users\Robert\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$taskLibrary = 'C:\Users\Robert\.codex\visualizations\2026\10\06\01a10f76-29b0-7f70-a89d-e40d3c0375bf\test-runtime'
$taskBootstrap = @'
import http.client, runpy, socket, ssl, sys
assert sys.flags.isolated and sys.dont_write_bytecode
assert not any(k in sys.modules for k in ('sitecustomize','usercustomize','scope_git_reader','pytest','tests.conftest'))
assert socket.create_connection.__module__ == 'socket' and socket.getaddrinfo.__module__ == 'socket'
assert getattr(socket.socket.connect, '__code__', None) is None
original = (socket.create_connection, socket.getaddrinfo, socket.socket.connect, http.client.HTTPSConnection.connect, ssl.SSLContext.wrap_socket)
library, cli, *arguments = sys.argv[1:]
sys.path.append(library)
loaded = runpy.run_path(cli, run_name='verified_explicit_cli_only')
assert original == (socket.create_connection, socket.getaddrinfo, socket.socket.connect, http.client.HTTPSConnection.connect, ssl.SSLContext.wrap_socket)
loaded['verify_runtime']()
raise SystemExit(loaded['main'](arguments))
'@
# Planning is safe now; use the reviewed checkout's absolute CLI/example paths.
& $taskPython -I -B -c $taskBootstrap $taskLibrary "$PWD\scripts\nfl_odds_acquisition.py" "$PWD\docs\paid-launch\nfl-odds-example-plan-v1.json"
# FUTURE template only, after separate authorization and configuration application:
# & $taskPython -I -B -c $taskBootstrap $taskLibrary "$PWD\scripts\nfl_odds_acquisition.py" <EXACT_PLAN_PATH> --execute --authorization <EXACT_AUTHORIZATION_PATH> --custody-root 'C:\Users\Robert\AppData\Local\ParlayPicker\ncaaf-pilot\custody' --controlled-bindings <EXACT_CONFIGURATION_PATH> --bindings-sha256 <VERIFIED_CONFIGURATION_FILE_SHA256>
```

The private masked key prompt occurs only after execution/runtime/configuration
gates and durable attempt spend. Never paste a key into chat, files or shell
arguments. A noninteractive credential entry rejects. Unsupported/instrumented
runtime rejects rather than disabling blockers. Tests keep real
HTTPSConnection and `_timed`, replace only synthetic socket/TLS boundaries, and
separately prove the offline blocker remains active. Isolated runtime/planning
is checked without making a connection.

Expected private report: status/reason, attempted count, original body bytes/hash,
bundle/journal hashes, actual receipt clocks and sanitized diagnostics if failed.
Files are `<plan-sha>.nfl-odds-attempts.jsonl` and
`<plan-sha>.nfl-odds-unreviewed.json` in the existing private custody root.
Stop on any failure, missing target or permission/access change. Do not reuse a
spent/expired plan. No opportunity is scheduled by this template.

## Remaining authentic decisions

1. Verify current included balance, reset and account access; Robert's existing
   personal 20,000-credit subscription and $0 extra ceiling are retained facts,
   not a current balance receipt.
2. Robert reviews the precise private acquisition/retention scope and named
   schedule target; prepare a fresh exact plan and practical window.
3. Robert separately approves collection and then the exact controlled
   configuration/application; recheck storage read-only using STORAGE_PASS and
   current ACL/EFS/no-reparse facts, without repeating setup.
4. Execute once only after a later explicit instruction. Retain unaccepted bytes.
5. Verify the observed crosswalk and offers afterward. Applicable Novig listing,
   product, jurisdiction, full-game/OT, effective rules/overrides and FVS remain
   missing provenance. Generic documentation cannot fill that binding. They do
   not prevent this acquisition capability and do not create wagering authority.

Required final-head CI and local validation receipts are recorded on the draft
PR against its exact implementation/seal. Raw failures/private bodies stay local.
Existing workflows, assertions, protected paths, frozen baseline, three-shard
timing profile and all historical readers are unchanged.

The new leaf reconstructs its exact predecessor by removing only its reviewed
insertions and restoring the dispatch line, then requires the unchanged complete
predecessor SHA-256. Its own normalized source hash is verified first. This avoids
duplicating the large predecessor tail in a new literal. All prior source,
matcher, baseline and policy assertions remain intact; this representation does
not grant a scope exception or establish CI timing/merge readiness.
