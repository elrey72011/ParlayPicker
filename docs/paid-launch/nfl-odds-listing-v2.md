# Unaccepted Novig source identifiers and links

This explicitly selected successor is acquisition only. `nfl-odds-acquisition-plan-v1`
and its rejection of extra parameters remain intact. No saved scenario, offer
reader, source acceptance, scientific qualification or wagering gate changes.

## Request contract

`nfl-odds-acquisition-plan-v2` keeps the exact NFL/Novig spreads endpoint and v1
parameters, adding only `includeSids` and/or `includeLinks`. At least one of the
two must be explicitly present; each value must be the string `true` or `false`.
Single flags and both flags, including explicit false values, are supported.
There are no implicit defaults or additional optional parameters. Booleans,
uppercase values, arbitrary parameters, changed endpoints/hosts and other sports
reject. The transport defaults to v1 and receives v2 only from the validated
selected plan. The actual HTTPS query serializes the exact reviewed strings.

Credential-free before/after examples (illustrative window, not an opportunity):

```text
v1: GET /v4/sports/americanfootball_nfl/odds?regions=us_ex&bookmakers=novig&markets=spreads&oddsFormat=american&dateFormat=iso&commenceTimeFrom=2030-10-20T20%3A00%3A00Z&commenceTimeTo=2030-10-20T20%3A00%3A01Z
v2: same request plus &includeSids=true&includeLinks=true
```

The [official V4 guide](https://the-odds-api.com/liveapi/guides/v4/) documents
bookmaker event/market/betslip links and source event/market/outcome IDs **if
available**. It does not establish that Novig supplies them. No request was made
to test live availability. The existing one-market/one-bookmaker cost formula
still caps the proposal at one included Odds credit; flags do not add a market
or bookmaker. Body-size feasibility remains unknown.

## Exact approvals and unchanged bounds

The complete request, including each present flag and its value, is hashed in
the permission receipt and plan. `nfl-odds-acquisition-authorization-v2` binds that
exact plan. `nfl-odds-controlled-bindings-v2` binds the plan, version-matched
authorization, advance permission, existing custody/storage receipt, account
entitlement, target review and actual owner configuration review. Changing a
flag requires new exact bindings/review; an existing approval does not carry
forward. V1 cannot borrow v2 authorization/configuration or vice versa.
Installation remains process-local. Production acceptance catalogs stay empty.

All original bounds remain: one durably spent HTTP attempt, one object, 512 KiB
wire and decoded body, 2 MiB serialized outcome, 128 KiB plan/configuration/journal,
64 headers/16 KiB headers, 10/10/60-second deadlines, at most one included Odds
credit, zero CFBD calls and $0 additional spending. No retries, redirects,
pagination, fallback or response-link requests. Oversize/incomplete/credential-
bearing bodies reject; no original is truncated, repaired or reserialized.

## Private custody and reference pointers

The successor outcome is `nfl-unaccepted-odds-body-v2`. It retains the complete
decoded original body, wire/body/bundle/journal hashes and actual receipt clocks
through the unchanged original-response custody mechanism. Other returned games
remain in those original bytes. Provider market observation remains distinct
from local receipt and unavailable original operator-update clocks.

For the matched event, v2 findings add private `listing_references` at event,
bookmaker, market and outcome levels. Each contains the exact original body hash,
JSON pointer, optional-field names/pointers and `trusted=false`. The metadata
schema is deliberately not interpreted as a contract. An empty optional-field
set is `UNKNOWN`; present fields are `PRESENT_UNREVIEWED`. Missing, malformed,
conflicting or malicious IDs/links cannot establish acceptance. Original values
remain accessible at their pointers without copying or normalizing URLs. Static
read-back recomputes these pointers and rejects rehashed derivative substitution.

No link is followed, constructed, used for credentials or treated as fetch
authorization. Raw response credential echoes still reject rather than leak
into custody or diagnostics. Execution reporting does not print these URLs.

Successful acquisition stops at `CAPTURED_UNREVIEWED`. Missing/ambiguous target
or absent Novig spreads remains incomplete with original valid bytes retained.
All outcomes remain unaccepted, nonpublic, unqualified, PASS and zero stake.

## Non-executable operator handoff

The committed example is an explicitly labelled synthetic PROPOSAL with unresolved
permissions and authorization, not a genuine event or an expiring approval window.
Planning reads no credentials, touches no store and makes no request:

```powershell
python -B scripts/nfl_odds_acquisition.py docs/paid-launch/nfl-odds-example-plan-v2.json
```

For future execution use Robert's corrected isolated, non-AppContainer runtime
and the complete bootstrap in [the unchanged v1 operator instructions](nfl-odds-acquisition-v1.md#planning-and-windows-operator-template).
Pass the reviewed v2 plan to the same CLI; it selects the matching transport.
After distinct exact collection and configuration-application approvals:

```powershell
# TEMPLATE ONLY: no execution is authorized by this document.
# & $taskPython -I -B -c $taskBootstrap $taskLibrary <REVIEWED_ABSOLUTE_CLI_PATH> <EXACT_V2_PLAN_PATH> --execute --authorization <EXACT_V2_AUTHORIZATION_PATH> --custody-root 'C:\Users\Robert\AppData\Local\ParlayPicker\ncaaf-pilot\custody' --controlled-bindings <EXACT_V2_CONFIGURATION_PATH> --bindings-sha256 <VERIFIED_CONFIGURATION_FILE_SHA256>
```

Reuse STORAGE_PASS, protections and original receipts; recheck only necessary
read-only gates and account balance/expiry. API key entry remains private and
masked after runtime/configuration checks and durable attempt spend. No keys in
chat, commands, evidence or logs. Test network blockers remain enabled; an
instrumented test runtime is not a supported acquisition runtime.

## Later mapping and edition review

Novig's [Market Rules](https://support.novig.com/en/articles/9612523-market-rules)
closes the general governing framework: contracts govern its platform trades.
Returned IDs might identify the exact event/market/position; verified listing
documentation might establish reference team, comparison/threshold, YES/NO,
declared period/OT and overrides. A listing's contract/edition and applicable
amendments must cover quote observation and genuinely new inference. An opaque
ID, generic homepage or file modification date is insufficient. Novig-specific
field availability and exact applicability stay UNKNOWN until observed/reviewed.

Only if attributable listing evidence fails to establish those facts is a narrow
provider/operator clarification necessary. No new license purchase or generic
rules inquiry is inferred. This PR does not implement the separately proposed
semantic source-projection successor, acquire a fresh quote, admit a listing or
authorize inference. The saved Rams +1.5 scenario remains a distinct hypothetical
result with its original probability and clocks.
