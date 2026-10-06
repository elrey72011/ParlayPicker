# Retained football catalog correction

The October 6 Stage 1 run `37430661515` failed its requested slate gate because
two quote-due NCAAF games had no verified pregame offer. All 19 HTTP requests
succeeded; restore and backup readback verified. A successful response is not
proof of complete provider coverage.

NFL retained 31 scheduled targets, 15 quote-due games and 15 current matched,
valid priced spread/total events. The other 16 games were outside the capture
horizon. NCAAF retained 583 scheduled rows: 116 FBS-versus-FBS targets and 467
policy exclusions. Of 58 quote-due targets, 56 matched and had valid prices.
The 59 returned odds events also contained the two rejected comparisons and
one Ohio State–Indiana event outside the schedule query window.

Miami (OH)–Massachusetts, `ncaaf:cfbd:401866440`, first failed identity matching.
The provider offered “UMass Minutemen”; CFBD's retained original catalog supplies
`alternateNames: ["UMass", "MASS", "UMass"]`. The actual caller read only the
legacy `alt_name` field. A September 24 catalog receipt reproduces the omission
independently; it is not represented as the missing October 6 original response.
The fixture records the original source SHA-256 and preserves the projected
fields exactly. Event/offer price fixtures are explicitly synthetic.

The correction reads the catalog's array of alternate names, retains legacy
names, and combines each explicit name with its supplied mascot. The existing
whole-catalog normalized-name owner check still rejects collisions. Canonical
event IDs, strict named orientation, 60-second kickoff tolerance, request count,
timestamps, prices, horizons and historical rows retain their existing rules.
Consumed aliases remain bound by the existing quote identity mapping hash and
the original complete catalog is retained in source receipts. No fuzzy matching
or new provider alias is introduced.

Hawaiʻi–Arizona State, `ncaaf:cfbd:401856808`, remains unverified. The retained
catalog does not establish the provider spelling “Hawaii Rainbow Warriors” as
an accepted alias. The possible provider comparison also starts one hour later
than the schedule, 02:30 versus 01:30 UTC October 11. Neither identity nor kickoff
is repaired by assumption. It stays excluded until source mapping and the
original revision facts are independently reviewed. Rejected diagnostic price
flags reflect skipped price evaluation, so their actual price validity is UNKNOWN.
Original quote-observation clocks and complete original response dependencies
are absent from sanitized artifacts; no inference clock is reconstructed.

Stage 1 produces no model candidates, displayed estimates or qualified selections.
Those counts are zero for this caller; the separate board population is UNKNOWN.
The complete private event-level matrix preserves every retained schedule row,
provider-only event and first-stage distinction.

The interrupted local canonical cache contains 2,084 NFL quote rows across 32
games (1,038 spreads, 1,046 totals; 674 half-point spread rows), and 1,626
score-derived settlement rows across 16 games. It contains no original predictor
packets or NHL rows. The local prediction database has zero snapshots; hosted
storage access and its effective directory remain UNKNOWN. Score-derived
settlements are not independently reviewed exact operator/product settlements.
There are zero demonstrated observations for every proposed scientific role.
These inventories overlap and are not summed. The cache remains BLOCKED/partial;
no acquisition or recovery is performed.

The merged NFL builder and explicit NHL selected-side full-game ±1.5 adapter are
preserved. No retained authentic NHL model/input packet supports additional wiring.
Goalie information stays explicitly missing under the current seven-feature
contract. NHL totals require a separate target; future default-board integration
requires separate review and prerequisites. All unqualified selections remain
PASS with zero stake. Synthetic path acceptance and CI provide no qualification.

New actual-cycle regressions cover the retained alternate-name projection,
malformed arrays, collision ownership, unchanged healthy NFL rows, repeat capture,
reversed teams, wrong sport, ambiguous games, kickoff revisions, bad prices and
future/missing clocks. All provider reads and remote sync are fixture replacements;
the complete run and collection execute with network blocked. The exact schema-18
successor preserves the baseline and every predecessor and binds this minimal
implementation to verified merged main, followed by a policy-only seal.

Provider inquiries, the raw-model calibration-stage proposal, original input
access, dependency/rights review, predictor training/out-of-sample lineage and
independent untouched custody remain separate unregistered, unfitted decisions.
Workflows, source registration, activation, fitting, wagering, deployment and
merging are not performed.
