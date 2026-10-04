# Versioned MLB home Run Line input contract

`mlb-novig-home-half-runline-asof-v1` binds names, indices, values, exact selected
home line, original price, receipt hash and source dependencies in one replayable
packet. It is an offline research-input contract. No application runtime uses it
to fit, execute inference, register an artifact, publish or authorize a wager.

The readiness V1 report lists the six summaries in a different order from
`exact_feature_values()` → `receipt_features()` → `mlb_research.FEATURES`.
The new schema follows the actual adapter:

| Index | Field |
| ---: | --- |
| 0 | home_ppg |
| 1 | away_ppg |
| 2 | home_oppg |
| 3 | away_oppg |
| 4 | home_win_pct |
| 5 | away_win_pct |
| 6 | exact_line |
| 7 | price_implied_probability |

The first six replay exactly ten retained prior observed finals per team. The
line stays signed for the selected home team, exactly ±1.5. The price input is
`1 / original selected decimal price`; it is market-derived, not independent or
no-vig. Other books, sides, lines and targets fail closed. Provider/team/start
mapping, game number, source checksums, original quote update/observation clocks,
pregame cutoff and all prior feeds must replay. UTC offsets must be explicit.

The retained Odds API event must map to the retained MLB schedule by **ordered**
home/away full-team aliases and Eastern scheduled date, then to the receipt's
official game ID and team IDs. Full MLB aliases preserve shared-city distinctions
(Cubs/White Sox, Yankees/Mets, Dodgers/Angels); bare ambiguous cities are rejected.
Unrelated matchups, swapped sides and conflicting explicit provider claims fail
even when every source and receipt checksum has been consistently recomputed.
More than one matching scheduled game is ambiguous: no nearest-start selection,
claim-based tie-break or removal of an already completed doubleheader game.

The exact retained Novig home outcome, signed line and selected decimal price
identify the market before checking its clock. Its market-level `last_update`
has precedence; the bookmaker's `last_update` is a fallback when the market has
none. A market clock is sufficient when the bookmaker clock is absent. Clocks
from another market/line cannot fill a missing clock; invalid clocks and a
recorded clock differing from that exact source fail closed. Duplicate exact
quotes conflict even when their update clocks differ. Original target-period
declarations are read from that same matched market.

Legacy schemas, vectors, coefficients, hashes and report interpretation are
unchanged. There is no automatic migration or relabeling. Consumers explicitly
select the new schema and call `verify_packet()` with the original receipt and
source observations. That function recomputes the packet and rejects reordered,
renamed, altered or resealed values/identity/authority fields.

Missing full-game or book-rule declarations leave `target_binding.status` UNKNOWN.
Feature verification never supplies those facts from defaults. Requiring a bound
target additionally needs the period in the original quote-provider market,
matching receipt declarations and a captured versioned Novig rule-document
observation with replayable UTF-8 bytes/hash and an original availability clock.
Those captured declarations are not a source-rights approval, authenticated
storage proof or an accepted production reader. Those decisions remain separate.

The target-vector helper accepts COVER/PUSH/NO_COVER with structural zero push for
the explicitly scoped ordinary half-lines. A vector alone does not assert that
inference ran successfully. Every feature packet records NOT_EXECUTED, no
production eligibility, no wager approval and zero stake. Valid negative-EV
research values do not gain authority.

Run `python -m pytest -q tests/test_mlb_home_runline_contract.py` for the
network-blocked harness. It hand-builds synthetic provider sources and uses pure
replay helpers; it performs no collection, recovery, database registration or
fitting. Synthetic cases supply no scientific event count. Private evidence and
the scientific calendar/optimization/uncertainty proposal are excluded from this
software PR.

The successor guard binds the exact starting-main tree, predecessor policy and
guard bytes, implementation blobs, retained protections and reconstructed prior
fixture bytes. A separate policy-only commit seals the implementation. Dirty,
staged, committed, resealed, ancestry, policy and runtime-shadowing changes fail.

This successor starts from actual merged main
`e24e814b6cb08a8d983de615a499035594893830`, the owner merge of #2383. Its schema-8
seal binds the merged NFL policy/guard and preserves the NFL producer, display,
private-retention/download code, policy, workflows and existing estimate fixture
bytes. The only existing fixture edit reconstructs the exact merged NFL source
when testing its predecessor; no assertions change. The top-level home selector
retains the NFL selector beneath it, and invalid home seals cannot fall back to
that earlier authorization. Offline regressions cover nested reconstruction and
dirty/staged/committed/resealed changes to the merged NFL evidence and policy.
