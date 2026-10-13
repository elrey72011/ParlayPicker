# NFL listing advisory metadata successor

The completed shard-3 log at sealed head 2e659169 recorded an assertion failure in the unchanged `test_revision_provenance_and_honest_incomplete_observations`: the listing follow-up changed frozen revision 3 to 4. This successor restores the exact revision-3 fields and records `nfl_listing_cost_revision=1` and `nfl_listing_cost_basis` separately. Every file cost, source receipt, interrupted observation, scheduling input and assertion is unchanged.

The immediate predecessor remains immutable, including its revision-4 record and failed CI. A hash-bound projection restores those exact bytes for predecessor verification. New implementation and policy-only seal bind only this metadata correction and its necessary scope plumbing. No new timing measurements, assignment changes, optimization, acquisition, inference or authority are supplied. The correction saves no claimed test runtime. Full CI remains withheld pending measured capacity evidence.
