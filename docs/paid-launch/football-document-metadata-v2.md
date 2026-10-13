# Exact historical-fixture metadata linkage

CI run 38468308884 on seal 006468faf4c098e62cfaca4ee03c57670d5db28d
completed all five shards and retained their reports. Shard 3 failed the unchanged
`test_historical_fixture_preimages_remain_exact`. A local network-blocked run
reproduced the same mismatch: the live prior reconstruction group named the
NCAAF successor after-image, while the live NFL binding had no entry for that
NCAAF path and the assertion fell back to its historical after-image.

Append only the exact already-bound NCAAF successor reconstruction entries that
are absent from the in-memory NFL view. Preserve all existing entries, base,
previous-guard identities, historical preimages, on-disk bindings and reconstructed
predecessor functions. Return copied metadata; do not cache validation decisions.
Changed source bytes still fail their exact hash/reconstruction gates. The
assertion is not edited. Acquisition, permissions, interpretation, mathematics,
workflows, costs, shard assignments, timeouts and original evidence are unchanged.

The correction supplies no runtime saving, source acceptance or wagering authority.
The preceding failed seal, implementation/seal pairs, logs and artifacts remain
retained. This implementation receives a new exact successor binding and a
subsequent policy-only seal; all required checks are fresh on that final head.
