# Transport-mirror acceptance evidence

All tests are synthetic, deny real sockets, and preserve the prior 82 assertions. The offline runner collects 97 cases. Windows runs all 97; Linux skips only the existing process-tree case and two real Windows sharing cases. Execution results and measurements must be retained from the final PR revision.

| ID | Executable evidence |
|---|---|
| M01 | Private before/after preservation register; application pin/tree; actual-main and overlap inspection; prior v2 fixture hash |
| M02 | test_M02_actual_old_writer_prefix_gap: actual baseline writer produces one-event lag; baseline loader rejects |
| M03 | test_M03_local_retry_no_charge_replay and test_M03_windows_subprocess_delete_sharing_release |
| M04 | persistent Windows lock, injected sharing bound, and access-denial/nonsharing faults |
| M05 | append/open/truncation, journal fsync, temp open/flush, replacement-success-then-error tests |
| M06 | test_M06_multithreaded_writer_and_short_observers: 320 updates, exact journal, valid old/new observations |
| M07 | full_generation_chain: new-only projection, one-byte preservation, immutable two predecessors |
| M08 | test_M08_mirror_and_journal_fault_rules: ten invalid mirror/journal classes fail closed |
| M09 | full_generation_chain: actual writer bytes, 15,032 accepted / 1,879 ledgers, bad payload and effective spec rejection |
| M10 | full_generation_chain: 22/40 OAuth; five initiated slices; three maximum successors; 3,695.718 elapsed; disk ancestry |
| M11 | first_error_survives_mirror_loading_and_diagnostic_write_failure: structured OS/retry fields and stderr fallback |
| M12 | full_generation_chain: actual supervised fake-transport capture, accepted SQLite, unchanged read-only input hash, blocked empty scientific tables |
| M13 | prior 82 cases plus 15 new; actual auth lifecycle; zero sockets; explicit platform skips |
| M14 | scoped PR, final checks/artifact hashes, public exclusion review, private preservation, one UNAPPROVED continuation |

The 15,032-shape fixture builds canonical bytes through the existing application writer/codec and uses actual wrapper seal/state/journal writers for inherited batches. It is not a reproduction of real provider throughput. Its successor runs the real supervisor and children with actual Google Auth lifecycle at a fake HTTP boundary; eight synthetic missing objects complete capture/import/assessment. No scientific row or production authority is manufactured. Models/calibrations/predictions/validation/review rows remain absent, and readiness stays blocked.

Historical diagnosis remains limited: PermissionError and retained journal/mirror state are known; the original OS operation and lock owner were not recorded. The actual Windows test proves a supported contention class, not attribution of that incident.
