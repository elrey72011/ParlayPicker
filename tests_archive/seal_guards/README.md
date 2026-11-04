# Retired successor-seal guard tests (archived 2026-11-04)

These 128 modules tested the per-change successor-seal chain:
`scripts/check_launch_change_scope.py`, the `scripts/*_scope.py` guards,
launch-scope policies, capacity profiles and historical reconstructions. They
were retired by owner decision when the guard was re-baselined. See
`docs/roadmap/m0-status-2026-11-04.md`.

- They verify historical seals and pinned hashes, not application behavior. By
  file name they accounted for about 83% of estimated CI time.
- The seals, policies and evidence they checked are unchanged and frozen under
  `docs/paid-launch/`, `docs/research/` and `docs/audits/`. The new
  `scripts/protected_change_guard.py` blocks any modification or deletion there.
- They are kept for audit and are not collected by pytest (`testpaths = tests`).
  They rely on the retired seal chain, so they are not expected to pass against
  later commits. To inspect one against the re-baseline commit:
  `git checkout 8181ff7d063b75985d51871b263425ba2f012861 && python -m pytest tests/<name>.py`.
