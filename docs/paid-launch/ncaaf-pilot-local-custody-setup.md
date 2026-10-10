# Isolated NCAAF local setup v1

`app_core/ncaaf_pilot_setup.py` provides read-only assessment and explicitly
authorized isolated setup. It never invokes providers, Drive, prediction,
acceptance registration or deployment. Default CLI is assessment:

```powershell
python scripts/ncaaf_pilot_setup.py --source <existing-evidence.sqlite3> --root <secured-private-root> --backup-root <secured-separate-backup-root>
```

The proposed Windows root is
`C:\Users\Robert\AppData\Local\ParlayPicker\ncaaf-pilot`, with `custody`,
`prediction\evidence.sqlite3` and `backups` children. This path is private only
after demonstrated restricted access and encryption; AppData is not proof.
OS inspection checks protected DACL/owner, grants restricted to current custodian,
SYSTEM and administrators, no reparse ancestors, and EFS encryption or fully
encrypted/protected BitLocker. Inaccessible information is UNKNOWN and blocks
copying. The separate backup location must independently pass those checks.
The custodian must establish backup media, lawful access, encryption/key recovery
and retention arrangements. Software close/reopen proves committed read-back,
not hardware durability or backup recovery across a destroyed device.

The October preparation check found an inherited sandbox-group read grant and
unencrypted EFS directory; BitLocker information was inaccessible even in the
owner-context read-only probe. No authentic store was copied, and no encryption
or ACL was silently changed. Actual local setup remains BLOCKED. A separate
encrypted offline backup arrangement has not been established.

After security and backup prerequisites are established, controlled
`AUTHORIZED_SETUPS` configuration must bind source/root/backup paths and setup
version. No CLI upload registers it. Explicit `--execute --authorization-ref`
performs setup only; missing/altered approval rejects before opening source.
For this task copying was authorized conditionally, not permission to bypass
unverified security. Production setup configuration remains empty.

## Coherent copy and migration

1. Coordinate stopped source writers and preserve original DB/WAL/SHM evidence.
   Existing source is the desktop prediction database, not canonical prospective
   SQLite or NHL market storage. Never initialize an absent source.
2. Open source SQLite `mode=ro`, query-only, and use `Connection.backup` to a NEW
   exclusive pre-migration snapshot. It includes committed WAL content. Do not
   checkpoint/migrate/replace source, use `immutable=1` on an active WAL database,
   or copy the main file alone. Derive the working copy from this same snapshot.
3. On the working copy only, enable foreign keys, begin a transaction, and call
   existing `research_replay.setup(db)`. It adds `research_replay_sources`,
   `research_replay_exports`, `research_source_intakes` and six immutable triggers.
   Exact table/trigger SQL is compared with existing setup output; names alone
   cannot conceal weakened triggers. No source-intake acceptance is inserted.
4. Verify quick_check, foreign_key_check, exact original six-table schema,
   record counts/content identities. Commit only after preservation verification.
   An exception rolls back all DDL, retaining pre-migration recovery evidence.
5. Close/reopen the working DB and verify commit/schema/content. Retain coherent
   post-migration backup, byte-verified separate pre/post copies and a private
   setup manifest. Receipts contain counts/hashes/security only, not raw rows.

The original source remains unchanged. The generic prediction initializer is
not used for inspection/migration. The canonical UI exporter has a different
store contract and is not repurposed. Synthetic tests exercise committed-WAL
copy, transactional failure, restart read-back and UPDATE/DELETE rejection.

## Reruns, custody and rollback

An exclusive fsynced `setup-attempt.jsonl` precedes child creation. Existing
stores, backups, custody directories and journals are never overwritten.
Interrupted/failed setup requires manual recovery review; no automatic rerun.
Before authentic additions, select a verified pre-migration recovery copy only
under explicit incident review, retaining failed copies. After ANY capture,
never restore an older copy that erases new evidence or resets spent attempts.
Stop writers, preserve full current backups/journals and repair append-only.
The setup API intentionally provides no destructive rollback operation.

Database backup does not back up future raw bodies or spent-attempt journals.
Owner custody must back up those immutable files with byte manifests and tested
recovery before real execution. No Drive/cloud synchronization is added here.
The existing pilot supports explicit `retain_analysis(path=working_store)`;
there is no hosted environment change. Collection, later verification/acceptance,
fresh inference and hosted display verification need separate authority.
