# NCAAF private Google Drive backup v1 — proposed local successor

This is a separate, unmerged local preparation route based on
d4b73dcc05b45abb1d57dc995571ce892d4cd5cf. It does not upload, share, restore a live
store, collect providers, infer predictions or register acceptance. Existing
ncaaf-pilot-local-setup-v1 and its local-destination checks are unchanged.

## Exact destination for owner review

My Drive / ParlayPicker NCAAF Private Backups.
The exact owner account, folder URL/ID, parent ID and raw permission receipt
remain in the private owner approval package, excluded from GitHub.

The folder was created empty during preparation. Read-back metadata reports
shared=false and only the owner's user/owner permission; no anyone, group or
domain grant was returned. Its immediate listing returned zero files. No upload,
sharing mutation or access to unrelated Drive content was performed. These are
current folder observations, not an enduring guarantee: inspect complete folder
AND file permissions again at upload/read-back. Ownership supports access, but
destination capacity, account retention/recovery arrangements and provider
permission for future raw-body cloud retention remain unresolved.

Proposed initial uploads, both NEW and never overwritten:
- ncaaf-pilot-initial-v1.ppbackup
- ncaaf-pilot-initial-v1.manifest.json

These are reserved names, not authentic files already generated. Final encrypted
bytes, hashes, byte sizes, complete contained-file manifest and key-recovery
arrangement must be presented with this exact folder for owner upload approval.
Unknown future hashes are not represented as approved. No upload function is
included in the new module or CLI.

## Local setup and backup boundary

The working DB remains local at
C:/Users/Robert/AppData/Local/ParlayPicker/ncaaf-pilot/prediction/evidence.sqlite3;
custody remains its sibling custody directory. Owner-authorized checks now verify
restricted ACLs, no reparse points and folder-level EFS. C: remains decrypted.
The password-encrypted EFS recovery key has separate approved off-PC backup and
isolated key verification; private security receipts remain local. No authentic
working copy has been created at this revision's sealing. Actual data-backup
recovery remains a separate requirement. Whole-volume encryption is unchanged.

ncaaf-pilot-cloud-setup-v2 is explicitly selected through a separate trusted
operator setup binding. It reuses readonly source opening, SQLite Connection.backup,
core content hashes, transactional research_replay.setup and exact immutable
schema checks. Source DB/WAL are never checkpointed, migrated or overwritten.
Its result is LOCAL_PREPARED_CLOUD_BACKUP_PENDING, collection_ready=false.
A same-disk staging copy is local recovery evidence only, not disaster recovery.

The new working copy installs three existing replay tables and six immutable
triggers. Original six-table contents and restart read-back must match. The
original v1 reader/setup, packet readers, model artifacts, formulas, clocks,
source/acceptance catalogs and wagering gates remain unchanged. Production
backup-setup/read-back configuration is empty; CLI uploads cannot populate it.

## Whole-bundle consistency and encryption

Before snapshotting, the operator must stop every local writer and retain the
coordination receipt. A fresh byte inventory of all custody files and journals
before/after snapshots detects observed changes; it does not prove that an
uncoordinated writer never ran. SQLite alone cannot make filesystem journals and
a database atomic together. Concurrent backup is not admitted by this contract.

The encrypted archive includes:
- consistent pre-migration and post-migration recovery snapshots;
- a fresh working database snapshot, including committed WAL contents;
- every custody file, including discovery/capture bundles and spent journals;
- original setup receipt and setup-attempt journal;
- an encrypted internal manifest of original relative paths, byte counts and hashes.

The coherent temporary database copies are created only in verified private,
encrypted local storage. SQLite changes journal_mode=DELETE on the COPIES only.
This avoids a reproduced WAL-header portability failure with memory serialization.
Source clocks, rows and original database bytes are preserved. Only successful
generated temporary copies are removed; failed copies/journals remain for review.

AES-256-GCM authenticates the complete archive and encryption header. A random
16-byte salt and 12-byte nonce are generated per package; Scrypt n=32768,r=8,p=1
derives the key from an owner-held recovery passphrase. Existing cryptography
dependency supplies these implementations. No custom cryptographic primitive,
password in CLI arguments, plaintext key file, credential-bearing receipt or
cloud key upload is introduced. The outer manifest contains ciphertext identity
only; private file paths/counts/hashes are inside encryption. Secrets/config from
outside approved custody are excluded; detected credential files/echoes reject
the package rather than stripping original evidence.

The owner must keep the recovery secret in a password manager or other agreed
recovery arrangement accessible after loss of this PC, separate from this Drive
backup folder. A Windows-only DPAPI key on this computer does not satisfy that
requirement. No hardware purchase is required. Key custody and recovery access
need explicit evidence, not a passing fixture.

Backup-specific ceilings are 64 MiB uncompressed archive and 512 entries.
They do not change provider body/object/request limits. Oversize stops; no
truncation, silent omission or automatic widening. Existing files are never
overwritten. Incomplete spent-attempt journals are preserved byte-for-byte.

## Cloud read-back and isolated recovery

After separate approval, use the connected owner Drive capability to upload only
the two approved ciphertext/manifest files. Existing Shared-Drive-only DriveStore
is not relaxed into a personal-Drive client. Existing generic JSON sync is not
called or counted as full pilot backup: it omits raw custody and spent journals.
Reuse its hash/identity/read-back principles without modifying protected modules.

The controlled AUTHORIZED_READBACKS binding identifies the exact folder/owner,
permitted users, NEW remote file IDs/names and immutable outgoing hashes. A
trusted reader must fetch fresh complete metadata and media of those exact IDs,
bounded to the declared size. Do not qualify a mirrored G:/H: cache as cloud
read-back. Missing ACL evidence, public/domain/group grants, unexpected parents,
identity/hash conflicts, download interruption or timeout reject; no retry/upload
is hidden in verification. Current production bindings remain empty.

After authenticated cloud download and ciphertext comparison, decrypt and check
every internal file hash, database integrity/FKs/schema and immutable replay SQL.
Recover into a NEW private encrypted location, e.g.
ncaaf-pilot/recovery-drills/ncaaf-pilot-initial-v1-isolated.
Never overwrite source/working/custody files, replace a spent journal, reset
attempts or configure the recovered copy as an operational runner store.
Byte-readback must preserve every journal and custody file. The drill remains
non-operational and unaccepted.

Local decryption/recovery alone reports cloud_origin_verified=false. Only fresh
authenticated remote read-back plus an isolated recovery drill can report
CLOUD_READBACK_AND_ISOLATED_RECOVERY_VERIFIED. Backup verification creates no
source acceptance, model qualification, collection authorization or wagering
authority. Unsupported/unqualified selections remain PASS at zero stake.

## Review and execution boundaries

This proposal permits local software preparation and synthetic verification.
It is not upload authorization or permission to activate deployment. Before any
real working copy: verify local encryption, scope/source hashes and explicit
operator setup binding. Before cloud upload: resolve key recovery, original-byte
cloud-retention rights, exact actual outgoing manifest/hashes, folder/file ACLs
and owner approval. Before real collection: independently close all existing
pilot source/event/history/authorization gates; none are closed by backup.

Official references:
- [Google Drive sharing and permissions](https://developers.google.com/workspace/drive/api/guides/manage-sharing)
- [Cryptography authenticated encryption](https://cryptography.io/en/latest/hazmat/primitives/aead/)

No authentic backup or real cloud recovery has been demonstrated yet.

## CI closure and preserved failures

The initial sealed head f6b98e730d80744f4073632b56471ee676e90069 failed shard 3
in run 38075071636. Two new scope tests attempted to read base history absent
from the application's shallow checkout. They now compare exact reviewed
baseline SHA-256 and Git blob identities without fetching history. The separate
full-history protected-scope check continues to verify every binding against
the actual base.

Four unchanged pilot-fixture cases also fail against an archived copy of exact
main d4b73dcc05b45abb1d57dc995571ce892d4cd5cf: the fixture's kickoff is
2026-10-10T12:00:00Z, but candidate selection used the real later clock and
correctly removed the started game. Only this labelled synthetic fixture's
candidate_chronology.now_utc is now aligned with its existing synthetic caller
and export instant. Its assertions are preserved byte-for-byte after removing
that exact fixture addition. No production clock, freshness/start rule, authentic
record, formula, protected path or workflow changes. Unchanged stale/future and
started-game regressions are required alongside the full-path fixture.

Raw logs/XML/assignment manifests and each local failure remain separately
retained. Original failed CI is not described as passing after correction.

The second head 236c5a8832f6ebb83104787465ddfd89022bc78c exposed a scope-report
classification error in subscriber-postgres run 38076961060 (22 passed, one
failed). Inherited `existing_test_changes` means unapproved test changes; the
approved, byte-exact fixture addition now appears separately under
`approved_fixture_clock_changes`. Its exact reconstruction check is unchanged,
and a new regression checks the report distinction. The original subscriber
scope assertion and all paid-launch workflows are preserved. This is not an
infrastructure failure or a passing original suite.

