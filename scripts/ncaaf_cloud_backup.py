"""Network-free cloud backup preparation. No upload or acceptance registration."""
import argparse
import getpass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core import ncaaf_cloud_backup as cloud


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--setup',action='store_true')
    mode.add_argument('--prepare',action='store_true')
    p.add_argument('--source',type=Path)
    p.add_argument('--authorization-ref')
    p.add_argument('--backup-id')
    p.add_argument('--writer-stop-reference')
    p.add_argument('--key-recovery-reference')
    a=p.parse_args(argv)
    if a.setup:
        if not a.source or not a.authorization_ref: p.error('setup requires source and controlled authorization reference')
        result=cloud.setup(a.source,a.root,authorization_ref=a.authorization_ref)
    elif a.prepare:
        if not all((a.backup_id,a.writer_stop_reference,a.key_recovery_reference)):
            p.error('prepare requires backup ID, writer-stop reference and key-recovery reference')
        cloud.private(a.root)
        if not (a.root/'setup-receipt.json').is_file(): p.error('existing cloud-v2 local setup is required')
        # Interactive secret only: never command-line/env/chat/receipt/key file.
        first=getpass.getpass('Owner vault recovery passphrase: ').encode()
        second=getpass.getpass('Confirm recovery passphrase: ').encode()
        if first!=second: p.error('passphrases do not match')
        result=cloud.prepare(a.root,a.root,backup_id=a.backup_id,secret=first,
            writer_stop_reference=a.writer_stop_reference,recovery_reference=a.key_recovery_reference)
    else:
        result=dict(version=cloud.VERSION,root=str(a.root.resolve()),
            security=cloud.local.inspect_security(a.root),
            status='ASSESSMENT_ONLY_CLOUD_NOT_VERIFIED',uploads=0,provider_requests=0,
            cloud_readback_verified=False,recovery_verified=False)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())

