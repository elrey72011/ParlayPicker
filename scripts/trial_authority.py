"""Owner-operated controlled-trial consent and reservation reconciliation.

Examples (these are commands, not automatic activation):
  python scripts/trial_authority.py status
  python scripts/trial_authority.py grant --owner OWNER --expires-at 2026-10-01T00:00:00Z --confirm
  python scripts/trial_authority.py revoke --owner OWNER --confirm
  python scripts/trial_authority.py release --owner OWNER --reservation-id HASH --confirm

The runtime additionally requires PARLAYPICKER_CONTROLLED_TRIALS_ENABLED=1,
PARLAYPICKER_CONTROLLED_TRIAL_RESERVATION_LEDGER pointing to durable storage,
and a valid configured exposure ledger. This script never places a wager.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_core.trial_authority import (
    CONSENT_PATH_ENV, RESERVATION_PATH_ENV, DEFAULT_CONSENT_PATH,
    DEFAULT_RESERVATION_PATH, consent_status, record_consent,
    release_reservation,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "grant", "revoke", "release"))
    parser.add_argument("--owner", help="Owner identifier for an explicit action")
    parser.add_argument("--expires-at", help="Timezone-aware expiry for a grant")
    parser.add_argument("--reservation-id", help="Exact reservation ID to release")
    parser.add_argument("--confirm", action="store_true", help="Confirm this owner action")
    parser.add_argument("--consent-ledger", default=os.getenv(CONSENT_PATH_ENV, DEFAULT_CONSENT_PATH))
    parser.add_argument("--reservation-ledger", default=os.getenv(RESERVATION_PATH_ENV, DEFAULT_RESERVATION_PATH))
    args = parser.parse_args(argv)
    if args.action == "status":
        _, reason = consent_status(args.consent_ledger)
        print(reason)
        return 0
    if not args.confirm or not args.owner:
        parser.error("--owner and --confirm are required for state changes")
    if args.action == "grant":
        if not args.expires_at:
            parser.error("--expires-at with timezone is required for grant")
        print(record_consent(args.consent_ledger, "GRANTED", args.owner,
                             expires_at=args.expires_at, confirmed=True))
    elif args.action == "revoke":
        print(record_consent(args.consent_ledger, "REVOKED", args.owner, confirmed=True))
    else:
        if not args.reservation_id:
            parser.error("--reservation-id is required for release")
        print(release_reservation(args.reservation_ledger, args.reservation_id,
                                  args.owner, confirmed=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
