"""Explicit operator submission; never imports analysis, Drive, or lock modules."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from services.subscriber.canonical import sign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--endpoint", required=True, help="Subscriber admin release endpoint")
    parser.add_argument("--session-cookie", help="Operator session cookie; prefer environment variable")
    parser.add_argument("--csrf-token", help="CSRF token; prefer environment variable")
    args = parser.parse_args()
    raw = json.loads(args.package.read_text(encoding="utf-8"))
    secret = os.environ.get("PAID_RELEASE_HMAC_SECRET", "")
    session = args.session_cookie or os.environ.get("PAID_OPERATOR_SESSION", "")
    csrf = args.csrf_token or os.environ.get("PAID_OPERATOR_CSRF", "")
    if not secret or not session or not csrf:
        parser.error("release secret, operator session, and CSRF token are required")
    response = httpx.post(
        args.endpoint,
        json={"submission": raw, "signature": sign(raw, secret)},
        cookies={"pp_session": session, "pp_csrf": csrf},
        headers={"X-CSRF-Token": csrf},
        timeout=20,
    )
    print(json.dumps(response.json(), indent=2, sort_keys=True))
    return 0 if response.status_code == 202 else 2


if __name__ == "__main__":
    raise SystemExit(main())
