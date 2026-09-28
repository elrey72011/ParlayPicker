"""Canonical serialization and digests for immutable service records."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_hash(value: Any) -> str:
    return sha256_hex(canonical_bytes(value))


def sign(value: Any, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_bytes(value), hashlib.sha256).hexdigest()


def verify_signature(value: Any, signature: str, secret: str) -> bool:
    return hmac.compare_digest(sign(value, secret), signature.strip().lower())
