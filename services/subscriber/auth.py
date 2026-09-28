"""OIDC PKCE and opaque server-side session helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import secrets
from typing import Any
from urllib.parse import urlencode
import uuid

from authlib.jose import JsonWebKey, jwt
import httpx

from .db import Database, new_id
from .settings import Settings


def token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token(size: int = 32) -> str:
    return secrets.token_urlsafe(size)


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@dataclass(frozen=True, slots=True)
class AuthenticatedCustomer:
    customer_id: uuid.UUID
    role: str
    auth_strength: str
    contact_email: str | None
    email_verified: bool
    session_id: uuid.UUID


class OidcClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        self.settings = settings
        self.http = http

    async def _get(self, url: str) -> dict[str, Any]:
        if self.http is not None:
            response = await self.http.get(url)
        else:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url)
        response.raise_for_status()
        return response.json()

    async def metadata(self) -> dict[str, Any]:
        metadata = await self._get(self.settings.oidc_issuer + "/.well-known/openid-configuration")
        if metadata.get("issuer") != self.settings.oidc_issuer:
            raise ValueError("OIDC_ISSUER_MISMATCH")
        return metadata

    def _trusted_endpoint(self, value: object, label: str) -> str:
        endpoint = str(value or "")
        expected = httpx.URL(self.settings.oidc_issuer)
        actual = httpx.URL(endpoint)
        if actual.scheme != expected.scheme or actual.host != expected.host or actual.port != expected.port:
            raise ValueError(f"OIDC_{label}_ENDPOINT_INVALID")
        return endpoint

    async def begin(self, db: Database, redirect_after: str) -> str:
        if redirect_after not in self.settings.allowed_return_urls:
            raise ValueError("OIDC_RETURN_URL_NOT_ALLOWED")
        metadata = await self.metadata()
        authorization_endpoint = self._trusted_endpoint(metadata.get("authorization_endpoint"), "AUTHORIZATION")
        state, nonce, verifier = _token(), _token(), _token(48)
        expires = datetime.now(timezone.utc) + timedelta(minutes=10)
        db.execute(
            "INSERT INTO subscriber.oidc_login(state_hash,nonce,code_verifier,redirect_after,expires_at) VALUES (%s,%s,%s,%s,%s)",
            (token_hash(state), nonce, verifier, redirect_after, expires),
        )
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.settings.oidc_client_id,
                "redirect_uri": self.settings.oidc_callback_url,
                "scope": "openid profile email",
                "state": state,
                "nonce": nonce,
                "code_challenge": _challenge(verifier),
                "code_challenge_method": "S256",
            }
        )
        return authorization_endpoint + "?" + query

    async def complete(self, db: Database, *, state: str, code: str) -> tuple[str, str, str]:
        with db.transaction() as connection:
            login = connection.execute(
                "SELECT * FROM subscriber.oidc_login WHERE state_hash=%s FOR UPDATE",
                (token_hash(state),),
            ).fetchone()
            now = datetime.now(timezone.utc)
            if not login or login["consumed_at"] or login["expires_at"] <= now:
                raise ValueError("OIDC_STATE_INVALID_OR_EXPIRED")
            connection.execute(
                "UPDATE subscriber.oidc_login SET consumed_at=%s WHERE state_hash=%s",
                (now, token_hash(state)),
            )
        metadata = await self.metadata()
        token_endpoint = self._trusted_endpoint(metadata.get("token_endpoint"), "TOKEN")
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.settings.oidc_callback_url,
            "client_id": self.settings.oidc_client_id,
            "client_secret": self.settings.oidc_client_secret,
            "code_verifier": login["code_verifier"],
        }
        if self.http is not None:
            response = await self.http.post(token_endpoint, data=form)
        else:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(token_endpoint, data=form)
        response.raise_for_status()
        token = response.json()
        id_token = token.get("id_token")
        if not isinstance(id_token, str):
            raise ValueError("OIDC_ID_TOKEN_MISSING")
        jwks_uri = self._trusted_endpoint(metadata.get("jwks_uri"), "JWKS")
        jwks = JsonWebKey.import_key_set(await self._get(jwks_uri))
        claims = jwt.decode(
            id_token,
            jwks,
            claims_options={
                "iss": {"essential": True, "value": self.settings.oidc_issuer},
                "aud": {"essential": True, "value": self.settings.oidc_client_id},
                "exp": {"essential": True},
                "nonce": {"essential": True, "value": login["nonce"]},
                "sub": {"essential": True},
            },
        )
        claims.validate(leeway=30)
        subject = str(claims["sub"])
        email = str(claims.get("email", "")).strip().lower() or None
        verified = claims.get("email_verified") is True
        groups = claims.get("groups", [])
        amr = set(claims.get("amr", []))
        is_admin = self.settings.oidc_admin_group in groups and bool(amr & {"mfa", "otp", "hwk"})
        role = "OWNER" if is_admin else "CUSTOMER"
        auth_strength = "mfa" if is_admin else "oidc"
        session_token, csrf_token = create_customer_session(
            db,
            issuer=self.settings.oidc_issuer,
            subject=subject,
            email=email,
            email_verified=verified,
            role=role,
            auth_strength=auth_strength,
            hours=self.settings.session_hours,
        )
        return session_token, csrf_token, login["redirect_after"]


def create_customer_session(
    db: Database,
    *,
    issuer: str,
    subject: str,
    email: str | None,
    email_verified: bool,
    role: str,
    auth_strength: str,
    hours: int,
) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    raw_token, raw_csrf = _token(48), _token(32)
    with db.transaction() as connection:
        customer = connection.execute(
            """
            INSERT INTO subscriber.customer(id,oidc_issuer,oidc_subject,contact_email,email_verified,status,role)
            VALUES (%s,%s,%s,%s,%s,'ACTIVE',%s)
            ON CONFLICT (oidc_issuer,oidc_subject) DO UPDATE SET
              contact_email=CASE WHEN excluded.email_verified THEN excluded.contact_email ELSE subscriber.customer.contact_email END,
              email_verified=subscriber.customer.email_verified OR excluded.email_verified,
              updated_at=now()
            RETURNING id,status,role
            """,
            (new_id(), issuer, subject, email, email_verified, role),
        ).fetchone()
        if customer["status"] != "ACTIVE":
            raise ValueError("CUSTOMER_DISABLED")
        connection.execute(
            "INSERT INTO subscriber.session(id,token_hash,csrf_hash,customer_id,auth_strength,expires_at) VALUES (%s,%s,%s,%s,%s,%s)",
            (new_id(), token_hash(raw_token), token_hash(raw_csrf), customer["id"], auth_strength, now + timedelta(hours=hours)),
        )
    return raw_token, raw_csrf


def authenticate(db: Database, raw_token: str | None) -> AuthenticatedCustomer | None:
    if not raw_token:
        return None
    row = db.fetch_one(
        """
        SELECT s.id AS session_id,s.auth_strength,c.id AS customer_id,c.role,c.contact_email,c.email_verified
        FROM subscriber.session s JOIN subscriber.customer c ON c.id=s.customer_id
        WHERE s.token_hash=%s AND s.revoked_at IS NULL AND s.expires_at>now() AND c.status='ACTIVE'
        """,
        (token_hash(raw_token),),
    )
    if not row:
        return None
    return AuthenticatedCustomer(**row)


def validate_csrf(db: Database, session_id: uuid.UUID, token: str | None) -> bool:
    if not token:
        return False
    row = db.fetch_one("SELECT csrf_hash FROM subscriber.session WHERE id=%s AND revoked_at IS NULL", (session_id,))
    return bool(row and secrets.compare_digest(row["csrf_hash"], token_hash(token)))


def revoke_session(db: Database, session_id: uuid.UUID) -> None:
    db.execute("UPDATE subscriber.session SET revoked_at=now() WHERE id=%s AND revoked_at IS NULL", (session_id,))
