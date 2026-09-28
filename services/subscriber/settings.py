"""Fail-closed configuration for the subscriber service."""

from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlsplit


def _required(env: dict[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _bool(env: dict[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _csv(env: dict[str, str], name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in env.get(name, "").split(",") if item.strip())


def _https_url(value: str, name: str, *, allow_http: bool) -> str:
    parsed = urlsplit(value)
    allowed = {"https"} | ({"http"} if allow_http else set())
    if parsed.scheme not in allowed or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{name} must be an allowed absolute URL")
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    environment: str
    public_base_url: str
    allowed_origins: tuple[str, ...]
    allowed_return_urls: tuple[str, ...]
    oidc_issuer: str
    oidc_client_id: str
    oidc_client_secret: str
    oidc_callback_url: str
    oidc_admin_group: str
    release_hmac_secret: str
    gateway_probe_token: str
    stripe_secret_key: str
    stripe_webhook_secret: str
    stripe_account_id: str
    stripe_price_id: str
    live_billing_enabled: bool
    sales_enabled: bool
    secure_cookies: bool
    session_hours: int
    email_from: str
    smtp_url: str
    release_probe_url: str
    source_revision: str

    @classmethod
    def from_env(cls, source: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if source is None else source)
        environment = env.get("PAID_ENVIRONMENT", "local").strip().lower()
        if environment not in {"local", "test", "staging", "production"}:
            raise ValueError("PAID_ENVIRONMENT must be local, test, staging, or production")
        allow_http = environment in {"local", "test"}
        public_base = _https_url(_required(env, "PAID_PUBLIC_BASE_URL"), "PAID_PUBLIC_BASE_URL", allow_http=allow_http)
        origins = tuple(
            _https_url(value, "PAID_ALLOWED_ORIGINS", allow_http=allow_http)
            for value in _csv(env, "PAID_ALLOWED_ORIGINS")
        )
        returns = tuple(
            _https_url(value, "PAID_ALLOWED_RETURN_URLS", allow_http=allow_http)
            for value in _csv(env, "PAID_ALLOWED_RETURN_URLS")
        )
        if not origins or not returns:
            raise ValueError("PAID_ALLOWED_ORIGINS and PAID_ALLOWED_RETURN_URLS must be non-empty")
        live_billing = _bool(env, "PAID_LIVE_BILLING_ENABLED")
        stripe_key = env.get("PAID_STRIPE_SECRET_KEY", "").strip()
        if live_billing:
            if environment != "production" or not stripe_key.startswith("sk_live_"):
                raise ValueError("live billing requires production and an explicit live Stripe key")
            if env.get("PAID_LIVE_BILLING_APPROVAL_REF", "").strip() == "":
                raise ValueError("live billing requires PAID_LIVE_BILLING_APPROVAL_REF")
        elif stripe_key.startswith("sk_live_"):
            raise ValueError("a live Stripe key is forbidden while live billing is disabled")
        hours = int(env.get("PAID_SESSION_HOURS", "12"))
        if hours < 1 or hours > 24:
            raise ValueError("PAID_SESSION_HOURS must be between 1 and 24")
        return cls(
            database_url=_required(env, "PAID_DATABASE_URL"),
            environment=environment,
            public_base_url=public_base,
            allowed_origins=origins,
            allowed_return_urls=returns,
            oidc_issuer=_https_url(_required(env, "PAID_OIDC_ISSUER"), "PAID_OIDC_ISSUER", allow_http=allow_http),
            oidc_client_id=_required(env, "PAID_OIDC_CLIENT_ID"),
            oidc_client_secret=_required(env, "PAID_OIDC_CLIENT_SECRET"),
            oidc_callback_url=_https_url(_required(env, "PAID_OIDC_CALLBACK_URL"), "PAID_OIDC_CALLBACK_URL", allow_http=allow_http),
            oidc_admin_group=_required(env, "PAID_OIDC_ADMIN_GROUP"),
            release_hmac_secret=_required(env, "PAID_RELEASE_HMAC_SECRET"),
            gateway_probe_token=_required(env, "PAID_GATEWAY_PROBE_TOKEN"),
            stripe_secret_key=stripe_key,
            stripe_webhook_secret=env.get("PAID_STRIPE_WEBHOOK_SECRET", "").strip(),
            stripe_account_id=env.get("PAID_STRIPE_ACCOUNT_ID", "platform").strip(),
            stripe_price_id=env.get("PAID_STRIPE_PRICE_ID", "").strip(),
            live_billing_enabled=live_billing,
            sales_enabled=_bool(env, "PAID_SALES_ENABLED"),
            secure_cookies=not allow_http,
            session_hours=hours,
            email_from=env.get("PAID_EMAIL_FROM", "").strip(),
            smtp_url=env.get("PAID_SMTP_URL", "").strip(),
            release_probe_url=env.get("PAID_RELEASE_PROBE_URL", "").strip(),
            source_revision=env.get("PAID_SOURCE_REVISION", "UNSET").strip() or "UNSET",
        )

    @property
    def billing_mode(self) -> str:
        return "live" if self.live_billing_enabled else "test"
