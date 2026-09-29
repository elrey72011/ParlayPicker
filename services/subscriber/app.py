"""FastAPI subscriber boundary.

Run with ``uvicorn services.subscriber.app:create_from_env --factory``.
"""

from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import hmac
import json
import time
from typing import Annotated, Any, Callable, Literal
from urllib.parse import urlsplit
import uuid

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from integrations.subscriber_release.authority import AuthorityError, verify_reviewed_submission
from .auth import AuthenticatedCustomer, OidcClient, authenticate, revoke_session, validate_csrf
from .billing import BillingUnavailable, StripeBilling
from .canonical import object_hash, sha256_hex
from .contracts import AuthorityBinding
from .db import Database, new_id
from .launch_gate import checkout_decision, release_decision
from .repository import ConflictError, SubscriberRepository
from .settings import Settings


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckoutBody(Body):
    success_url: str
    cancel_url: str
    terms_version: str
    renewal_disclosure_version: str
    cancellation_policy_version: str
    refund_policy_version: str


class PortalBody(Body):
    return_url: str


class AlertPreferenceBody(Body):
    enabled: bool


class ReleaseBody(Body):
    submission: dict[str, Any]
    signature: str = Field(min_length=64, max_length=64)


class WithdrawalBody(Body):
    reason: str = Field(min_length=10, max_length=500)


class ControlBody(Body):
    control: Literal["SALES", "RELEASES", "ALERTS"]
    action: Literal["ENABLE", "PAUSE"]
    reason: str = Field(min_length=10, max_length=500)


class MarketSuspendBody(Body):
    exact_market: str = Field(pattern=r"^(NFL|NCAAF|NBA|NCAAB):(SPREAD|TOTAL)$|^(MLB):(RUN_LINE|TOTAL)$|^(NHL):(PUCK_LINE|TOTAL)$")
    reason: str = Field(min_length=10, max_length=500)


class FixedWindowLimiter:
    """Per-process backstop; deploy routing also applies an external limit."""

    def __init__(self) -> None:
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, limit: int, window: int) -> None:
        current = time.monotonic()
        values = self.hits[key]
        while values and values[0] <= current - window:
            values.popleft()
        if len(values) >= limit:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(window)})
        values.append(current)


def _safe_request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if supplied and len(supplied) <= 100 and all(char.isalnum() or char in "-_." for char in supplied):
        return supplied
    return str(uuid.uuid4())


def _serialize(row: Any) -> Any:
    if isinstance(row, dict):
        return {key: _serialize(value) for key, value in row.items()}
    if isinstance(row, list):
        return [_serialize(value) for value in row]
    if isinstance(row, (datetime, uuid.UUID)):
        return str(row)
    return row


def create_app(
    settings: Settings,
    *,
    db: Database | None = None,
    billing_factory: Callable[[Settings], StripeBilling] = StripeBilling,
    oidc: OidcClient | None = None,
) -> FastAPI:
    database = db or Database(settings.database_url)
    repository = SubscriberRepository(database, settings.environment)
    oidc_client = oidc or OidcClient(settings)
    limiter = FixedWindowLimiter()
    app = FastAPI(title="ParlayPicker Subscriber Service", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.db = database
    app.state.repository = repository
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-ID", "Idempotency-Key"],
        max_age=300,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable) -> Response:
        request.state.request_id = _safe_request_id(request)
        try:
            response = await call_next(request)
        except Exception:
            # Let FastAPI handlers format known errors; unknowns remain opaque.
            raise
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.url.path.startswith(("/api/v1/me", "/api/v1/offer", "/api/v1/picks", "/api/v1/releases", "/api/v1/results", "/api/v1/admin", "/internal/")):
            response.headers["Cache-Control"] = "private, no-store"
            response.headers["Pragma"] = "no-cache"
            response.headers["Vary"] = "Cookie, Authorization"
        return response

    def current_customer(
        request: Request,
        pp_session: Annotated[str | None, Cookie(alias="pp_session")] = None,
    ) -> AuthenticatedCustomer:
        limiter.check(f"protected:{request.client.host if request.client else 'unknown'}", limit=240, window=60)
        customer = authenticate(database, pp_session)
        if not customer:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required")
        return customer

    def csrf_customer(
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(current_customer)],
        csrf_header: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
        csrf_cookie: Annotated[str | None, Cookie(alias="pp_csrf")] = None,
    ) -> AuthenticatedCustomer:
        if not csrf_header or not csrf_cookie or csrf_header != csrf_cookie or not validate_csrf(database, customer.session_id, csrf_header):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF validation failed")
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") not in settings.allowed_origins:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "origin not allowed")
        return customer

    def owner(
        customer: Annotated[AuthenticatedCustomer, Depends(csrf_customer)],
    ) -> AuthenticatedCustomer:
        if customer.role not in {"OWNER", "OPERATOR"} or customer.auth_strength != "mfa":
            raise HTTPException(status.HTTP_403_FORBIDDEN, "operator MFA required")
        return customer

    @app.exception_handler(AuthorityError)
    async def authority_error(_: Request, exc: AuthorityError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_error(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/api/v1/status")
    def public_status(request: Request) -> dict[str, Any]:
        limiter.check(f"status:{request.client.host if request.client else 'unknown'}", limit=120, window=60)
        state = repository.service_state()
        available = state["engineering_status"] == "STAGING_VERIFIED"
        return {
            "schema_version": 1,
            "service": "available" if available else "limited",
            "sales": "enabled" if state["sales_status"] == "OWNER_ENABLED" else "disabled",
            "source_revision": settings.source_revision,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/auth/login")
    async def login(request: Request, redirect_after: str) -> RedirectResponse:
        limiter.check(f"login:{request.client.host if request.client else 'unknown'}", limit=10, window=60)
        try:
            location = await oidc_client.begin(database, redirect_after)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse(location, status_code=303)

    @app.get("/auth/callback")
    async def oidc_callback(request: Request, state: str, code: str) -> RedirectResponse:
        limiter.check(f"callback:{request.client.host if request.client else 'unknown'}", limit=20, window=60)
        try:
            session_token, csrf_token, redirect_after = await oidc_client.complete(database, state=state, code=code)
        except (ValueError, KeyError) as exc:
            raise HTTPException(401, "OIDC callback rejected") from exc
        response = RedirectResponse(redirect_after, status_code=303)
        response.set_cookie(
            "pp_session", session_token, secure=settings.secure_cookies, httponly=True,
            samesite="lax", max_age=settings.session_hours * 3600, path="/",
        )
        response.set_cookie(
            "pp_csrf", csrf_token, secure=settings.secure_cookies, httponly=False,
            samesite="strict", max_age=settings.session_hours * 3600, path="/",
        )
        return response

    @app.post("/auth/logout", status_code=204)
    def logout(customer: Annotated[AuthenticatedCustomer, Depends(csrf_customer)]) -> Response:
        revoke_session(database, customer.session_id)
        response = Response(status_code=204)
        response.delete_cookie("pp_session", path="/")
        response.delete_cookie("pp_csrf", path="/")
        return response

    @app.get("/api/v1/me")
    def me(customer: Annotated[AuthenticatedCustomer, Depends(current_customer)]) -> dict[str, Any]:
        return _serialize(repository.customer_account(customer.customer_id))

    @app.get("/api/v1/offer")
    def offer(customer: Annotated[AuthenticatedCustomer, Depends(current_customer)]) -> dict[str, Any]:
        """Return the server-owned offer contract without provider authority fields."""
        state, approvals = repository.service_state(), repository.current_approvals()
        decision = checkout_decision(
            engineering_status=state["engineering_status"], commercial_status=state["commercial_status"],
            sales_status=state["sales_status"], configured_terms=bool(state["configured_terms"]),
            qualified_markets=state["qualified_markets"], commercially_enabled_markets=state["commercially_enabled_markets"],
            current_approvals=approvals, billing_mode=settings.billing_mode,
            live_billing_enabled=settings.live_billing_enabled,
        )
        product = repository.active_product()
        reasons = list(decision.reason_codes)
        if not product or product["provider_price_id"] != settings.stripe_price_id:
            reasons.append("CONFIGURED_PRODUCT_UNAVAILABLE")
        public_product = None if not product else {
            "product_code": product["product_code"],
            "version": product["version"],
            "currency": product["currency"],
            "amount_minor": product["amount_minor"],
            "selling_entity": product["selling_entity"],
            "offered_markets": product["offered_markets"],
            "eligible_jurisdictions": product["eligible_jurisdictions"],
            "terms_version": product["terms_version"],
            "renewal_disclosure_version": product["renewal_disclosure_version"],
            "cancellation_policy_version": product["cancellation_policy_version"],
            "refund_policy_version": product["refund_policy_version"],
        }
        return {
            "schema_version": 1,
            "checkout_enabled": not reasons,
            "reason_codes": reasons,
            "product": _serialize(public_product),
            "evaluated_at": decision.evaluated_at.isoformat(),
        }

    @app.post("/api/v1/me/alerts")
    def alert_preferences(
        body: AlertPreferenceBody,
        customer: Annotated[AuthenticatedCustomer, Depends(csrf_customer)],
    ) -> dict[str, bool]:
        database.execute(
            "UPDATE subscriber.customer SET alerts_enabled=%s,updated_at=now() WHERE id=%s",
            (body.enabled, customer.customer_id),
        )
        return {"alerts_enabled": body.enabled}

    @app.get("/api/v1/picks/current")
    def current_picks(customer: Annotated[AuthenticatedCustomer, Depends(current_customer)]) -> dict[str, Any]:
        if not repository.entitlement(customer.customer_id):
            raise HTTPException(403, "active entitlement required")
        release = repository.current_release(customer.customer_id)
        if not release:
            return {
                "schema_version": 1,
                "status": "QUALIFIED_FEED_UNAVAILABLE",
                "as_of": datetime.now(timezone.utc).isoformat(),
                "release_id": None,
                "recommendations": [],
                "reason_codes": ["NO_CURRENT_VERIFIED_RELEASE"],
                "retryable": False,
            }
        payload = dict(release["payload"])
        payload.update(
            {
                "status": "CURRENT",
                "published_at": _serialize(release["promoted_at"]),
                "content_hash": release["customer_payload_hash"],
                "as_of": datetime.now(timezone.utc).isoformat(),
            }
        )
        return payload

    @app.get("/api/v1/releases/{release_id}")
    def release(release_id: str, customer: Annotated[AuthenticatedCustomer, Depends(current_customer)]) -> dict[str, Any]:
        row = repository.release_by_id(customer.customer_id, release_id)
        if not row:
            raise HTTPException(404, "release not found")
        return _serialize(row)

    @app.get("/api/v1/results")
    def results(customer: Annotated[AuthenticatedCustomer, Depends(current_customer)]) -> dict[str, Any]:
        if not repository.entitlement(customer.customer_id):
            raise HTTPException(403, "active entitlement required")
        rows = repository.results(customer.customer_id)
        return {"schema_version": 1, "paper_results": True, "actual_wagers": "UNAVAILABLE", "items": _serialize(rows)}

    @app.post("/api/v1/billing/checkout")
    def checkout(
        body: CheckoutBody,
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(csrf_customer)],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> dict[str, str]:
        limiter.check(f"checkout:{customer.customer_id}", limit=5, window=300)
        if not customer.email_verified or not customer.contact_email:
            raise HTTPException(409, "verified contact email required")
        state, approvals = repository.service_state(), repository.current_approvals()
        decision = checkout_decision(
            engineering_status=state["engineering_status"], commercial_status=state["commercial_status"],
            sales_status=state["sales_status"], configured_terms=bool(state["configured_terms"]),
            qualified_markets=state["qualified_markets"], commercially_enabled_markets=state["commercially_enabled_markets"],
            current_approvals=approvals, billing_mode=settings.billing_mode,
            live_billing_enabled=settings.live_billing_enabled,
        )
        if not decision.allowed:
            raise HTTPException(503, {"status": "CHECKOUT_DISABLED", "reason_codes": decision.reason_codes})
        product = repository.active_product()
        if not product or product["provider_price_id"] != settings.stripe_price_id:
            raise HTTPException(503, "configured product is unavailable")
        expected_versions = {
            "terms_version": product["terms_version"],
            "renewal_disclosure_version": product["renewal_disclosure_version"],
            "cancellation_policy_version": product["cancellation_policy_version"],
            "refund_policy_version": product["refund_policy_version"],
        }
        supplied = body.model_dump()
        for key, expected in expected_versions.items():
            if supplied[key] != expected:
                raise HTTPException(409, f"{key} is stale")
        key = idempotency_key or request.state.request_id
        scoped_key = hashlib.sha256(f"checkout:{customer.customer_id}:{key}".encode()).hexdigest()
        with database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO subscriber.consent_record(id,customer_id,product_version_id,terms_version,renewal_disclosure_version,
                  cancellation_policy_version,refund_policy_version,consented_at,request_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,now(),%s)
                """,
                (new_id(), customer.customer_id, product["id"], body.terms_version, body.renewal_disclosure_version,
                 body.cancellation_policy_version, body.refund_policy_version, request.state.request_id),
            )
        try:
            billing = billing_factory(settings)
            return billing.checkout(
                customer_email=customer.contact_email, internal_customer_id=str(customer.customer_id),
                price_id=product["provider_price_id"], success_url=body.success_url, cancel_url=body.cancel_url,
                idempotency_key=scoped_key,
            )
        except BillingUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/v1/billing/portal")
    def portal(
        body: PortalBody,
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(csrf_customer)],
    ) -> dict[str, str]:
        subscription = database.fetch_one(
            "SELECT provider_customer_id FROM subscriber.subscription WHERE customer_id=%s ORDER BY updated_at DESC LIMIT 1",
            (customer.customer_id,),
        )
        if not subscription:
            raise HTTPException(404, "billing account not found")
        try:
            return billing_factory(settings).portal(
                provider_customer_id=subscription["provider_customer_id"],
                return_url=body.return_url,
                idempotency_key=hashlib.sha256(f"portal:{customer.customer_id}:{request.state.request_id}".encode()).hexdigest(),
            )
        except BillingUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/v1/billing/cancel", status_code=202)
    def cancel(
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(csrf_customer)],
    ) -> dict[str, str]:
        # Cancellation remains available even when sales are paused. The worker
        # reconciles uncertain provider outcomes before presenting confirmation.
        with database.transaction() as connection:
            subscription = connection.execute(
                "SELECT id,provider_subscription_id FROM subscriber.subscription WHERE customer_id=%s ORDER BY updated_at DESC LIMIT 1 FOR UPDATE",
                (customer.customer_id,),
            ).fetchone()
            if not subscription:
                raise HTTPException(404, "subscription not found")
            connection.execute(
                "UPDATE subscriber.subscription SET cancellation_requested_at=COALESCE(cancellation_requested_at,now()) WHERE id=%s",
                (subscription["id"],),
            )
            connection.execute(
                """
                INSERT INTO subscriber.job_queue(id,job_type,deduplication_key,payload,status)
                VALUES (%s,'CANCEL_SUBSCRIPTION',%s,%s,'PENDING') ON CONFLICT (deduplication_key) DO NOTHING
                """,
                (new_id(), f"cancel:{subscription['id']}", {"subscription_id": str(subscription["id"]), "request_id": request.state.request_id}),
            )
        return {"status": "CANCELLATION_PENDING_CONFIRMATION"}

    @app.post("/api/v1/webhooks/billing", status_code=202)
    async def billing_webhook(request: Request, stripe_signature: Annotated[str | None, Header(alias="Stripe-Signature")] = None) -> dict[str, Any]:
        limiter.check(f"webhook:{request.client.host if request.client else 'unknown'}", limit=600, window=60)
        raw = await request.body()
        try:
            event = billing_factory(settings).construct_event(raw, stripe_signature or "")
        except Exception as exc:
            raise HTTPException(400, "invalid billing webhook") from exc
        event_mode = "live" if event.get("livemode") is True else "test"
        inserted = repository.record_billing_event(
            event, sha256_hex(raw), settings.stripe_account_id, event_mode, settings.billing_mode,
        )
        return {"accepted": True, "duplicate": not inserted}

    @app.post("/api/v1/admin/releases", status_code=202)
    def submit_release(
        body: ReleaseBody,
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(owner)],
    ) -> dict[str, Any]:
        submission = verify_reviewed_submission(
            body.submission, signature=body.signature, secret=settings.release_hmac_secret,
            environment=settings.environment,
        )
        state = repository.service_state()
        if state["release_status"] != "OWNER_ENABLED":
            raise HTTPException(503, {"status": "RELEASE_BLOCKED", "reason_codes": ["NEW_RELEASES_PAUSED"]})
        exact_markets = [f"{item.exact_sport}:{item.exact_market_family}" for item in submission.recommendations]
        decision = release_decision(
            authority=submission.authority.model_dump(), exact_markets=exact_markets,
            commercially_enabled_markets=state["commercially_enabled_markets"],
            expiry_at=min(item.expiry_at for item in submission.recommendations),
            event_starts=[item.event_start_utc for item in submission.recommendations],
            reviewed_hash_matches=object_hash(submission.review_payload()) == submission.reviewed_payload_hash,
            rights_present=bool(submission.authority.content_rights_references) and "DATA_RIGHTS" in repository.current_approvals(),
        )
        if not decision.allowed:
            raise HTTPException(503, {"status": "RELEASE_BLOCKED", "reason_codes": decision.reason_codes})
        return repository.stage_release(submission, request.state.request_id, customer.customer_id)

    @app.post("/api/v1/admin/releases/{release_id}/withdraw")
    def withdraw_release(
        release_id: str,
        body: WithdrawalBody,
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(owner)],
    ) -> dict[str, str]:
        if not repository.withdraw_release(release_id, body.reason, request.state.request_id, customer.customer_id):
            raise HTTPException(404, "release not found")
        return {"status": "WITHDRAWN"}

    @app.post("/api/v1/admin/controls")
    def update_control(
        body: ControlBody,
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(owner)],
    ) -> dict[str, str]:
        state, approvals = repository.service_state(), repository.current_approvals()
        key = {"SALES": "sales_status", "RELEASES": "release_status", "ALERTS": "alerts_status"}[body.control]
        value = "PAUSED"
        if body.action == "ENABLE":
            if body.control == "SALES":
                decision = checkout_decision(
                    engineering_status=state["engineering_status"], commercial_status=state["commercial_status"],
                    sales_status="OWNER_ENABLED", configured_terms=bool(state["configured_terms"]),
                    qualified_markets=state["qualified_markets"], commercially_enabled_markets=state["commercially_enabled_markets"],
                    current_approvals=approvals, billing_mode=settings.billing_mode,
                    live_billing_enabled=settings.live_billing_enabled,
                )
                if not decision.allowed:
                    raise HTTPException(503, {"status": "CONTROL_BLOCKED", "reason_codes": decision.reason_codes})
            elif body.control == "RELEASES":
                reasons = []
                if state["engineering_status"] != "STAGING_VERIFIED":
                    reasons.append("ENGINEERING_NOT_STAGING_VERIFIED")
                if not set(state["qualified_markets"]) & set(state["commercially_enabled_markets"]):
                    reasons.append("NO_COMMERCIALLY_ENABLED_QUALIFIED_MARKET")
                if "DATA_RIGHTS" not in approvals:
                    reasons.append("MISSING_APPROVAL_DATA_RIGHTS")
                if reasons:
                    raise HTTPException(503, {"status": "CONTROL_BLOCKED", "reason_codes": reasons})
            elif not settings.smtp_url or state["engineering_status"] != "STAGING_VERIFIED":
                raise HTTPException(503, {"status": "CONTROL_BLOCKED", "reason_codes": ["ALERT_DELIVERY_NOT_STAGING_VERIFIED"]})
            value = "OWNER_ENABLED"
        with database.transaction() as connection:
            before = state[key]
            connection.execute(
                "INSERT INTO subscriber.service_state(key,value) VALUES (%s,%s) ON CONFLICT (key) DO UPDATE SET value=excluded.value,updated_at=now()",
                (key, json.dumps(value)),
            )
            connection.execute(
                "INSERT INTO subscriber.admin_audit(id,actor_customer_id,action,scoped_target,before_reference,after_reference,reason,request_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (new_id(), customer.customer_id, f"{body.control}_{body.action}", key, str(before), value, body.reason, request.state.request_id),
            )
        return {"control": body.control, "status": value}

    @app.post("/api/v1/admin/markets/suspend")
    def suspend_market(
        body: MarketSuspendBody,
        request: Request,
        customer: Annotated[AuthenticatedCustomer, Depends(owner)],
    ) -> dict[str, Any]:
        state = repository.service_state()
        before = list(state["commercially_enabled_markets"])
        after = [market for market in before if market != body.exact_market]
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO subscriber.service_state(key,value) VALUES ('commercially_enabled_markets',%s) ON CONFLICT (key) DO UPDATE SET value=excluded.value,updated_at=now()",
                (json.dumps(after),),
            )
            connection.execute(
                "INSERT INTO subscriber.admin_audit(id,actor_customer_id,action,scoped_target,before_reference,after_reference,reason,request_id) VALUES (%s,%s,'MARKET_SUSPENDED',%s,%s,%s,%s,%s)",
                (new_id(), customer.customer_id, body.exact_market, json.dumps(before), json.dumps(after), body.reason, request.state.request_id),
            )
        return {"status": "SUSPENDED", "exact_market": body.exact_market, "commercially_enabled_markets": after}

    @app.get("/api/v1/admin/launch-readiness")
    def launch_readiness(customer: Annotated[AuthenticatedCustomer, Depends(current_customer)]) -> dict[str, Any]:
        if customer.role not in {"OWNER", "OPERATOR"} or customer.auth_strength != "mfa":
            raise HTTPException(403, "operator MFA required")
        state, approvals = repository.service_state(), repository.current_approvals()
        decision = checkout_decision(
            engineering_status=state["engineering_status"], commercial_status=state["commercial_status"],
            sales_status=state["sales_status"], configured_terms=bool(state["configured_terms"]),
            qualified_markets=state["qualified_markets"], commercially_enabled_markets=state["commercially_enabled_markets"],
            current_approvals=approvals, billing_mode=settings.billing_mode,
            live_billing_enabled=settings.live_billing_enabled,
        )
        operational = {
            "billing_event_backlog": database.fetch_one("SELECT count(*) AS count FROM subscriber.billing_event WHERE status IN ('RECEIVED','PROCESSING','RETRY')")["count"],
            "billing_reconciliation_last_scheduled": (database.fetch_one("SELECT value FROM subscriber.service_state WHERE key='billing_reconciliation_last_scheduled'") or {}).get("value"),
            "notification_backlog": database.fetch_one("SELECT count(*) AS count FROM subscriber.notification_outbox WHERE status IN ('PENDING','SENDING','FAILED','UNKNOWN_OUTCOME')")["count"],
            "dead_letter_jobs": database.fetch_one("SELECT count(*) AS count FROM subscriber.job_queue WHERE status='DEAD_LETTER'")["count"],
        }
        return {"schema_version": 1, "checkout": decision.model_dump(mode="json"), "state": state, "approvals": sorted(approvals), "operational": _serialize(operational)}

    @app.get("/internal/v1/releases/{revision_id}")
    def routed_release_probe(revision_id: str, authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
        expected = "Bearer " + settings.gateway_probe_token
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(403, "probe authorization rejected")
        row = database.fetch_one(
            "SELECT payload,customer_payload_hash,status FROM subscriber.release_revision WHERE revision_id=%s AND environment=%s",
            (revision_id, settings.environment),
        )
        if not row:
            raise HTTPException(404, "revision not found")
        return {"payload": row["payload"], "content_hash": row["customer_payload_hash"], "status": row["status"]}

    return app


def create_from_env() -> FastAPI:
    return create_app(Settings.from_env())
