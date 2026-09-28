from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import uuid

from fastapi.testclient import TestClient

from services.subscriber.app import create_app
from services.subscriber.auth import revoke_session
from services.subscriber.canonical import object_hash, sign
from services.subscriber.contracts import ReleaseSubmission

from conftest import grant_entitlement, seed_product, session
from case_policy_and_contracts import submission


class FakeBilling:
    webhook_event = {"id": "evt_1", "type": "customer.subscription.updated", "livemode": False, "data": {"object": {}}}

    def __init__(self, settings):
        self.settings = settings

    def construct_event(self, raw, signature):
        if signature != "valid":
            raise ValueError("bad signature")
        return self.webhook_event

    def checkout(self, **kwargs):
        return {"checkout_session_id": "cs_test", "url": "https://checkout.stripe.test/session"}

    def portal(self, **kwargs):
        return {"url": "https://billing.stripe.test/portal"}


def client(settings, database):
    return TestClient(create_app(settings, db=database, billing_factory=FakeBilling))


def auth_cookies(token, csrf):
    return {"pp_session": token, "pp_csrf": csrf}, {"X-CSRF-Token": csrf, "Origin": "http://subscriber.test"}


def test_anonymous_requests_never_return_premium_payload(settings, database):
    api = client(settings, database)
    status = api.get("/api/v1/status")
    assert status.status_code == 200
    assert "recommendations" not in status.json()
    assert api.get("/api/v1/picks/current").status_code == 401
    assert api.get("/api/v1/results").status_code == 401


def test_session_logout_and_csrf_are_enforced(settings, database):
    token, csrf, _ = session(database)
    api = client(settings, database)
    cookies, headers = auth_cookies(token, csrf)
    assert api.get("/api/v1/me", cookies=cookies).status_code == 200
    assert api.post("/auth/logout", cookies=cookies).status_code == 403
    assert api.post("/auth/logout", cookies=cookies, headers=headers).status_code == 204
    assert api.get("/api/v1/me", cookies=cookies).status_code == 401


def test_cross_customer_entitlement_and_direct_id_changes_do_not_grant_access(settings, database):
    product = seed_product(database)
    token_a, csrf_a, customer_a = session(database, subject="customer-a")
    token_b, _, customer_b = session(database, subject="customer-b")
    grant_entitlement(database, customer_a, product)
    api = client(settings, database)
    assert api.get("/api/v1/picks/current", cookies={"pp_session": token_a}).status_code == 200
    assert api.get("/api/v1/picks/current", cookies={"pp_session": token_b}).status_code == 403
    assert api.get(f"/api/v1/releases/{customer_a}", cookies={"pp_session": token_b}).status_code == 404


def test_client_cannot_supply_price_customer_or_role(settings, database):
    seed_product(database)
    token, csrf, _ = session(database)
    api = client(settings, database)
    cookies, headers = auth_cookies(token, csrf)
    body = {
        "success_url": "http://subscriber.test/account", "cancel_url": "http://subscriber.test/",
        "terms_version": "terms-v1", "renewal_disclosure_version": "renewal-v1",
        "cancellation_policy_version": "cancel-v1", "refund_policy_version": "refund-v1",
        "price_id": "attacker-price", "customer_id": str(uuid.uuid4()), "role": "OWNER",
    }
    response = api.post("/api/v1/billing/checkout", json=body, cookies=cookies, headers=headers)
    assert response.status_code == 422


def test_webhook_signature_and_deduplication(settings, database):
    api = client(settings, database)
    assert api.post("/api/v1/webhooks/billing", content=b"{}", headers={"Stripe-Signature": "invalid"}).status_code == 400
    first = api.post("/api/v1/webhooks/billing", content=b"{}", headers={"Stripe-Signature": "valid"})
    second = api.post("/api/v1/webhooks/billing", content=b"{}", headers={"Stripe-Signature": "valid"})
    assert first.status_code == 202 and not first.json()["duplicate"]
    assert second.status_code == 202 and second.json()["duplicate"]
    assert database.fetch_one("SELECT count(*) AS count FROM subscriber.billing_event")["count"] == 1
    assert database.fetch_one("SELECT count(*) AS count FROM subscriber.job_queue")["count"] == 1


def test_reviewed_release_is_staged_without_lock_or_publication_write(settings, database):
    product = seed_product(database)
    token, csrf, owner_id = session(database, role="OWNER", auth_strength="mfa", subject="owner")
    for key, value in {
        "commercially_enabled_markets": ["NFL:SPREAD"], "qualified_markets": ["NFL:SPREAD"],
        "release_status": "OWNER_ENABLED",
    }.items():
        database.execute(
            "INSERT INTO subscriber.service_state(key,value) VALUES (%s,%s) ON CONFLICT (key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
    database.execute(
        """
        INSERT INTO subscriber.approval_record(id,approval_type,scope,status,reviewer,evidence_reference,evidence_hash,reviewed_at)
        VALUES (%s,'DATA_RIGHTS','paid-launch','APPROVED','owner','rights-evidence',%s,now())
        """,
        (uuid.uuid4(), "d" * 64),
    )
    raw = submission()
    api = client(settings, database)
    cookies, headers = auth_cookies(token, csrf)
    response = api.post(
        "/api/v1/admin/releases", json={"submission": raw, "signature": sign(raw, settings.release_hmac_secret)},
        cookies=cookies, headers=headers,
    )
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "PENDING_REVISION_COMMITTED"
    assert database.fetch_one("SELECT count(*) AS count FROM subscriber.release_revision")["count"] == 1
    assert database.fetch_one("SELECT count(*) AS count FROM subscriber.job_queue WHERE job_type='VERIFY_RELEASE_ROUTE'")["count"] == 1


def test_unknown_authority_and_generic_publish_text_cannot_approve(settings, database):
    seed_product(database)
    token, csrf, _ = session(database, role="OWNER", auth_strength="mfa", subject="owner")
    database.execute(
        "INSERT INTO subscriber.service_state(key,value) VALUES ('commercially_enabled_markets',%s)",
        (json.dumps(["NFL:SPREAD"]),),
    )
    raw = submission()
    raw["authority"]["market_status"] = "RESEARCH"
    raw["authority"]["upstream_gate_result"] = "Published: website complete"
    parsed = ReleaseSubmission.model_validate(raw)
    raw["reviewed_payload_hash"] = object_hash(parsed.review_payload())
    api = client(settings, database)
    cookies, headers = auth_cookies(token, csrf)
    response = api.post(
        "/api/v1/admin/releases", json={"submission": raw, "signature": sign(raw, settings.release_hmac_secret)},
        cookies=cookies, headers=headers,
    )
    assert response.status_code == 503
    assert database.fetch_one("SELECT count(*) AS count FROM subscriber.release_revision")["count"] == 0
