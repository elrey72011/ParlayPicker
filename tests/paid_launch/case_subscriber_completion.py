from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import uuid

from case_postgres_api import auth_cookies, client
from conftest import grant_entitlement, seed_product, session


def _seed_release_and_results(database, *, product_code: str, suffix: str = "owned") -> None:
    release_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    statuses = ["WIN", "LOSS", "PUSH", "VOID", "PENDING", "NEEDS_REVIEW", "CORRECTED"]
    recommendations = [
        {
            "recommendation_id": f"rec-{suffix}-{index}",
            "selection": f"Selection {suffix} {status}",
            "line": 3.5,
            "odds_american": -110,
            "sportsbook_id": "book-test",
            "quote_observed_at": (now - timedelta(hours=2)).isoformat(),
            "analysis_generated_at": (now - timedelta(hours=1)).isoformat(),
            "event_start_utc": (now + timedelta(hours=1)).isoformat(),
            "expiry_at": (now + timedelta(minutes=30)).isoformat(),
            "exact_sport": "NFL",
            "exact_market_family": "SPREAD",
        }
        for index, status in enumerate(statuses, start=1)
    ]
    database.execute(
        """
        INSERT INTO subscriber.release_revision(
          id,release_id,revision_id,product_code,reviewed_payload_hash,customer_payload_hash,
          source_commit,environment,operator_review_id,upstream_authority_id,upstream_authority_hash,
          authority,status,payload,promoted_at,expires_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,'test',%s,%s,%s,%s,'ACTIVE_REVISION_PROMOTED',%s,%s,%s)
        """,
        (
            release_id,
            f"release-{suffix}",
            f"revision-{suffix}",
            product_code,
            "a" * 64,
            ("b" if suffix == "owned" else "c") * 64,
            "d" * 40,
            f"review-{suffix}",
            f"authority-{suffix}",
            "e" * 64,
            json.dumps({"fixture": True}),
            json.dumps({"schema_version": 2, "recommendations": recommendations}),
            now - timedelta(hours=1),
            now + timedelta(hours=1),
        ),
    )
    for index, status in enumerate(statuses, start=1):
        database.execute(
            """
            INSERT INTO subscriber.result_projection(
              id,recommendation_id,release_revision_id,source_settlement_reference,
              source_settlement_hash,settlement_rules_version,status,paper_return,projection_revision)
            VALUES (%s,%s,%s,%s,%s,'settlement-v1',%s,%s,%s)
            """,
            (
                uuid.uuid4(),
                f"rec-{suffix}-{index}",
                release_id,
                f"settlement-{suffix}-{index}",
                f"{index:064x}",
                status,
                0.91 if status == "WIN" else -1 if status == "LOSS" else 0 if status in {"PUSH", "VOID"} else None,
                2 if status == "CORRECTED" else 1,
            ),
        )


def test_offer_and_account_preferences_are_server_owned(settings, database):
    seed_product(database)
    token, csrf, customer_id = session(database, subject="journey-customer")
    api = client(settings, database)
    cookies, headers = auth_cookies(token, csrf)

    offer = api.get("/api/v1/offer", cookies=cookies)
    assert offer.status_code == 200
    payload = offer.json()
    assert payload["checkout_enabled"] is False
    assert "SALES_NOT_OWNER_ENABLED" in payload["reason_codes"]
    assert payload["product"]["amount_minor"] == 1000
    assert payload["product"]["terms_version"] == "terms-v1"
    assert "provider_price_id" not in payload["product"]

    account = api.get("/api/v1/me", cookies=cookies).json()
    assert account["customer"]["alerts_enabled"] is True
    assert api.post("/api/v1/me/alerts", json={"enabled": False}, cookies=cookies).status_code == 403
    saved = api.post("/api/v1/me/alerts", json={"enabled": False}, cookies=cookies, headers=headers)
    assert saved.status_code == 200 and saved.json() == {"alerts_enabled": False}
    refreshed = api.get("/api/v1/me", cookies=cookies).json()
    assert refreshed["customer"]["alerts_enabled"] is False
    assert database.fetch_one("SELECT alerts_enabled FROM subscriber.customer WHERE id=%s", (customer_id,))["alerts_enabled"] is False


def test_results_are_complete_and_scoped_to_entitled_product(settings, database):
    product_id = seed_product(database)
    token, _, customer_id = session(database, subject="results-customer")
    grant_entitlement(database, customer_id, product_id)
    _seed_release_and_results(database, product_code="monthly-qualified-straights")
    _seed_release_and_results(database, product_code="other-product", suffix="foreign")

    response = client(settings, database).get("/api/v1/results", cookies={"pp_session": token})
    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_results"] is True
    assert payload["actual_wagers"] == "UNAVAILABLE"
    assert len(payload["items"]) == 7
    assert {item["status"] for item in payload["items"]} == {
        "WIN", "LOSS", "PUSH", "VOID", "PENDING", "NEEDS_REVIEW", "CORRECTED"
    }
    assert all(item["release_id"] == "release-owned" for item in payload["items"])
    assert all(item["recommendation"]["selection"].startswith("Selection owned ") for item in payload["items"])
    corrected = next(item for item in payload["items"] if item["status"] == "CORRECTED")
    assert corrected["projection_revision"] == 2
    assert corrected["recommendation"]["quote_observed_at"]


def test_cancellation_remains_available_when_sales_are_disabled(settings, database):
    product_id = seed_product(database)
    token, csrf, customer_id = session(database, subject="cancel-customer")
    subscription_id = uuid.uuid4()
    database.execute(
        """
        INSERT INTO subscriber.subscription(
          id,customer_id,product_version_id,provider,provider_account_id,environment,
          provider_customer_id,provider_subscription_id,state,paid_through)
        VALUES (%s,%s,%s,'stripe','acct_test','test','cus_test','sub_test','ACTIVE',now()+interval '30 days')
        """,
        (subscription_id, customer_id, product_id),
    )
    api = client(settings, database)
    cookies, headers = auth_cookies(token, csrf)
    response = api.post("/api/v1/billing/cancel", cookies=cookies, headers=headers)
    assert response.status_code == 202
    assert response.json()["status"] == "CANCELLATION_PENDING_CONFIRMATION"
    saved = database.fetch_one("SELECT cancellation_requested_at FROM subscriber.subscription WHERE id=%s", (subscription_id,))
    assert saved["cancellation_requested_at"] is not None
    assert database.fetch_one("SELECT count(*) AS count FROM subscriber.job_queue WHERE job_type='CANCEL_SUBSCRIPTION'")["count"] == 1
