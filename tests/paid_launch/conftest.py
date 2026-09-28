from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import uuid

import pytest


@pytest.fixture(scope="session")
def database_url() -> str:
    value = os.environ.get("PAID_TEST_DATABASE_URL", "")
    if not value:
        pytest.skip("PAID_TEST_DATABASE_URL is required for PostgreSQL integration tests")
    return value


@pytest.fixture()
def database(database_url):
    from services.subscriber.db import Database

    db = Database(database_url)
    with db.transaction() as connection:
        connection.execute("DROP SCHEMA IF EXISTS subscriber CASCADE")
    db.migrate()
    return db


@pytest.fixture()
def settings(database_url):
    from services.subscriber.settings import Settings

    return Settings.from_env(
        {
            "PAID_DATABASE_URL": database_url,
            "PAID_ENVIRONMENT": "test",
            "PAID_PUBLIC_BASE_URL": "http://subscriber.test",
            "PAID_ALLOWED_ORIGINS": "http://subscriber.test",
            "PAID_ALLOWED_RETURN_URLS": "http://subscriber.test/account,http://subscriber.test/",
            "PAID_OIDC_ISSUER": "http://issuer.test",
            "PAID_OIDC_CLIENT_ID": "client",
            "PAID_OIDC_CLIENT_SECRET": "secret",
            "PAID_OIDC_CALLBACK_URL": "http://subscriber.test/auth/callback",
            "PAID_OIDC_ADMIN_GROUP": "owners",
            "PAID_RELEASE_HMAC_SECRET": "release-secret-at-least-32-bytes-long",
            "PAID_GATEWAY_PROBE_TOKEN": "gateway-secret-at-least-32-bytes-long",
            "PAID_RELEASE_PROBE_URL": "http://subscriber.test/internal/v1/releases",
            "PAID_STRIPE_SECRET_KEY": "sk_test_fixture",
            "PAID_STRIPE_WEBHOOK_SECRET": "whsec_fixture",
            "PAID_STRIPE_ACCOUNT_ID": "acct_test",
            "PAID_STRIPE_PRICE_ID": "price_monthly",
            "PAID_LIVE_BILLING_ENABLED": "false",
            "PAID_SALES_ENABLED": "false",
            "PAID_EMAIL_FROM": "service@example.test",
            "PAID_SMTP_URL": "smtp://mailpit:1025",
        }
    )


def seed_product(db, *, active=True):
    product_id = uuid.uuid4()
    db.execute(
        """
        INSERT INTO subscriber.product_version(id,product_code,version,offered_markets,provider,provider_price_id,
          currency,amount_minor,selling_entity,eligible_jurisdictions,terms_version,renewal_disclosure_version,
          cancellation_policy_version,refund_policy_version,environment,active)
        VALUES (%s,'monthly-qualified-straights',1,ARRAY['NFL:SPREAD'],'stripe','price_monthly','USD',1000,
          'Test Entity',ARRAY['TEST'],'terms-v1','renewal-v1','cancel-v1','refund-v1','test',%s)
        """,
        (product_id, active),
    )
    return product_id


def grant_entitlement(db, customer_id, product_id):
    db.execute(
        """
        INSERT INTO subscriber.entitlement(id,customer_id,product_version_id,effective_at,expires_at,revision,reason)
        VALUES (%s,%s,%s,%s,%s,1,'test fixture')
        """,
        (uuid.uuid4(), customer_id, product_id, datetime.now(timezone.utc) - timedelta(minutes=1), datetime.now(timezone.utc) + timedelta(days=1)),
    )


def session(db, *, role="CUSTOMER", auth_strength="oidc", subject=None):
    from services.subscriber.auth import create_customer_session, token_hash

    subject = subject or str(uuid.uuid4())
    token, csrf = create_customer_session(
        db, issuer="http://issuer.test", subject=subject, email=f"{subject}@example.test",
        email_verified=True, role=role, auth_strength=auth_strength, hours=12,
    )
    row = db.fetch_one(
        "SELECT customer_id FROM subscriber.session WHERE token_hash=%s",
        (token_hash(token),),
    )
    return token, csrf, row["customer_id"]
