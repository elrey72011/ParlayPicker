"""Hosted Stripe adapter. Live mode is impossible without explicit configuration."""

from __future__ import annotations

import json
from typing import Any

import stripe

from .settings import Settings


class BillingUnavailable(RuntimeError):
    pass


class StripeBilling:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.stripe_secret_key:
            raise BillingUnavailable("STRIPE_SANDBOX_NOT_CONFIGURED")
        if settings.billing_mode == "test" and not settings.stripe_secret_key.startswith("sk_test_"):
            raise BillingUnavailable("STRIPE_TEST_KEY_REQUIRED")
        if settings.billing_mode == "live" and not settings.live_billing_enabled:
            raise BillingUnavailable("LIVE_BILLING_DISABLED")
        stripe.api_key = settings.stripe_secret_key

    def checkout(
        self,
        *,
        customer_email: str,
        internal_customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        idempotency_key: str,
    ) -> dict[str, str]:
        if price_id != self.settings.stripe_price_id or not price_id:
            raise ValueError("PRICE_NOT_ALLOWLISTED")
        if success_url not in self.settings.allowed_return_urls or cancel_url not in self.settings.allowed_return_urls:
            raise ValueError("RETURN_URL_NOT_ALLOWLISTED")
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=customer_email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=internal_customer_id,
            metadata={"internal_customer_id": internal_customer_id, "environment": self.settings.environment},
            idempotency_key=idempotency_key,
        )
        return {"checkout_session_id": session.id, "url": session.url}

    def portal(self, *, provider_customer_id: str, return_url: str, idempotency_key: str) -> dict[str, str]:
        if return_url not in self.settings.allowed_return_urls:
            raise ValueError("RETURN_URL_NOT_ALLOWLISTED")
        session = stripe.billing_portal.Session.create(
            customer=provider_customer_id,
            return_url=return_url,
            idempotency_key=idempotency_key,
        )
        return {"url": session.url}

    def cancel_at_period_end(self, provider_subscription_id: str, idempotency_key: str) -> dict[str, Any]:
        subscription = stripe.Subscription.modify(
            provider_subscription_id,
            cancel_at_period_end=True,
            idempotency_key=idempotency_key,
        )
        return {"id": subscription.id, "cancel_at_period_end": bool(subscription.cancel_at_period_end)}

    def construct_event(self, raw_body: bytes, signature: str) -> dict[str, Any]:
        if not self.settings.stripe_webhook_secret:
            raise BillingUnavailable("STRIPE_WEBHOOK_SECRET_NOT_CONFIGURED")
        event = stripe.Webhook.construct_event(raw_body, signature, self.settings.stripe_webhook_secret)
        if hasattr(event, "to_dict_recursive"):
            return event.to_dict_recursive()
        return json.loads(str(event))

    def retrieve_subscription(self, subscription_id: str) -> dict[str, Any]:
        return dict(stripe.Subscription.retrieve(subscription_id))
