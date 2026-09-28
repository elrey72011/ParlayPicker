from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy

import pytest

from integrations.subscriber_release.authority import AuthorityError, verify_reviewed_submission
from services.subscriber.canonical import object_hash, sign
from services.subscriber.contracts import Recommendation, ReleaseSubmission
from services.subscriber.launch_gate import checkout_decision
from services.subscriber.settings import Settings


def recommendation(now):
    p_win, p_push, p_loss, odds = 0.55, 0.02, 0.43, 1.91
    p_win_conservative = 0.52
    return {
        "schema_version": 2, "recommendation_id": "rec-1", "exact_sport": "NFL",
        "exact_market_family": "SPREAD", "canonical_event_id": "event-1", "selection": "Example +3",
        "line": 3.0, "sportsbook_id": "book-1", "odds_american": -110, "odds_decimal": odds,
        "quote_id": "quote-1", "quote_observed_at": now.isoformat(), "provider_updated_at": None,
        "analysis_generated_at": now.isoformat(), "event_start_utc": (now + timedelta(hours=3)).isoformat(),
        "expiry_at": (now + timedelta(minutes=10)).isoformat(), "model_id": "model-1",
        "model_artifact_hash": "a" * 64, "model_target_semantics": "unconditional win/push/loss",
        "calibration_id": "cal-1", "validation_artifact_id": "validation-1", "policy_id": "policy-1",
        "activation_reference": "activation-1", "probability_semantics": "unconditional",
        "p_win": p_win, "p_push": p_push, "p_loss": p_loss,
        "mean_ev_per_unit": p_win * odds + p_push - 1,
        "p_win_conservative": p_win_conservative,
        "conservative_ev_per_unit": p_win_conservative * odds + p_push - 1,
        "uncertainty_method": "fixed_push_lower_win_bound",
        "minimum_acceptable_decimal_odds": 1.9, "minimum_acceptable_line": 3.0,
        "disclosure_version": "disclosure-v1",
    }


def submission(now=None):
    now = now or datetime.now(timezone.utc)
    raw = {
        "schema_version": 2, "release_id": "release-1", "revision_id": "revision-1",
        "source_commit": "b" * 40, "environment": "test", "reviewed_payload_hash": "0" * 64,
        "operator_review_id": "review-1", "operator_reviewed_at": now.isoformat(),
        "product_code": "monthly-qualified-straights", "recommendations": [recommendation(now)],
        "authority": {
            "schema_version": 1, "authority_id": "authority-1", "source_hash": "c" * 64,
            "upstream_decision_reference": "decision-1", "upstream_gate_result": "APPROVED",
            "market_status": "QUALIFIED", "activation_reference": "activation-1",
            "validation_artifact_id": "validation-1", "calibration_id": "cal-1",
            "content_rights_references": ["rights-1"], "effective_at": (now - timedelta(hours=1)).isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(), "revoked": False,
        },
    }
    parsed = ReleaseSubmission.model_validate(raw)
    raw["reviewed_payload_hash"] = object_hash(parsed.review_payload())
    return raw


def test_checkout_is_fail_closed_without_owner_gates():
    result = checkout_decision(
        engineering_status="NOT_VERIFIED", commercial_status="PENDING", sales_status="DISABLED",
        configured_terms=False, qualified_markets=[], commercially_enabled_markets=[], current_approvals=[],
        billing_mode="test", live_billing_enabled=False,
    )
    assert not result.allowed
    assert "NO_COMMERCIALLY_ENABLED_QUALIFIED_MARKET" in result.reason_codes
    assert "COMMERCIAL_APPROVAL_PENDING" in result.reason_codes


def test_unknown_or_stale_authority_cannot_be_promoted():
    raw = submission()
    raw["authority"]["market_status"] = "RESEARCH"
    parsed = ReleaseSubmission.model_validate(raw)
    raw["reviewed_payload_hash"] = object_hash(parsed.review_payload())
    signature = sign(raw, "secret")
    with pytest.raises(AuthorityError, match="MARKET_NOT_QUALIFIED"):
        verify_reviewed_submission(raw, signature=signature, secret="secret", environment="test")


def test_review_hash_binds_every_recommendation_fact():
    raw = submission()
    signature = sign(raw, "secret")
    assert verify_reviewed_submission(raw, signature=signature, secret="secret", environment="test")
    changed = copy.deepcopy(raw)
    changed["recommendations"][0]["line"] = 4.0
    with pytest.raises(AuthorityError, match="REVIEW_HASH_MISMATCH"):
        verify_reviewed_submission(changed, signature=sign(changed, "secret"), secret="secret", environment="test")


def test_probability_semantics_and_supported_market_are_enforced():
    now = datetime.now(timezone.utc)
    bad = recommendation(now)
    bad["p_loss"] = 0.2
    with pytest.raises(ValueError, match="must equal 1"):
        Recommendation.model_validate(bad)
    bad = recommendation(now)
    bad["exact_market_family"] = "MONEYLINE"
    with pytest.raises(ValueError, match="unsupported exact market"):
        Recommendation.model_validate(bad)


def test_probability_semantics_reject_unknown_and_conditional_inputs():
    now = datetime.now(timezone.utc)
    for semantics in ("UNKNOWN", "win_conditional_on_decision", "conditional"):
        bad = recommendation(now)
        bad["probability_semantics"] = semantics
        with pytest.raises(ValueError, match="unsupported probability semantics"):
            Recommendation.model_validate(bad)


def test_mean_and_conservative_ev_are_distinct_and_push_aware():
    now = datetime.now(timezone.utc)
    parsed = Recommendation.model_validate(recommendation(now))
    assert parsed.mean_ev_per_unit == pytest.approx(0.55 * 1.91 + 0.02 - 1)
    assert parsed.conservative_ev_per_unit == pytest.approx(0.52 * 1.91 + 0.02 - 1)
    assert parsed.conservative_ev_per_unit < parsed.mean_ev_per_unit

    bad = recommendation(now)
    bad["conservative_ev_per_unit"] = bad["mean_ev_per_unit"]
    with pytest.raises(ValueError, match="conservative EV"):
        Recommendation.model_validate(bad)


def test_conservative_win_bound_cannot_exceed_mean_probability():
    now = datetime.now(timezone.utc)
    bad = recommendation(now)
    bad["p_win_conservative"] = 0.56
    bad["conservative_ev_per_unit"] = 0.56 * bad["odds_decimal"] + bad["p_push"] - 1
    with pytest.raises(ValueError, match="cannot exceed"):
        Recommendation.model_validate(bad)


def test_subscriber_ev_matches_isolated_core_price_helper():
    from core.price_value import price_value

    now = datetime.now(timezone.utc)
    parsed = Recommendation.model_validate(recommendation(now))
    mean = price_value(parsed.p_win, parsed.p_push, parsed.odds_decimal)
    conservative = price_value(
        parsed.p_win_conservative, parsed.p_push, parsed.odds_decimal
    )
    assert mean is not None and conservative is not None
    assert parsed.mean_ev_per_unit == pytest.approx(mean["expected_value"])
    assert parsed.conservative_ev_per_unit == pytest.approx(
        conservative["expected_value"]
    )


def test_legacy_v1_release_is_not_silently_reinterpreted():
    raw = submission()
    raw["schema_version"] = 1
    raw["recommendations"][0]["schema_version"] = 1
    with pytest.raises(ValueError):
        ReleaseSubmission.model_validate(raw)


def test_blocked_upstream_result_cannot_release_with_qualified_market():
    raw = submission()
    raw["authority"]["upstream_gate_result"] = "BLOCKED"
    parsed = ReleaseSubmission.model_validate(raw)
    raw["reviewed_payload_hash"] = object_hash(parsed.review_payload())
    with pytest.raises(AuthorityError, match="UPSTREAM_GATE_NOT_APPROVED"):
        verify_reviewed_submission(
            raw, signature=sign(raw, "secret"), secret="secret", environment="test"
        )


def test_valid_signature_cannot_hide_invalid_probability_math():
    raw = submission()
    raw["recommendations"][0]["mean_ev_per_unit"] = 99.0
    with pytest.raises(ValueError, match="mean EV"):
        verify_reviewed_submission(
            raw, signature=sign(raw, "secret"), secret="secret", environment="test"
        )


def test_changed_calibration_id_invalidates_bound_release_scope():
    raw = submission()
    raw["recommendations"][0]["calibration_id"] = "cal-2"
    with pytest.raises(ValueError, match="authority calibration"):
        verify_reviewed_submission(
            raw, signature=sign(raw, "secret"), secret="secret", environment="test"
        )


def test_live_key_is_rejected_when_live_billing_is_disabled():
    environment = {
        "PAID_DATABASE_URL": "postgresql://example", "PAID_ENVIRONMENT": "production",
        "PAID_PUBLIC_BASE_URL": "https://subscriber.example", "PAID_ALLOWED_ORIGINS": "https://subscriber.example",
        "PAID_ALLOWED_RETURN_URLS": "https://subscriber.example/account", "PAID_OIDC_ISSUER": "https://issuer.example",
        "PAID_OIDC_CLIENT_ID": "client", "PAID_OIDC_CLIENT_SECRET": "secret",
        "PAID_OIDC_CALLBACK_URL": "https://subscriber.example/auth/callback", "PAID_OIDC_ADMIN_GROUP": "owners",
        "PAID_RELEASE_HMAC_SECRET": "secret", "PAID_GATEWAY_PROBE_TOKEN": "secret",
        "PAID_STRIPE_SECRET_KEY": "sk_live_forbidden", "PAID_LIVE_BILLING_ENABLED": "false",
    }
    with pytest.raises(ValueError, match="live Stripe key is forbidden"):
        Settings.from_env(environment)
