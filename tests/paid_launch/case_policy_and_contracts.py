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


# Synthetic signed OIDC tokens exercise the service's real JOSE interfaces.
@pytest.fixture(scope="module")
def authlib_rsa_keys():
    from authlib.jose import JsonWebKey
    key = JsonWebKey.generate_key("RSA", 2048, is_private=True, options={"kid": "synthetic-authlib"})
    return key, {"keys": [key.as_dict(is_private=False)]}


def authlib_complete_fixture(monkeypatch, keys, changes=None, *, signature="valid"):
    """Only HTTP and database boundaries are synthetic; JOSE is unmodified."""
    import asyncio
    from contextlib import contextmanager
    from types import SimpleNamespace
    import httpx
    from authlib.jose import jwt
    from services.subscriber import auth

    now = datetime.now(timezone.utc)
    claims = {"iss": "https://issuer.example.test", "aud": "synthetic-client", "sub": "synthetic-subject",
              "nonce": "synthetic-nonce", "exp": int(now.timestamp()) + 600,
              "iat": int(now.timestamp()) - 60, "groups": [], "amr": [],
              "email": "synthetic@example.test", "email_verified": True}
    claims.update(changes or {})
    for name in [name for name, value in claims.items() if value is None]:
        del claims[name]
    encoded = jwt.encode({"alg": "RS256", "kid": "synthetic-authlib"}, claims, keys[0]).decode()
    if signature == "invalid":
        parts = encoded.split(".")
        parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
        encoded = ".".join(parts)
    config = SimpleNamespace(oidc_issuer="https://issuer.example.test", oidc_client_id="synthetic-client",
        oidc_client_secret="synthetic-only", oidc_callback_url="https://app.example.test/auth/callback",
        oidc_admin_group="owners", session_hours=12)
    created = []; requests = []; state_updates = []

    class Login:
        def fetchone(self):
            return {"consumed_at": None, "expires_at": now + timedelta(minutes=10),
                    "nonce": "synthetic-nonce", "code_verifier": "synthetic-pkce",
                    "redirect_after": "https://app.example.test/account"}

    class Database:
        @contextmanager
        def transaction(self):
            yield self

        def execute(self, sql, params):
            if sql.startswith("SELECT * FROM subscriber.oidc_login"):
                return Login()
            assert sql.startswith("UPDATE subscriber.oidc_login SET consumed_at=")
            state_updates.append(params)

    def create_session(db, **kwargs):
        created.append(kwargs)
        return "synthetic-session", "synthetic-csrf"

    monkeypatch.setattr(auth, "create_customer_session", create_session)

    def transport(request):
        requests.append(request)
        assert request.url.host == "issuer.example.test"
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={"issuer": config.oidc_issuer,
                "token_endpoint": config.oidc_issuer + "/token", "jwks_uri": config.oidc_issuer + "/jwks"})
        if request.url.path == "/token":
            assert request.method == "POST" and b"code_verifier=synthetic-pkce" in request.content
            return httpx.Response(200, json={"id_token": encoded})
        assert request.url.path == "/jwks"
        return httpx.Response(200, json=keys[1])

    async def complete():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            return await auth.OidcClient(config, client).complete(Database(), state="synthetic-state", code="synthetic-code")

    return lambda: asyncio.run(complete()), created, requests, state_updates


@pytest.mark.parametrize("claims,role,strength", [
    ({}, "CUSTOMER", "oidc"),
    ({"groups": ["owners"], "amr": ["mfa"]}, "OWNER", "mfa"),
    ({"groups": ["owners"], "amr": ["pwd"]}, "CUSTOMER", "oidc"),
    ({"groups": ["customers"], "amr": ["mfa"], "role": "OWNER", "admin": True}, "CUSTOMER", "oidc"),
    ({"groups": ["not-owners"], "amr": ["mfa"]}, "CUSTOMER", "oidc"),
    ({"aud": ["another-client", "synthetic-client"]}, "CUSTOMER", "oidc"),
])
def test_authlib_real_oidc_valid_and_privilege_boundaries(monkeypatch, authlib_rsa_keys, claims, role, strength):
    run, created, requests, state = authlib_complete_fixture(monkeypatch, authlib_rsa_keys, claims)
    assert run() == ("synthetic-session", "synthetic-csrf", "https://app.example.test/account")
    assert len(created) == 1 and created[0]["role"] == role and created[0]["auth_strength"] == strength
    assert created[0]["issuer"] == "https://issuer.example.test" and created[0]["subject"] == "synthetic-subject"
    assert len(requests) == 3 and len(state) == 1


@pytest.mark.parametrize("claims,signature,error", [
    ({}, "invalid", "BadSignatureError"),
    ({"iss": "https://other.example.test"}, "valid", "InvalidClaimError"),
    ({"aud": "other-client"}, "valid", "InvalidClaimError"),
    ({"nonce": "other-nonce"}, "valid", "InvalidClaimError"),
    ({"nonce": "other-nonce", "nonce_supported": False}, "valid", "InvalidClaimError"),
    ({"nonce": None}, "valid", "MissingClaimError"),
    ({"exp": 1}, "valid", "ExpiredTokenError"),
    ({"exp": None}, "valid", "MissingClaimError"),
    ({"exp": True}, "valid", "InvalidClaimError"),
    ({"nbf": True}, "valid", "InvalidClaimError"),
    ({"iat": True}, "valid", "InvalidClaimError"),
])
def test_authlib_real_oidc_rejects_invalid_token(monkeypatch, authlib_rsa_keys, claims, signature, error):
    from authlib.jose import errors
    run, created, requests, state = authlib_complete_fixture(monkeypatch, authlib_rsa_keys, claims, signature=signature)
    with pytest.raises(getattr(errors, error)):
        run()
    assert created == [] and len(requests) == 3 and len(state) == 1


@pytest.mark.parametrize("claims", [
    {"groups": "not-owners", "amr": ["mfa"]},
    {"groups": {"owners": True}, "amr": ["mfa"]},
    {"groups": ["owners"], "amr": "mfa"},
    {"groups": ["owners"], "amr": ["mfa", 1]},
])
def test_authlib_real_oidc_rejects_malformed_privilege_claims(monkeypatch, authlib_rsa_keys, claims):
    run, created, _, state = authlib_complete_fixture(monkeypatch, authlib_rsa_keys, claims)
    with pytest.raises(ValueError, match="OIDC_PRIVILEGE_CLAIMS_INVALID"):
        run()
    assert created == [] and len(state) == 1


def test_authlib_empty_hmac_and_asymmetric_key_confusion_rejected(authlib_rsa_keys):
    import base64,hashlib,hmac,json
    from authlib.jose import jwt
    encode = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=")
    signing = encode(b'{"alg":"HS256"}') + b"." + encode(b'{"sub":"synthetic"}')
    forged = signing + b"." + encode(hmac.new(b"", signing, hashlib.sha256).digest())
    with pytest.raises(ValueError):
        jwt.decode(forged, b"")
    public = authlib_rsa_keys[0].as_pem(is_private=False)
    with pytest.raises(ValueError):
        jwt.encode({"alg": "HS256"}, {"sub": "synthetic"}, public)


def test_authlib_empty_signature_collection_rejected(authlib_rsa_keys):
    from authlib.jose import JsonWebSignature
    from authlib.jose.errors import BadSignatureError,DecodeError
    with pytest.raises((BadSignatureError, DecodeError, ValueError)):
        JsonWebSignature().deserialize_json({"payload": "e30", "signatures": []}, authlib_rsa_keys[0])


@pytest.mark.parametrize("relative,expected", [('services/subscriber/requirements.txt', 'fe8228774ff9e7b34c8bb2d88000ef4233d71bfc5cfa5bb4a9811c33f314366d'), ('services/subscriber/auth.py', '9a01cdee0153a81a54809115f534f9e4114a53b017efdba3499a6833ad769159'), ('tests/paid_launch/case_policy_and_contracts.py', 'fcd7c95a43aa796aa8605df690084c612578ec5cbc9ee6f4ec9116d8b22f12bf')])
def test_authlib_successor_preserves_independent_predecessor_bytes(relative, expected):
    import hashlib
    from pathlib import Path
    from scripts import ncaaf_filtered_history_scope as history
    root = Path(__file__).resolve().parents[2]
    binding = history.authenticate(root)
    current = history.physical(root / relative)
    prior = history.capacity_predecessor(binding, relative, current)
    assert hashlib.sha256(prior).hexdigest() == expected
    assert prior != current
    changed = current + b"SYNTHETIC_UNAUTHORIZED_CHANGE"
    assert history.capacity_predecessor(binding, relative, changed) == changed
