BEGIN;

CREATE SCHEMA IF NOT EXISTS subscriber;

CREATE TABLE IF NOT EXISTS subscriber.schema_migration (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS subscriber.customer (
    id uuid PRIMARY KEY,
    oidc_issuer text NOT NULL,
    oidc_subject text NOT NULL,
    contact_email text,
    email_verified boolean NOT NULL DEFAULT false,
    status text NOT NULL CHECK (status IN ('ACTIVE','DISABLED','DELETED')),
    role text NOT NULL DEFAULT 'CUSTOMER' CHECK (role IN ('CUSTOMER','OPERATOR','OWNER')),
    alerts_enabled boolean NOT NULL DEFAULT true,
    email_suppressed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (oidc_issuer, oidc_subject)
);

CREATE TABLE IF NOT EXISTS subscriber.oidc_login (
    state_hash char(64) PRIMARY KEY,
    nonce text NOT NULL,
    code_verifier text NOT NULL,
    redirect_after text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    consumed_at timestamptz
);

CREATE TABLE IF NOT EXISTS subscriber.session (
    id uuid PRIMARY KEY,
    token_hash char(64) NOT NULL UNIQUE,
    csrf_hash char(64) NOT NULL,
    customer_id uuid NOT NULL REFERENCES subscriber.customer(id),
    auth_strength text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    user_agent_hash char(64)
);

CREATE TABLE IF NOT EXISTS subscriber.product_version (
    id uuid PRIMARY KEY,
    product_code text NOT NULL,
    version integer NOT NULL,
    offered_markets text[] NOT NULL DEFAULT '{}',
    provider text NOT NULL,
    provider_price_id text NOT NULL,
    currency char(3) NOT NULL,
    amount_minor integer NOT NULL CHECK (amount_minor > 0),
    selling_entity text NOT NULL,
    eligible_jurisdictions text[] NOT NULL,
    terms_version text NOT NULL,
    renewal_disclosure_version text NOT NULL,
    cancellation_policy_version text NOT NULL,
    refund_policy_version text NOT NULL,
    environment text NOT NULL CHECK (environment IN ('test','staging','production')),
    approval_references jsonb NOT NULL DEFAULT '{}',
    active boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (product_code, version, environment),
    UNIQUE (provider, provider_price_id, environment)
);

CREATE TABLE IF NOT EXISTS subscriber.subscription (
    id uuid PRIMARY KEY,
    customer_id uuid NOT NULL REFERENCES subscriber.customer(id),
    product_version_id uuid NOT NULL REFERENCES subscriber.product_version(id),
    provider text NOT NULL,
    provider_account_id text NOT NULL,
    environment text NOT NULL,
    provider_customer_id text NOT NULL,
    provider_subscription_id text NOT NULL,
    state text NOT NULL CHECK (state IN ('INCOMPLETE','ACTIVE','PAST_DUE','CANCEL_AT_PERIOD_END','CANCELLED','EXPIRED','DISPUTED')),
    provider_version text,
    paid_through timestamptz,
    cancel_at_period_end boolean NOT NULL DEFAULT false,
    cancellation_requested_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (provider, provider_account_id, environment, provider_subscription_id)
);
CREATE INDEX IF NOT EXISTS subscription_provider_customer_idx
    ON subscriber.subscription(provider,provider_account_id,environment,provider_customer_id);

CREATE TABLE IF NOT EXISTS subscriber.billing_event (
    id uuid PRIMARY KEY,
    provider text NOT NULL,
    provider_account_id text NOT NULL,
    environment text NOT NULL,
    provider_event_id text NOT NULL,
    event_type text NOT NULL,
    payload_hash char(64) NOT NULL,
    payload jsonb NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    processed_at timestamptz,
    status text NOT NULL CHECK (status IN ('RECEIVED','PROCESSING','PROCESSED','RETRY','DEAD_LETTER','QUARANTINED')),
    attempts integer NOT NULL DEFAULT 0,
    next_attempt_at timestamptz,
    sanitized_error text,
    UNIQUE (provider, provider_account_id, environment, provider_event_id)
);

CREATE TABLE IF NOT EXISTS subscriber.entitlement (
    id uuid PRIMARY KEY,
    customer_id uuid NOT NULL REFERENCES subscriber.customer(id),
    product_version_id uuid NOT NULL REFERENCES subscriber.product_version(id),
    subscription_id uuid REFERENCES subscriber.subscription(id),
    source_billing_event_id uuid REFERENCES subscriber.billing_event(id),
    effective_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    revision integer NOT NULL,
    revoked_at timestamptz,
    reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (customer_id, product_version_id, revision),
    CHECK (expires_at > effective_at)
);

CREATE TABLE IF NOT EXISTS subscriber.approval_record (
    id uuid PRIMARY KEY,
    approval_type text NOT NULL,
    scope text NOT NULL,
    status text NOT NULL CHECK (status IN ('APPROVED','REVOKED','PENDING')),
    reviewer text NOT NULL,
    evidence_reference text NOT NULL,
    evidence_hash char(64) NOT NULL,
    reviewed_at timestamptz NOT NULL,
    expires_at timestamptz,
    revoked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS approval_current_idx ON subscriber.approval_record (approval_type, scope, reviewed_at DESC);

CREATE TABLE IF NOT EXISTS subscriber.release_revision (
    id uuid PRIMARY KEY,
    release_id text NOT NULL,
    revision_id text NOT NULL,
    product_code text NOT NULL,
    reviewed_payload_hash char(64) NOT NULL,
    customer_payload_hash char(64) NOT NULL,
    source_commit char(40) NOT NULL,
    environment text NOT NULL,
    operator_review_id text NOT NULL,
    upstream_authority_id text NOT NULL,
    upstream_authority_hash char(64) NOT NULL,
    authority jsonb NOT NULL,
    status text NOT NULL CHECK (status IN ('PENDING_REVISION_COMMITTED','ROUTED_VERSION_VERIFIED','ACTIVE_REVISION_PROMOTED','INVALIDATED','WITHDRAWN')),
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    verified_at timestamptz,
    promoted_at timestamptz,
    expires_at timestamptz NOT NULL,
    UNIQUE (release_id, revision_id, environment),
    UNIQUE (customer_payload_hash, environment)
);

CREATE TABLE IF NOT EXISTS subscriber.active_release (
    product_code text NOT NULL,
    scope text NOT NULL,
    environment text NOT NULL,
    release_revision_id uuid NOT NULL REFERENCES subscriber.release_revision(id),
    pointer_revision bigint NOT NULL,
    status text NOT NULL CHECK (status IN ('ACTIVE','SUSPENDED','UNAVAILABLE')),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (product_code, scope, environment)
);

CREATE TABLE IF NOT EXISTS subscriber.release_event (
    id uuid PRIMARY KEY,
    release_revision_id uuid NOT NULL REFERENCES subscriber.release_revision(id),
    event_type text NOT NULL,
    actor_customer_id uuid REFERENCES subscriber.customer(id),
    reason text NOT NULL,
    request_id text NOT NULL,
    facts jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS subscriber.result_projection (
    id uuid PRIMARY KEY,
    recommendation_id text NOT NULL,
    release_revision_id uuid NOT NULL REFERENCES subscriber.release_revision(id),
    source_settlement_reference text NOT NULL,
    source_settlement_hash char(64) NOT NULL,
    settlement_rules_version text NOT NULL,
    status text NOT NULL CHECK (status IN ('PENDING','WIN','LOSS','PUSH','VOID','NEEDS_REVIEW','CORRECTED')),
    paper_return numeric,
    projection_revision integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (recommendation_id, release_revision_id, projection_revision)
);

CREATE TABLE IF NOT EXISTS subscriber.notification_outbox (
    id uuid PRIMARY KEY,
    recipient_customer_id uuid NOT NULL REFERENCES subscriber.customer(id),
    release_revision_id uuid REFERENCES subscriber.release_revision(id),
    notification_type text NOT NULL,
    channel text NOT NULL,
    status text NOT NULL CHECK (status IN ('PENDING','SENDING','ACCEPTED_BY_PROVIDER','DELIVERED','FAILED','SUPPRESSED','UNKNOWN_OUTCOME','DEAD_LETTER')),
    attempts integer NOT NULL DEFAULT 0,
    next_attempt_at timestamptz,
    provider_message_id text,
    suppression_reason text,
    sanitized_error text,
    lease_owner text,
    lease_expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (recipient_customer_id, release_revision_id, notification_type, channel)
);

CREATE TABLE IF NOT EXISTS subscriber.job_queue (
    id uuid PRIMARY KEY,
    job_type text NOT NULL,
    deduplication_key text NOT NULL UNIQUE,
    payload jsonb NOT NULL,
    status text NOT NULL CHECK (status IN ('PENDING','RUNNING','DONE','RETRY','DEAD_LETTER')),
    attempts integer NOT NULL DEFAULT 0,
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    lease_owner text,
    lease_expires_at timestamptz,
    sanitized_error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS subscriber.admin_audit (
    id uuid PRIMARY KEY,
    actor_customer_id uuid REFERENCES subscriber.customer(id),
    action text NOT NULL,
    scoped_target text NOT NULL,
    before_reference text,
    after_reference text,
    reason text NOT NULL,
    request_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS subscriber.consent_record (
    id uuid PRIMARY KEY,
    customer_id uuid NOT NULL REFERENCES subscriber.customer(id),
    product_version_id uuid REFERENCES subscriber.product_version(id),
    terms_version text NOT NULL,
    renewal_disclosure_version text NOT NULL,
    cancellation_policy_version text NOT NULL,
    refund_policy_version text NOT NULL,
    consented_at timestamptz NOT NULL,
    request_id text NOT NULL
);

CREATE TABLE IF NOT EXISTS subscriber.service_state (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO subscriber.schema_migration(version) VALUES ('0001_customer_service') ON CONFLICT DO NOTHING;
COMMIT;
