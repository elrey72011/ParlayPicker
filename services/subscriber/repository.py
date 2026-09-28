"""Transactional customer-domain operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .canonical import object_hash
from .contracts import AuthorityBinding, ReleaseSubmission
from .db import Database, new_id


class ConflictError(RuntimeError):
    pass


class SubscriberRepository:
    def __init__(self, db: Database, environment: str):
        self.db = db
        self.environment = environment

    def service_state(self) -> dict[str, Any]:
        rows = self.db.fetch_all("SELECT key,value FROM subscriber.service_state")
        values = {row["key"]: row["value"] for row in rows}
        return {
            "engineering_status": values.get("engineering_status", "NOT_VERIFIED"),
            "commercial_status": values.get("commercial_status", "PENDING"),
            "sales_status": values.get("sales_status", "DISABLED"),
            "configured_terms": values.get("configured_terms", False),
            "qualified_markets": values.get("qualified_markets", []),
            "commercially_enabled_markets": values.get("commercially_enabled_markets", []),
            "release_status": values.get("release_status", "PAUSED"),
            "alerts_status": values.get("alerts_status", "PAUSED"),
        }

    def current_approvals(self, scope: str = "paid-launch") -> set[str]:
        rows = self.db.fetch_all(
            """
            SELECT DISTINCT ON (approval_type) approval_type,status,expires_at
            FROM subscriber.approval_record WHERE scope=%s ORDER BY approval_type,reviewed_at DESC
            """,
            (scope,),
        )
        now = datetime.now(timezone.utc)
        return {
            row["approval_type"] for row in rows
            if row["status"] == "APPROVED" and (row["expires_at"] is None or row["expires_at"] > now)
        }

    def active_product(self, product_code: str | None = None) -> dict[str, Any] | None:
        clause, params = "", (self.environment,)
        if product_code:
            clause, params = " AND product_code=%s", (self.environment, product_code)
        row = self.db.fetch_one(
            "SELECT * FROM subscriber.product_version WHERE environment=%s AND active=true" + clause + " ORDER BY version DESC LIMIT 1",
            params,
        )

    def entitlement(self, customer_id: object, product_id: object | None = None) -> dict[str, Any] | None:
        sql = """
          SELECT e.*,p.product_code,p.terms_version FROM subscriber.entitlement e
          JOIN subscriber.product_version p ON p.id=e.product_version_id
          WHERE e.customer_id=%s AND e.revoked_at IS NULL AND e.effective_at<=now() AND e.expires_at>now()
        """
        params: tuple[object, ...] = (customer_id,)
        if product_id:
            sql += " AND e.product_version_id=%s"
            params += (product_id,)
        sql += " ORDER BY e.revision DESC LIMIT 1"
        return self.db.fetch_one(sql, params)

    def customer_account(self, customer_id: object) -> dict[str, Any]:
        customer = self.db.fetch_one(
            "SELECT id,contact_email,email_verified,status,role,created_at FROM subscriber.customer WHERE id=%s",
            (customer_id,),
        )
        entitlement = self.entitlement(customer_id)
        subscription = self.db.fetch_one(
            "SELECT state,paid_through,cancel_at_period_end,updated_at FROM subscriber.subscription WHERE customer_id=%s ORDER BY updated_at DESC LIMIT 1",
            (customer_id,),
        )
        return {"customer": customer, "entitlement": entitlement, "subscription": subscription}

    def current_release(self, customer_id: object) -> dict[str, Any] | None:
        entitlement = self.entitlement(customer_id)
        if not entitlement:
            return None
        return self.db.fetch_one(
            """
            SELECT rr.id,rr.release_id,rr.revision_id,rr.payload,rr.customer_payload_hash,rr.promoted_at,rr.expires_at,rr.authority
            FROM subscriber.active_release ar
            JOIN subscriber.release_revision rr ON rr.id=ar.release_revision_id
            WHERE ar.product_code=%s AND ar.environment=%s AND ar.status='ACTIVE'
              AND rr.status='ACTIVE_REVISION_PROMOTED' AND rr.expires_at>now()
            ORDER BY ar.updated_at DESC LIMIT 1
            """,
            (entitlement["product_code"], self.environment),
        )
        if not row:
            return None
        authority = AuthorityBinding.model_validate(row["authority"])
        now = datetime.now(timezone.utc)
        if authority.revoked or authority.market_status.value != "QUALIFIED" or not (authority.effective_at <= now < authority.expires_at):
            return None
        return row

    def release_by_id(self, customer_id: object, release_id: str) -> dict[str, Any] | None:
        if not self.entitlement(customer_id):
            return None
        return self.db.fetch_one(
            """
            SELECT release_id,revision_id,payload,customer_payload_hash,promoted_at,expires_at,status
            FROM subscriber.release_revision
            WHERE release_id=%s AND environment=%s AND status IN ('ACTIVE_REVISION_PROMOTED','WITHDRAWN')
            ORDER BY created_at DESC LIMIT 1
            """,
            (release_id, self.environment),
        )

    def results(self, customer_id: object) -> list[dict[str, Any]]:
        if not self.entitlement(customer_id):
            return []
        return self.db.fetch_all(
            """
            SELECT rp.recommendation_id,rp.status,rp.paper_return,rp.settlement_rules_version,
                   rp.projection_revision,rp.created_at,rr.release_id,rr.revision_id
            FROM subscriber.result_projection rp JOIN subscriber.release_revision rr ON rr.id=rp.release_revision_id
            WHERE rr.environment=%s ORDER BY rp.created_at DESC
            """,
            (self.environment,),
        )

    def stage_release(self, submission: ReleaseSubmission, request_id: str, actor_id: object) -> dict[str, Any]:
        projection = submission.customer_projection()
        payload_hash = object_hash(projection)
        expires_at = min(item.expiry_at for item in submission.recommendations)
        authority_hash = object_hash(submission.authority.model_dump(mode="json"))
        release_db_id = new_id()
        with self.db.transaction() as connection:
            existing = connection.execute(
                "SELECT customer_payload_hash,status FROM subscriber.release_revision WHERE release_id=%s AND revision_id=%s AND environment=%s",
                (submission.release_id, submission.revision_id, self.environment),
            ).fetchone()
            if existing:
                if existing["customer_payload_hash"] != payload_hash:
                    raise ConflictError("REVISION_ID_ALREADY_BOUND_TO_DIFFERENT_PAYLOAD")
                return {"status": existing["status"], "customer_payload_hash": payload_hash}
            connection.execute(
                """
                INSERT INTO subscriber.release_revision(
                  id,release_id,revision_id,product_code,reviewed_payload_hash,customer_payload_hash,
                  source_commit,environment,operator_review_id,upstream_authority_id,upstream_authority_hash,authority,
                  status,payload,expires_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING_REVISION_COMMITTED',%s,%s)
                """,
                (
                    release_db_id, submission.release_id, submission.revision_id, submission.product_code,
                    submission.reviewed_payload_hash, payload_hash, submission.source_commit, self.environment,
                    submission.operator_review_id, submission.authority.authority_id, authority_hash,
                    submission.authority.model_dump(mode="json"), projection, expires_at,
                ),
            )
            connection.execute(
                "INSERT INTO subscriber.release_event(id,release_revision_id,event_type,actor_customer_id,reason,request_id,facts) VALUES (%s,%s,'STAGED',%s,'reviewed package accepted',%s,%s)",
                (new_id(), release_db_id, actor_id, request_id, {"customer_payload_hash": payload_hash}),
            )
            connection.execute(
                "INSERT INTO subscriber.job_queue(id,job_type,deduplication_key,payload,status) VALUES (%s,'VERIFY_RELEASE_ROUTE',%s,%s,'PENDING')",
                (new_id(), f"verify-release:{release_db_id}", {"release_revision_id": str(release_db_id)}),
            )
        return {"status": "PENDING_REVISION_COMMITTED", "customer_payload_hash": payload_hash}

    def withdraw_release(self, release_id: str, reason: str, request_id: str, actor_id: object) -> bool:
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT id,status FROM subscriber.release_revision WHERE release_id=%s AND environment=%s ORDER BY created_at DESC LIMIT 1 FOR UPDATE",
                (release_id, self.environment),
            ).fetchone()
            if not row:
                return False
            connection.execute("UPDATE subscriber.release_revision SET status='WITHDRAWN' WHERE id=%s", (row["id"],))
            connection.execute("UPDATE subscriber.active_release SET status='SUSPENDED',updated_at=now() WHERE release_revision_id=%s", (row["id"],))
            connection.execute(
                "UPDATE subscriber.notification_outbox SET status='SUPPRESSED',suppression_reason='RELEASE_WITHDRAWN' WHERE release_revision_id=%s AND status IN ('PENDING','FAILED')",
                (row["id"],),
            )
            connection.execute(
                "INSERT INTO subscriber.release_event(id,release_revision_id,event_type,actor_customer_id,reason,request_id) VALUES (%s,%s,'WITHDRAWN',%s,%s,%s)",
                (new_id(), row["id"], actor_id, reason, request_id),
            )
            return True

    def promote_verified_release(self, release_revision_id: object, observed_hash: str, request_id: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM subscriber.release_revision WHERE id=%s FOR UPDATE",
                (release_revision_id,),
            ).fetchone()
            if not row:
                raise ValueError("RELEASE_REVISION_NOT_FOUND")
            if row["status"] == "ACTIVE_REVISION_PROMOTED":
                return {"status": row["status"], "customer_payload_hash": row["customer_payload_hash"]}
            if row["status"] != "PENDING_REVISION_COMMITTED":
                raise ConflictError("RELEASE_REVISION_NOT_PROMOTABLE")
            if observed_hash != row["customer_payload_hash"] or object_hash(row["payload"]) != row["customer_payload_hash"]:
                connection.execute("UPDATE subscriber.release_revision SET status='INVALIDATED' WHERE id=%s", (row["id"],))
                connection.execute(
                    "INSERT INTO subscriber.release_event(id,release_revision_id,event_type,reason,request_id,facts) VALUES (%s,%s,'ROUTED_HASH_MISMATCH','gateway bytes do not match staged revision',%s,%s)",
                    (new_id(), row["id"], request_id, {"observed_hash": observed_hash}),
                )
                raise ConflictError("ROUTED_HASH_MISMATCH")
            authority = AuthorityBinding.model_validate(row["authority"])
            if authority.revoked or authority.market_status.value != "QUALIFIED" or not (authority.effective_at <= now < authority.expires_at):
                raise ConflictError("AUTHORITY_NOT_CURRENT_AT_PROMOTION")
            if row["expires_at"] <= now:
                raise ConflictError("RELEASE_EXPIRED_BEFORE_PROMOTION")
            active = connection.execute(
                "SELECT ar.*,rr.created_at AS active_created_at FROM subscriber.active_release ar JOIN subscriber.release_revision rr ON rr.id=ar.release_revision_id WHERE ar.product_code=%s AND ar.scope='qualified-straight-picks' AND ar.environment=%s FOR UPDATE",
                (row["product_code"], self.environment),
            ).fetchone()
            if active and active["active_created_at"] >= row["created_at"]:
                raise ConflictError("STALE_RELEASE_CANNOT_REPLACE_ACTIVE_REVISION")
            pointer_revision = (active["pointer_revision"] + 1) if active else 1
            connection.execute(
                "UPDATE subscriber.release_revision SET status='ACTIVE_REVISION_PROMOTED',verified_at=%s,promoted_at=%s WHERE id=%s",
                (now, now, row["id"]),
            )
            connection.execute(
                """
                INSERT INTO subscriber.active_release(product_code,scope,environment,release_revision_id,pointer_revision,status)
                VALUES (%s,'qualified-straight-picks',%s,%s,%s,'ACTIVE')
                ON CONFLICT (product_code,scope,environment) DO UPDATE SET
                  release_revision_id=excluded.release_revision_id,pointer_revision=excluded.pointer_revision,status='ACTIVE',updated_at=now()
                """,
                (row["product_code"], self.environment, row["id"], pointer_revision),
            )
            connection.execute(
                "INSERT INTO subscriber.release_event(id,release_revision_id,event_type,reason,request_id,facts) VALUES (%s,%s,'ROUTED_VERSION_VERIFIED','gateway hash verified and active pointer promoted',%s,%s)",
                (new_id(), row["id"], request_id, {"customer_payload_hash": observed_hash, "pointer_revision": pointer_revision}),
            )
            connection.execute(
                """
                INSERT INTO subscriber.notification_outbox(
                  id,recipient_customer_id,release_revision_id,notification_type,channel,status,next_attempt_at)
                SELECT gen_random_uuid(),e.customer_id,%s,'NEW_RELEASE','EMAIL','PENDING',now()
                FROM subscriber.entitlement e JOIN subscriber.customer c ON c.id=e.customer_id
                JOIN subscriber.product_version p ON p.id=e.product_version_id
                WHERE p.product_code=%s AND e.revoked_at IS NULL AND e.effective_at<=now() AND e.expires_at>now()
                  AND c.status='ACTIVE' AND c.email_verified=true
                ON CONFLICT (recipient_customer_id,release_revision_id,notification_type,channel) DO NOTHING
                """,
                (row["id"], row["product_code"]),
            )
            return {"status": "ACTIVE_REVISION_PROMOTED", "customer_payload_hash": observed_hash}

    def record_billing_event(
        self,
        event: dict[str, Any],
        payload_hash: str,
        account_id: str,
        event_environment: str,
        expected_billing_environment: str,
    ) -> bool:
        event_id, event_type = str(event.get("id", "")), str(event.get("type", ""))
        if not event_id or not event_type:
            raise ValueError("BILLING_EVENT_ID_OR_TYPE_MISSING")
        with self.db.transaction() as connection:
            inserted = connection.execute(
                """
                INSERT INTO subscriber.billing_event(id,provider,provider_account_id,environment,provider_event_id,event_type,payload_hash,payload,status)
                VALUES (%s,'stripe',%s,%s,%s,%s,%s,%s,'RECEIVED')
                ON CONFLICT (provider,provider_account_id,environment,provider_event_id) DO NOTHING RETURNING id
                """,
                (new_id(), account_id, event_environment, event_id, event_type, payload_hash, event),
            ).fetchone()
            if not inserted:
                return False
            if event_environment != expected_billing_environment:
                connection.execute(
                    "UPDATE subscriber.billing_event SET status='QUARANTINED',sanitized_error='environment mismatch' WHERE id=%s",
                    (inserted["id"],),
                )
                return True
            connection.execute(
                "INSERT INTO subscriber.job_queue(id,job_type,deduplication_key,payload,status) VALUES (%s,'PROCESS_BILLING_EVENT',%s,%s,'PENDING')",
                (new_id(), f"billing-event:{inserted['id']}", {"billing_event_id": str(inserted["id"])}),
            )
            return True
