"""Lease-based worker for release, billing, cancellation and notifications."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import email.message
import json
import os
import smtplib
import socket
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import httpx

from ..billing import StripeBilling
from ..canonical import object_hash
from ..db import Database, new_id
from ..repository import SubscriberRepository
from ..settings import Settings


MAX_ATTEMPTS = 8


class Worker:
    def __init__(self, settings: Settings, worker_id: str | None = None):
        self.settings = settings
        self.db = Database(settings.database_url)
        self.repository = SubscriberRepository(self.db, settings.environment)
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def lease_job(self) -> dict[str, Any] | None:
        with self.db.transaction() as connection:
            job = connection.execute(
                """
                SELECT * FROM subscriber.job_queue
                WHERE status IN ('PENDING','RETRY') AND next_attempt_at<=now()
                  AND (lease_expires_at IS NULL OR lease_expires_at<now())
                ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
                """
            ).fetchone()
            if not job:
                return None
            connection.execute(
                "UPDATE subscriber.job_queue SET status='RUNNING',lease_owner=%s,lease_expires_at=now()+interval '2 minutes',attempts=attempts+1 WHERE id=%s",
                (self.worker_id, job["id"]),
            )
            job["attempts"] += 1
            return job

    def complete(self, job_id: object) -> None:
        self.db.execute(
            "UPDATE subscriber.job_queue SET status='DONE',completed_at=now(),lease_owner=NULL,lease_expires_at=NULL,sanitized_error=NULL WHERE id=%s AND lease_owner=%s",
            (job_id, self.worker_id),
        )

    def fail(self, job: dict[str, Any], exc: Exception) -> None:
        attempts = int(job["attempts"])
        dead = attempts >= MAX_ATTEMPTS
        delay = min(3600, 2 ** attempts * 5)
        safe_error = type(exc).__name__ + ": " + str(exc)[:300]
        self.db.execute(
            """
            UPDATE subscriber.job_queue SET status=%s,next_attempt_at=%s,lease_owner=NULL,lease_expires_at=NULL,sanitized_error=%s
            WHERE id=%s AND lease_owner=%s
            """,
            (
                "DEAD_LETTER" if dead else "RETRY",
                datetime.now(timezone.utc) + timedelta(seconds=delay),
                safe_error,
                job["id"], self.worker_id,
            ),
        )

    def verify_release(self, payload: dict[str, Any]) -> None:
        release_id = uuid.UUID(payload["release_revision_id"])
        row = self.db.fetch_one(
            "SELECT revision_id,customer_payload_hash FROM subscriber.release_revision WHERE id=%s",
            (release_id,),
        )
        if not row:
            raise ValueError("release revision missing")
        if not self.settings.release_probe_url:
            raise ValueError("release gateway probe URL is not configured")
        base = self.settings.release_probe_url.rstrip("/")
        response = httpx.get(
            f"{base}/{row['revision_id']}",
            headers={"Authorization": "Bearer " + self.settings.gateway_probe_token},
            timeout=10,
        )
        response.raise_for_status()
        routed = response.json()
        if routed.get("content_hash") != row["customer_payload_hash"]:
            observed = str(routed.get("content_hash", ""))
        else:
            observed = row["customer_payload_hash"]
        self.repository.promote_verified_release(release_id, observed, str(uuid.uuid4()))

    @staticmethod
    def _subscription_id(event: dict[str, Any]) -> str | None:
        obj = ((event.get("data") or {}).get("object") or {})
        if obj.get("object") == "subscription":
            return str(obj.get("id") or "") or None
        subscription = obj.get("subscription")
        if isinstance(subscription, str):
            return subscription
        parent = obj.get("parent") or {}
        details = parent.get("subscription_details") or {}
        return str(details.get("subscription") or "") or None

    def process_billing_event(self, payload: dict[str, Any]) -> None:
        event_id = uuid.UUID(payload["billing_event_id"])
        with self.db.transaction() as connection:
            event_row = connection.execute(
                "SELECT * FROM subscriber.billing_event WHERE id=%s FOR UPDATE",
                (event_id,),
            ).fetchone()
            if not event_row or event_row["status"] in {"PROCESSED", "QUARANTINED"}:
                return
            connection.execute("UPDATE subscriber.billing_event SET status='PROCESSING',attempts=attempts+1 WHERE id=%s", (event_id,))
        event = event_row["payload"]
        subscription_id = self._subscription_id(event)
        if not subscription_id:
            self.db.execute(
                "UPDATE subscriber.billing_event SET status='PROCESSED',processed_at=now() WHERE id=%s",
                (event_id,),
            )
            return
        current = StripeBilling(self.settings).retrieve_subscription(subscription_id)
        metadata = current.get("metadata") or {}
        internal_customer = metadata.get("internal_customer_id")
        provider_customer = str(current.get("customer") or "")
        items = ((current.get("items") or {}).get("data") or [])
        price_id = str((((items[0] if items else {}).get("price") or {}).get("id")) or "")
        if not internal_customer or not provider_customer or not price_id:
            raise ValueError("provider subscription lacks bound customer or price")
        customer_id = uuid.UUID(str(internal_customer))
        provider_status = str(current.get("status") or "")
        normalized = {
            "active": "ACTIVE",
            "past_due": "PAST_DUE",
            "unpaid": "PAST_DUE",
            "canceled": "CANCELLED",
            "incomplete": "INCOMPLETE",
            "incomplete_expired": "EXPIRED",
            "paused": "PAST_DUE",
        }.get(provider_status, "INCOMPLETE")
        paid_through_epoch = current.get("current_period_end")
        paid_through = datetime.fromtimestamp(int(paid_through_epoch), timezone.utc) if paid_through_epoch else None
        cancel_at_end = bool(current.get("cancel_at_period_end"))
        if cancel_at_end and normalized == "ACTIVE":
            normalized = "CANCEL_AT_PERIOD_END"
        with self.db.transaction() as connection:
            product = connection.execute(
                "SELECT * FROM subscriber.product_version WHERE provider='stripe' AND provider_price_id=%s AND environment=%s",
                (price_id, self.settings.environment),
            ).fetchone()
            if not product:
                raise ValueError("provider price is not an allowlisted product")
            subscription = connection.execute(
                """
                INSERT INTO subscriber.subscription(id,customer_id,product_version_id,provider,provider_account_id,environment,
                  provider_customer_id,provider_subscription_id,state,provider_version,paid_through,cancel_at_period_end)
                VALUES (%s,%s,%s,'stripe',%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (provider,provider_account_id,environment,provider_subscription_id) DO UPDATE SET
                  state=excluded.state,provider_version=excluded.provider_version,paid_through=excluded.paid_through,
                  cancel_at_period_end=excluded.cancel_at_period_end,updated_at=now()
                RETURNING id
                """,
                (new_id(), customer_id, product["id"], self.settings.stripe_account_id, self.settings.environment,
                 provider_customer, subscription_id, normalized, str(current.get("created", "")), paid_through, cancel_at_end),
            ).fetchone()
            latest = connection.execute(
                "SELECT revision FROM subscriber.entitlement WHERE customer_id=%s AND product_version_id=%s ORDER BY revision DESC FOR UPDATE LIMIT 1",
                (customer_id, product["id"]),
            ).fetchone()
            revision = int(latest["revision"]) + 1 if latest else 1
            active = normalized in {"ACTIVE", "CANCEL_AT_PERIOD_END"} and paid_through and paid_through > datetime.now(timezone.utc)
            if active:
                connection.execute(
                    """
                    INSERT INTO subscriber.entitlement(id,customer_id,product_version_id,subscription_id,source_billing_event_id,
                      effective_at,expires_at,revision,reason) VALUES (%s,%s,%s,%s,%s,now(),%s,%s,'provider reconciliation')
                    """,
                    (new_id(), customer_id, product["id"], subscription["id"], event_id, paid_through, revision),
                )
            else:
                connection.execute(
                    "UPDATE subscriber.entitlement SET revoked_at=COALESCE(revoked_at,now()),reason=%s WHERE customer_id=%s AND product_version_id=%s AND revoked_at IS NULL",
                    (f"provider state {normalized}", customer_id, product["id"]),
                )
            connection.execute(
                "UPDATE subscriber.billing_event SET status='PROCESSED',processed_at=now(),sanitized_error=NULL WHERE id=%s",
                (event_id,),
            )

    def cancel_subscription(self, payload: dict[str, Any]) -> None:
        subscription_id = uuid.UUID(payload["subscription_id"])
        row = self.db.fetch_one(
            "SELECT provider_subscription_id FROM subscriber.subscription WHERE id=%s",
            (subscription_id,),
        )
        if not row:
            raise ValueError("subscription missing")
        key = f"cancel:{self.settings.environment}:{subscription_id}"
        result = StripeBilling(self.settings).cancel_at_period_end(row["provider_subscription_id"], key)
        if not result["cancel_at_period_end"]:
            raise ValueError("provider did not confirm cancellation")
        self.db.execute(
            "UPDATE subscriber.subscription SET cancel_at_period_end=true,state='CANCEL_AT_PERIOD_END',updated_at=now() WHERE id=%s",
            (subscription_id,),
        )

    def schedule_billing_reconciliation(self) -> int:
        bucket = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
        last = self.db.fetch_one("SELECT value FROM subscriber.service_state WHERE key='billing_reconciliation_last_scheduled'")
        if last and last["value"] == bucket:
            return 0
        rows = self.db.fetch_all(
            "SELECT provider_subscription_id FROM subscriber.subscription WHERE environment=%s AND state IN ('ACTIVE','PAST_DUE','CANCEL_AT_PERIOD_END','INCOMPLETE')",
            (self.settings.environment,),
        )
        created = 0
        for row in rows:
            provider_id = row["provider_subscription_id"]
            event = {
                "id": f"reconcile:{provider_id}:{bucket}",
                "type": "internal.subscription.reconciliation",
                "livemode": self.settings.billing_mode == "live",
                "data": {"object": {"object": "subscription", "id": provider_id}},
            }
            if self.repository.record_billing_event(
                event, object_hash(event), self.settings.stripe_account_id,
                self.settings.billing_mode, self.settings.billing_mode,
            ):
                created += 1
        self.db.execute(
            """
            INSERT INTO subscriber.service_state(key,value) VALUES ('billing_reconciliation_last_scheduled',%s)
            ON CONFLICT (key) DO UPDATE SET value=excluded.value,updated_at=now()
            """,
            (json.dumps(bucket),),
        )
        return created

    def expire_stale_releases(self) -> int:
        with self.db.transaction() as connection:
            rows = connection.execute(
                """
                SELECT rr.id FROM subscriber.release_revision rr
                JOIN subscriber.active_release ar ON ar.release_revision_id=rr.id
                WHERE rr.status='ACTIVE_REVISION_PROMOTED' AND ar.status='ACTIVE'
                  AND (rr.expires_at<=now() OR (rr.authority->>'expires_at')::timestamptz<=now()
                       OR COALESCE((rr.authority->>'revoked')::boolean,false)=true)
                FOR UPDATE OF rr,ar
                """
            ).fetchall()
            for row in rows:
                connection.execute("UPDATE subscriber.release_revision SET status='INVALIDATED' WHERE id=%s", (row["id"],))
                connection.execute("UPDATE subscriber.active_release SET status='SUSPENDED',updated_at=now() WHERE release_revision_id=%s", (row["id"],))
                connection.execute(
                    "UPDATE subscriber.notification_outbox SET status='SUPPRESSED',suppression_reason='RELEASE_OR_AUTHORITY_EXPIRED' WHERE release_revision_id=%s AND status IN ('PENDING','FAILED')",
                    (row["id"],),
                )
                connection.execute(
                    "INSERT INTO subscriber.release_event(id,release_revision_id,event_type,reason,request_id) VALUES (%s,%s,'INVALIDATED','release or authority expired',%s)",
                    (new_id(), row["id"], str(uuid.uuid4())),
                )
            return len(rows)

    def enforce_market_gate(self) -> bool:
        state = self.repository.service_state()
        if set(state["qualified_markets"]) & set(state["commercially_enabled_markets"]):
            return False
        if state["sales_status"] != "OWNER_ENABLED" and state["release_status"] != "OWNER_ENABLED":
            return False
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO subscriber.service_state(key,value) VALUES ('sales_status','\"DISABLED\"'::jsonb) ON CONFLICT (key) DO UPDATE SET value=excluded.value,updated_at=now()"
            )
            connection.execute(
                "INSERT INTO subscriber.service_state(key,value) VALUES ('release_status','\"PAUSED\"'::jsonb) ON CONFLICT (key) DO UPDATE SET value=excluded.value,updated_at=now()"
            )
            connection.execute(
                "INSERT INTO subscriber.admin_audit(id,action,scoped_target,before_reference,after_reference,reason,request_id) VALUES (%s,'AUTOMATIC_MARKET_GATE','paid-launch',%s,'sales disabled; releases paused','no commercially enabled qualified market',%s)",
                (new_id(), json.dumps({"sales": state["sales_status"], "releases": state["release_status"]}), str(uuid.uuid4())),
            )
        return True

    def dispatch_notification(self) -> bool:
        if self.repository.service_state()["alerts_status"] != "OWNER_ENABLED":
            count = self.db.execute(
                "UPDATE subscriber.notification_outbox SET status='SUPPRESSED',suppression_reason='ALERTS_PAUSED' WHERE status IN ('PENDING','FAILED')"
            )
            return bool(count)
        with self.db.transaction() as connection:
            row = connection.execute(
                """
                SELECT n.*,c.contact_email,rr.release_id,rr.expires_at,rr.status AS release_status,e.expires_at AS entitlement_expires
                FROM subscriber.notification_outbox n
                JOIN subscriber.customer c ON c.id=n.recipient_customer_id
                JOIN subscriber.release_revision rr ON rr.id=n.release_revision_id
                LEFT JOIN LATERAL (
                  SELECT expires_at FROM subscriber.entitlement e
                  WHERE e.customer_id=n.recipient_customer_id AND e.revoked_at IS NULL
                    AND e.effective_at<=now() AND e.expires_at>now() ORDER BY revision DESC LIMIT 1
                ) e ON true
                WHERE n.status IN ('PENDING','FAILED') AND COALESCE(n.next_attempt_at,now())<=now()
                  AND (n.lease_expires_at IS NULL OR n.lease_expires_at<now())
                  AND c.alerts_enabled=true AND c.email_suppressed_at IS NULL
                ORDER BY n.created_at FOR UPDATE SKIP LOCKED LIMIT 1
                """
            ).fetchone()
            if not row:
                return False
            if not row["entitlement_expires"] or row["release_status"] != "ACTIVE_REVISION_PROMOTED" or row["expires_at"] <= datetime.now(timezone.utc):
                connection.execute(
                    "UPDATE subscriber.notification_outbox SET status='SUPPRESSED',suppression_reason='ENTITLEMENT_OR_RELEASE_NOT_CURRENT' WHERE id=%s",
                    (row["id"],),
                )
                return True
            connection.execute(
                "UPDATE subscriber.notification_outbox SET status='SENDING',attempts=attempts+1,lease_owner=%s,lease_expires_at=now()+interval '2 minutes' WHERE id=%s",
                (self.worker_id, row["id"]),
            )
        if not self.settings.smtp_url or not self.settings.email_from:
            raise ValueError("SMTP delivery is not configured")
        parsed = urlsplit(self.settings.smtp_url)
        if parsed.scheme not in {"smtp", "smtps"} or not parsed.hostname:
            raise ValueError("SMTP URL is invalid")
        message = email.message.EmailMessage()
        message["From"] = self.settings.email_from
        message["To"] = row["contact_email"]
        message["Subject"] = "A new ParlayPicker release is available"
        message.set_content(f"A new release is available. Sign in to view it: {self.settings.public_base_url}/picks\n")
        smtp_cls = smtplib.SMTP_SSL if parsed.scheme == "smtps" else smtplib.SMTP
        try:
            with smtp_cls(parsed.hostname, parsed.port or (465 if parsed.scheme == "smtps" else 25), timeout=10) as smtp:
                if parsed.username:
                    smtp.login(parsed.username, parsed.password or "")
                provider_id = smtp.send_message(message)
        except Exception:
            self.db.execute(
                "UPDATE subscriber.notification_outbox SET status='UNKNOWN_OUTCOME',sanitized_error='provider submission outcome unknown',lease_owner=NULL,lease_expires_at=NULL WHERE id=%s",
                (row["id"],),
            )
            raise
        self.db.execute(
            "UPDATE subscriber.notification_outbox SET status='ACCEPTED_BY_PROVIDER',provider_message_id=%s,lease_owner=NULL,lease_expires_at=NULL WHERE id=%s",
            (json.dumps(provider_id, sort_keys=True), row["id"]),
        )
        return True

    def run_once(self) -> bool:
        if self.enforce_market_gate():
            return True
        if self.expire_stale_releases():
            return True
        job = self.lease_job()
        if job:
            try:
                if job["job_type"] == "VERIFY_RELEASE_ROUTE":
                    self.verify_release(job["payload"])
                elif job["job_type"] == "PROCESS_BILLING_EVENT":
                    self.process_billing_event(job["payload"])
                elif job["job_type"] == "CANCEL_SUBSCRIPTION":
                    self.cancel_subscription(job["payload"])
                else:
                    raise ValueError("unknown job type")
                self.complete(job["id"])
            except Exception as exc:
                self.fail(job, exc)
            return True
        if self.schedule_billing_reconciliation():
            return True
        return self.dispatch_notification()

    def run(self) -> None:
        while True:
            try:
                worked = self.run_once()
            except Exception:
                worked = False
            time.sleep(0.25 if worked else 2.0)


def main() -> None:
    Worker(Settings.from_env()).run()


if __name__ == "__main__":
    main()
