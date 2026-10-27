# Private NFL storage diagnostic successor

This engineering scope adds a default-off, owner-token-protected diagnostic to
the existing Streamlit owner panel. It uses the existing settings resolver and
the released `production_service` and `Store.verify` path. It does not authorize
deployment, a database connection, acquisition, inference or wagering.

One explicit owner action starts at most one released connection call in that
session. The action is recorded before the attempt and rerenders do not retry.
A new session or process needs a new explicit owner action and separate actual
authority; this session guard is not a durable cross-restart allowance ledger.
The diagnostic does not initialize or modify the database, and requires both
NFL acquisition and inference settings to be the literal string `false`.

The owner must privately enter the independently verified production direct hostname; only the match boolean is returned. Settings checks also require the restricted role, intended database and port, system trust roots, and no unexpected connection parameters.

The same read-only transaction additionally checks the connected identity, read-only mode and required positive and negative privileges; an inactive TLS connection stops before the transaction.

Instrumentation permits at most one driver connection-generator invocation, blocking a second resolved-address attempt before it begins; unsupported driver instrumentation fails closed.

The released `verify-full` setting, hostname verification, five-second connect
deadline, 5,000 ms statement and lock limits, read-only transaction, role and
schema checks remain unchanged. The diagnostic observes the original failure
before it is translated, and emits only a sanitized phase, exception class,
SQLSTATE/category and nonsecret runtime data. It does not return exception text,
credentials, URLs, hostnames, key material or secret-derived hashes. The
diagnostic suppresses psycopg log records only from its own thread while active.

The guard blocks a second psycopg connection-generator attempt. It does not
prove libpq opens only one socket internally, and its session-only action guard
does not enforce a cross-session allowance. A stricter allowance must be reviewed
before any hosted invocation.
The result is diagnostic evidence, not a prediction or a validation of stored
forecast accuracy. A new hosted connection requires a later exact review.
