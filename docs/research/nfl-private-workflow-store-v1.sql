-- PREPARATION ONLY. Run by a database administrator after owner approval.
-- Dedicated database, encrypted durable storage/backups, private network/TLS.
-- Replace nfl_app with the dedicated NON-OWNER login; do not grant membership
-- in the schema/table owner role. Never use the subscriber/billing schema.
BEGIN;
CREATE SCHEMA nfl_private;
REVOKE ALL ON SCHEMA nfl_private FROM PUBLIC;
CREATE TABLE nfl_private.meta (version TEXT PRIMARY KEY);
INSERT INTO nfl_private.meta VALUES ('nfl-private-workflow-store-v1');
CREATE TABLE nfl_private.records (
  owner_id TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL,
  raw TEXT NOT NULL CHECK (octet_length(raw) <= 33554432),
  signature TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(owner_id,id)
);
CREATE TABLE nfl_private.claims (LIKE nfl_private.records INCLUDING ALL);
CREATE FUNCTION nfl_private.immutable() RETURNS trigger LANGUAGE plpgsql AS
$$BEGIN RAISE EXCEPTION 'NFL private records are append-only'; END;$$;
CREATE TRIGGER records_immutable BEFORE UPDATE OR DELETE ON nfl_private.records
  FOR EACH ROW EXECUTE FUNCTION nfl_private.immutable();
CREATE TRIGGER claims_immutable BEFORE UPDATE OR DELETE ON nfl_private.claims
  FOR EACH ROW EXECUTE FUNCTION nfl_private.immutable();
REVOKE ALL ON ALL TABLES IN SCHEMA nfl_private FROM PUBLIC;
REVOKE ALL ON FUNCTION nfl_private.immutable() FROM PUBLIC;
GRANT USAGE ON SCHEMA nfl_private TO nfl_app;
GRANT SELECT ON nfl_private.meta TO nfl_app;
GRANT SELECT, INSERT ON nfl_private.records, nfl_private.claims TO nfl_app;
COMMIT;
