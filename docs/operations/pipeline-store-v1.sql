-- PREPARATION ONLY. A database administrator runs this after owner approval.
-- Use a dedicated database with encrypted durable storage and backups, reached
-- over private network/TLS. Replace pipeline_app with the dedicated NON-OWNER
-- login, and do not grant it membership in the schema/table owner role. Never
-- use the subscriber/billing schema. The existing nfl_private schema is separate
-- and stays unchanged.
BEGIN;
CREATE SCHEMA pipeline_private;
REVOKE ALL ON SCHEMA pipeline_private FROM PUBLIC;
CREATE TABLE pipeline_private.meta (version TEXT PRIMARY KEY);
INSERT INTO pipeline_private.meta VALUES ('pipeline-store-v1');
CREATE TABLE pipeline_private.records (
  owner_id TEXT NOT NULL, sport TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL,
  raw TEXT NOT NULL CHECK (octet_length(raw) <= 33554432),
  signature TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(owner_id,sport,id)
);
CREATE TABLE pipeline_private.claims (LIKE pipeline_private.records INCLUDING ALL);
CREATE FUNCTION pipeline_private.immutable() RETURNS trigger LANGUAGE plpgsql AS
$$BEGIN RAISE EXCEPTION 'pipeline records are append-only'; END;$$;
CREATE TRIGGER records_immutable BEFORE UPDATE OR DELETE ON pipeline_private.records
  FOR EACH ROW EXECUTE FUNCTION pipeline_private.immutable();
CREATE TRIGGER claims_immutable BEFORE UPDATE OR DELETE ON pipeline_private.claims
  FOR EACH ROW EXECUTE FUNCTION pipeline_private.immutable();
REVOKE ALL ON ALL TABLES IN SCHEMA pipeline_private FROM PUBLIC;
REVOKE ALL ON FUNCTION pipeline_private.immutable() FROM PUBLIC;
GRANT USAGE ON SCHEMA pipeline_private TO pipeline_app;
GRANT SELECT ON pipeline_private.meta TO pipeline_app;
GRANT SELECT, INSERT ON pipeline_private.records, pipeline_private.claims TO pipeline_app;
COMMIT;
