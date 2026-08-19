PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  goal TEXT NOT NULL,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  revision INTEGER NOT NULL,
  current_node TEXT NOT NULL,
  config_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_items (
  run_id TEXT NOT NULL,
  id TEXT NOT NULL,
  record_json TEXT NOT NULL,
  status TEXT NOT NULL,
  pinned INTEGER NOT NULL,
  revision_updated INTEGER NOT NULL,
  PRIMARY KEY (run_id, id),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memory_run_status ON memory_items(run_id, status);

CREATE TABLE IF NOT EXISTS evidence_records (
  run_id TEXT NOT NULL,
  id TEXT NOT NULL,
  record_json TEXT NOT NULL,
  PRIMARY KEY (run_id, id),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_records (
  run_id TEXT NOT NULL,
  id TEXT NOT NULL,
  record_json TEXT NOT NULL,
  PRIMARY KEY (run_id, id),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS state_events (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  trace_id TEXT,
  span_id TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_state_events_run_revision ON state_events(run_id, revision, created_at);

CREATE TABLE IF NOT EXISTS checkpoints (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  source_revision INTEGER NOT NULL,
  label TEXT NOT NULL,
  state_hash TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idempotency_records (
  run_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  patch_id TEXT NOT NULL,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, idempotency_key),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
