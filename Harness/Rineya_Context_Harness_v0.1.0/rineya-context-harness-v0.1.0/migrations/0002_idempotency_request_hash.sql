ALTER TABLE idempotency_records
ADD COLUMN request_hash TEXT NOT NULL DEFAULT 'legacy-unverifiable';
