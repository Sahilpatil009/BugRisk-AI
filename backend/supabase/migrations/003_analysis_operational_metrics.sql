alter table analyses add column if not exists started_at timestamptz;
alter table analyses add column if not exists attempt_count integer not null default 0;

create index if not exists ix_analyses_status_started_at on analyses(status, started_at);
