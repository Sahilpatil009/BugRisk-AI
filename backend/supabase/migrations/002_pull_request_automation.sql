create table if not exists pull_requests (
  id varchar(36) primary key,
  repository_id varchar(36) not null references repositories(id) on delete cascade,
  analysis_id varchar(36) references analyses(id) on delete set null,
  github_pr_number integer not null,
  installation_id varchar(64) not null,
  title varchar(500) not null,
  author varchar(255) not null,
  html_url text not null,
  base_sha varchar(64) not null,
  head_sha varchar(64) not null,
  state varchar(32) not null default 'open',
  status analysisstatus not null default 'QUEUED',
  risk_score double precision,
  risk_level risklevel,
  changed_files jsonb not null default '[]'::jsonb,
  github_comment_id varchar(64),
  last_delivery_id varchar(100),
  created_at timestamptz not null,
  updated_at timestamptz not null,
  unique(repository_id, github_pr_number)
);

create index if not exists ix_pull_requests_repository_id on pull_requests(repository_id);
create index if not exists ix_pull_requests_analysis_id on pull_requests(analysis_id);
create index if not exists ix_pull_requests_status on pull_requests(status);
create index if not exists ix_pull_requests_last_delivery_id on pull_requests(last_delivery_id);

alter table pull_requests enable row level security;
drop policy if exists pull_requests_owner on pull_requests;
create policy pull_requests_owner on pull_requests for all
using (
  exists (
    select 1 from repositories r
    where r.id=pull_requests.repository_id
      and r.user_id=current_setting('app.current_user_id', true)
  )
)
with check (
  exists (
    select 1 from repositories r
    where r.id=pull_requests.repository_id
      and r.user_id=current_setting('app.current_user_id', true)
  )
);
