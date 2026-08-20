-- Standalone PostgreSQL/Supabase schema. The trusted API role also performs
-- ownership checks and sets app.current_user_id for each user transaction.
create extension if not exists pgcrypto;
do $$ begin create type analysisstatus as enum ('QUEUED','ANALYZING','PREDICTING','COMPLETED','FAILED'); exception when duplicate_object then null; end $$;
do $$ begin create type risklevel as enum ('LOW','MEDIUM','HIGH','CRITICAL'); exception when duplicate_object then null; end $$;

create table if not exists users (id varchar(36) primary key, email varchar(320), github_username varchar(255) not null unique, github_id varchar(64) not null unique, encrypted_github_token text, created_at timestamptz not null);
create table if not exists repositories (id varchar(36) primary key, user_id varchar(36) not null references users(id) on delete cascade, github_repo_id varchar(64) not null, name varchar(255) not null, owner varchar(255) not null, url text not null, default_branch varchar(255) not null default 'main', is_private boolean not null default false, created_at timestamptz not null, unique(user_id, github_repo_id));
create table if not exists analyses (id varchar(36) primary key, repository_id varchar(36) not null references repositories(id) on delete cascade, commit_sha varchar(64), status analysisstatus not null default 'QUEUED', change_risk_probability double precision, overall_priority_score double precision, risk_level risklevel, model_version varchar(100), error_message text, created_at timestamptz not null, completed_at timestamptz);
create table if not exists files (id varchar(36) primary key, analysis_id varchar(36) not null references analyses(id) on delete cascade, file_path text not null, file_priority_score double precision not null, risk_level risklevel not null, loc integer not null default 0, complexity double precision not null default 0, code_churn integer not null default 0, commit_count integer not null default 0, contributor_count integer not null default 0, file_age_days integer not null default 0, lines_added integer not null default 0, lines_deleted integer not null default 0, dependency_count integer not null default 0, explanations jsonb not null default '[]'::jsonb, recommendations jsonb not null default '[]'::jsonb);
create table if not exists predictions (id varchar(36) primary key, file_id varchar(36) not null unique references files(id) on delete cascade, model_version varchar(100) not null, change_risk_probability double precision not null, created_at timestamptz not null);
create table if not exists explanations (id varchar(36) primary key, prediction_id varchar(36) not null references predictions(id) on delete cascade, feature_name varchar(100) not null, feature_value double precision not null, shap_value double precision not null);
create table if not exists recommendations (id varchar(36) primary key, prediction_id varchar(36) not null references predictions(id) on delete cascade, text text not null, source varchar(32) not null default 'deterministic');

create index if not exists ix_repositories_user_id on repositories(user_id);
create index if not exists ix_analyses_repository_id on analyses(repository_id);
create index if not exists ix_analyses_status on analyses(status);
create index if not exists ix_files_analysis_id on files(analysis_id);
create index if not exists ix_predictions_file_id on predictions(file_id);
create index if not exists ix_explanations_prediction_id on explanations(prediction_id);
create index if not exists ix_recommendations_prediction_id on recommendations(prediction_id);

alter table users enable row level security;
alter table repositories enable row level security;
alter table analyses enable row level security;
alter table files enable row level security;
alter table predictions enable row level security;
alter table explanations enable row level security;
alter table recommendations enable row level security;

drop policy if exists users_self on users;
create policy users_self on users for all using (id = current_setting('app.current_user_id', true)) with check (id = current_setting('app.current_user_id', true));
drop policy if exists repositories_owner on repositories;
create policy repositories_owner on repositories for all using (user_id = current_setting('app.current_user_id', true)) with check (user_id = current_setting('app.current_user_id', true));
drop policy if exists analyses_owner on analyses;
create policy analyses_owner on analyses for all using (exists (select 1 from repositories r where r.id=analyses.repository_id and r.user_id=current_setting('app.current_user_id',true))) with check (exists (select 1 from repositories r where r.id=analyses.repository_id and r.user_id=current_setting('app.current_user_id',true)));
drop policy if exists files_owner on files;
create policy files_owner on files for all using (exists (select 1 from analyses a join repositories r on r.id=a.repository_id where a.id=files.analysis_id and r.user_id=current_setting('app.current_user_id',true))) with check (exists (select 1 from analyses a join repositories r on r.id=a.repository_id where a.id=files.analysis_id and r.user_id=current_setting('app.current_user_id',true)));
drop policy if exists predictions_owner on predictions;
create policy predictions_owner on predictions for all using (exists (select 1 from files f join analyses a on a.id=f.analysis_id join repositories r on r.id=a.repository_id where f.id=predictions.file_id and r.user_id=current_setting('app.current_user_id',true))) with check (exists (select 1 from files f join analyses a on a.id=f.analysis_id join repositories r on r.id=a.repository_id where f.id=predictions.file_id and r.user_id=current_setting('app.current_user_id',true)));
drop policy if exists explanations_owner on explanations;
create policy explanations_owner on explanations for all using (exists (select 1 from predictions p join files f on f.id=p.file_id join analyses a on a.id=f.analysis_id join repositories r on r.id=a.repository_id where p.id=explanations.prediction_id and r.user_id=current_setting('app.current_user_id',true))) with check (exists (select 1 from predictions p join files f on f.id=p.file_id join analyses a on a.id=f.analysis_id join repositories r on r.id=a.repository_id where p.id=explanations.prediction_id and r.user_id=current_setting('app.current_user_id',true)));
drop policy if exists recommendations_owner on recommendations;
create policy recommendations_owner on recommendations for all using (exists (select 1 from predictions p join files f on f.id=p.file_id join analyses a on a.id=f.analysis_id join repositories r on r.id=a.repository_id where p.id=recommendations.prediction_id and r.user_id=current_setting('app.current_user_id',true))) with check (exists (select 1 from predictions p join files f on f.id=p.file_id join analyses a on a.id=f.analysis_id join repositories r on r.id=a.repository_id where p.id=recommendations.prediction_id and r.user_id=current_setting('app.current_user_id',true)));
