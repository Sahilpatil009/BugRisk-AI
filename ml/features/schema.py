FEATURE_NAMES = [
    "lines_added",
    "lines_deleted",
    "files_changed",
    "code_churn",
    "commit_entropy",
    "developer_experience",
    "file_age_days",
    "commit_frequency",
    "contributor_count",
    "complexity",
    "previous_defects",
]

FEATURE_ALIASES = {
    "lines_added": ("lines_added", "la"),
    "lines_deleted": ("lines_deleted", "ld"),
    "files_changed": ("files_changed", "nf"),
    "commit_entropy": ("commit_entropy", "entropy", "ent"),
    "developer_experience": ("developer_experience", "exp"),
    "file_age_days": ("file_age_days", "age"),
    "commit_frequency": ("commit_frequency", "nuc"),
    "contributor_count": ("contributor_count", "ndev"),
    "complexity": ("complexity",),
    "previous_defects": ("previous_defects",),
}

TARGET_ALIASES = ("buggy", "is_buggy", "target")
TIME_ALIASES = ("author_date", "commit_date", "date", "timestamp")
PROJECT_ALIASES = ("project", "project_name", "repository", "repo")
