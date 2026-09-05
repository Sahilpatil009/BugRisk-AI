from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from .models import AnalysisStatus, RiskLevel


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class UserOut(ApiModel):
    id: str
    github_username: str
    email: str | None


class RepositoryConnect(BaseModel):
    github_repo_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    owner: str = Field(min_length=1, max_length=255)
    url: HttpUrl
    default_branch: str = Field(default="main", min_length=1, max_length=255)
    is_private: bool = False

    @field_validator("url")
    @classmethod
    def github_only(cls, value: HttpUrl) -> HttpUrl:
        if value.host not in {"github.com", "www.github.com"}:
            raise ValueError("Only github.com repositories are supported")
        return value


class RepositoryOut(ApiModel):
    id: str
    github_repo_id: str
    name: str
    owner: str
    url: str
    default_branch: str
    is_private: bool
    created_at: datetime


class AnalysisCreate(BaseModel):
    repository_id: str
    commit_sha: str | None = Field(default=None, max_length=64, pattern=r"^[0-9a-fA-F]*$")


class AnalysisOut(ApiModel):
    id: str
    repository_id: str
    commit_sha: str | None
    status: AnalysisStatus
    change_risk_probability: float | None
    overall_priority_score: float | None
    risk_level: RiskLevel | None
    model_version: str | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None


class Explanation(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float


class FileResultOut(ApiModel):
    id: str
    analysis_id: str
    file_path: str
    file_priority_score: float
    risk_level: RiskLevel
    loc: int
    complexity: float
    code_churn: int
    commit_count: int
    contributor_count: int
    file_age_days: int
    lines_added: int
    lines_deleted: int
    dependency_count: int
    explanations: list[Explanation]
    recommendations: list[str]


class PaginatedFiles(BaseModel):
    items: list[FileResultOut]
    total: int
    page: int
    page_size: int


class PredictionInput(BaseModel):
    lines_added: float = Field(ge=0)
    lines_deleted: float = Field(ge=0)
    files_changed: float = Field(ge=1)
    code_churn: float = Field(ge=0)
    commit_entropy: float = Field(ge=0)
    developer_experience: float = Field(ge=0)
    file_age_days: float = Field(ge=0)
    commit_frequency: float = Field(ge=0)
    contributor_count: float = Field(ge=0)
    complexity: float = Field(ge=0)
    previous_defects: float = Field(ge=0)


class PredictionOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    change_risk_probability: float
    risk_level: RiskLevel
    model_version: str
    explanations: list[Explanation]


class ModelMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_name: str
    trained: bool
    metrics: dict[str, float]
    feature_names: list[str]
    note: str | None = None


class PullRequestOut(ApiModel):
    id: str
    repository_id: str
    analysis_id: str | None
    github_pr_number: int
    title: str
    author: str
    html_url: str
    base_sha: str
    head_sha: str
    state: str
    status: AnalysisStatus
    risk_score: float | None
    risk_level: RiskLevel | None
    changed_files: list[str]
    created_at: datetime
    updated_at: datetime
