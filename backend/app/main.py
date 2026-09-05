import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from .auth import begin_oauth, current_user, decrypt_token, finish_oauth
from .config import get_settings
from .database import Base, SessionLocal, engine, get_db
from .github_app import installation_token, pull_request_files, verify_webhook_signature
from .models import Analysis, FileResult, PullRequest, Repository, User, utcnow
from .risk import ModelService, risk_level
from .schemas import (
    AnalysisCreate,
    AnalysisOut,
    Explanation,
    FileResultOut,
    ModelMetrics,
    PaginatedFiles,
    PredictionInput,
    PredictionOut,
    PullRequestOut,
    RepositoryConnect,
    RepositoryOut,
    UserOut,
)
from .seed import seed_demo
from .services import process_analysis

settings = get_settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if (
        not settings.demo_mode
        and settings.session_secret == "local-demo-secret-change-in-production"
    ):
        raise RuntimeError("SESSION_SECRET must be changed outside demo mode")
    if not settings.demo_mode and not settings.model_path.exists():
        raise RuntimeError("A trained MODEL_PATH is required outside demo mode")
    if not settings.demo_mode and not settings.token_encryption_key:
        raise RuntimeError("TOKEN_ENCRYPTION_KEY is required outside demo mode")
    Base.metadata.create_all(engine)
    if settings.demo_mode:
        with SessionLocal() as db:
            seed_demo(db)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Commit change-risk prediction and evidence-based file prioritization.",
    lifespan=lifespan,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    same_site=settings.cookie_same_site,
    https_only=settings.secure_cookies,
    max_age=60 * 60 * 12,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "message": str(exc.errors())}},
    )


Db = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(current_user)]


@app.get("/health")
def health(db: Db) -> dict[str, str]:
    db.execute(select(1))
    return {"status": "healthy", "service": "bugrisk-api"}


@app.get("/auth/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user


@app.get("/auth/github")
def github_login(request: Request) -> RedirectResponse:
    return RedirectResponse(begin_oauth(request, settings))


@app.get("/auth/github/callback")
async def github_callback(code: str, state: str, request: Request, db: Db) -> RedirectResponse:
    await finish_oauth(code, state, request, db, settings)
    return RedirectResponse(f"{settings.frontend_url}/dashboard")


@app.post("/auth/logout", status_code=204)
def logout(request: Request) -> None:
    request.session.clear()


@app.get("/repositories", response_model=list[RepositoryOut])
def repositories(db: Db, user: CurrentUser) -> list[Repository]:
    return list(
        db.scalars(
            select(Repository)
            .where(Repository.user_id == user.id)
            .order_by(desc(Repository.created_at))
        )
    )


@app.get("/repositories/github")
async def github_repositories(db: Db, user: CurrentUser) -> list[dict]:
    token = decrypt_token(user.encrypted_github_token, settings)
    if not token:
        if settings.demo_mode:
            return []
        raise HTTPException(401, "GitHub connection required")
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            "https://api.github.com/user/repos?sort=updated&per_page=100",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        response.raise_for_status()
    return [
        {
            "github_repo_id": str(item["id"]),
            "name": item["name"],
            "owner": item["owner"]["login"],
            "url": item["clone_url"],
            "default_branch": item["default_branch"],
            "is_private": item["private"],
        }
        for item in response.json()
    ]


@app.post("/repositories/connect", response_model=RepositoryOut, status_code=201)
def connect_repository(payload: RepositoryConnect, db: Db, user: CurrentUser) -> Repository:
    existing = db.scalar(
        select(Repository).where(
            Repository.user_id == user.id, Repository.github_repo_id == payload.github_repo_id
        )
    )
    if existing:
        return existing
    repository = Repository(user_id=user.id, **payload.model_dump(mode="json"))
    db.add(repository)
    db.commit()
    db.refresh(repository)
    return repository


@app.post("/webhooks/github", status_code=202)
async def github_webhook(request: Request, background: BackgroundTasks, db: Db) -> dict[str, str]:
    body = await request.body()
    if not verify_webhook_signature(
        body, request.headers.get("X-Hub-Signature-256"), settings.github_webhook_secret
    ):
        raise HTTPException(401, "Invalid GitHub webhook signature")
    event = request.headers.get("X-GitHub-Event")
    if event == "ping":
        return {"status": "accepted", "event": "ping"}
    payload = await request.json()
    action = payload.get("action")
    if event != "pull_request" or action not in {"opened", "synchronize"}:
        return {"status": "ignored", "event": event or "unknown"}
    try:
        github_repository = payload["repository"]
        github_pull_request = payload["pull_request"]
        installation_id = str(payload["installation"]["id"])
        repository_id = str(github_repository["id"])
        number = int(payload["number"])
        head_sha = str(github_pull_request["head"]["sha"])
        base_sha = str(github_pull_request["base"]["sha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "Malformed pull request webhook payload") from exc
    repository = db.scalar(
        select(Repository)
        .where(Repository.github_repo_id == repository_id)
        .order_by(desc(Repository.created_at))
    )
    if not repository:
        return {"status": "ignored", "event": "repository_not_connected"}
    delivery_id = request.headers.get("X-GitHub-Delivery")
    pull_request = db.scalar(
        select(PullRequest).where(
            PullRequest.repository_id == repository.id,
            PullRequest.github_pr_number == number,
        )
    )
    if pull_request and delivery_id and pull_request.last_delivery_id == delivery_id:
        return {"status": "duplicate", "event": action}
    token = await installation_token(settings, installation_id)
    changed_files = await pull_request_files(repository.owner, repository.name, number, token)
    if not pull_request:
        pull_request = PullRequest(
            repository_id=repository.id,
            github_pr_number=number,
            installation_id=installation_id,
            title=str(github_pull_request.get("title") or f"Pull request #{number}"),
            author=str(github_pull_request.get("user", {}).get("login") or "unknown"),
            html_url=str(github_pull_request["html_url"]),
            base_sha=base_sha,
            head_sha=head_sha,
        )
        db.add(pull_request)
        db.flush()
    analysis = Analysis(repository_id=repository.id, commit_sha=head_sha)
    db.add(analysis)
    db.flush()
    pull_request.analysis_id = analysis.id
    pull_request.installation_id = installation_id
    pull_request.title = str(github_pull_request.get("title") or pull_request.title)
    pull_request.author = str(github_pull_request.get("user", {}).get("login") or "unknown")
    pull_request.html_url = str(github_pull_request["html_url"])
    pull_request.base_sha = base_sha
    pull_request.head_sha = head_sha
    pull_request.state = str(github_pull_request.get("state") or "open")
    pull_request.status = analysis.status
    pull_request.risk_score = None
    pull_request.risk_level = None
    pull_request.changed_files = changed_files
    pull_request.last_delivery_id = delivery_id
    pull_request.updated_at = utcnow()
    db.commit()
    if settings.inline_worker:
        background.add_task(process_analysis, analysis.id)
    return {"status": "accepted", "event": action}


@app.get("/repositories/{repository_id}/pull-requests", response_model=list[PullRequestOut])
def repository_pull_requests(repository_id: str, db: Db, user: CurrentUser) -> list[PullRequest]:
    repository = db.scalar(
        select(Repository).where(Repository.id == repository_id, Repository.user_id == user.id)
    )
    if not repository:
        raise HTTPException(404, "Repository not found")
    return list(
        db.scalars(
            select(PullRequest)
            .where(PullRequest.repository_id == repository.id)
            .order_by(desc(PullRequest.updated_at))
        )
    )


@app.get("/analyses", response_model=list[AnalysisOut])
def analyses(db: Db, user: CurrentUser, repository_id: str | None = None) -> list[Analysis]:
    statement = select(Analysis).join(Repository).where(Repository.user_id == user.id)
    if repository_id:
        statement = statement.where(Analysis.repository_id == repository_id)
    return list(db.scalars(statement.order_by(desc(Analysis.created_at))))


@app.post("/analyses", response_model=AnalysisOut, status_code=202)
def create_analysis(
    payload: AnalysisCreate, background: BackgroundTasks, db: Db, user: CurrentUser
) -> Analysis:
    repository = db.scalar(
        select(Repository).where(
            Repository.id == payload.repository_id, Repository.user_id == user.id
        )
    )
    if not repository:
        raise HTTPException(404, "Repository not found")
    analysis = Analysis(repository_id=repository.id, commit_sha=payload.commit_sha)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    if settings.inline_worker:
        background.add_task(process_analysis, analysis.id)
    return analysis


def owned_analysis(analysis_id: str, db: Session, user: User) -> Analysis:
    analysis = db.scalar(
        select(Analysis)
        .join(Repository)
        .where(Analysis.id == analysis_id, Repository.user_id == user.id)
    )
    if not analysis:
        raise HTTPException(404, "Analysis not found")
    return analysis


@app.get("/analyses/{analysis_id}", response_model=AnalysisOut)
def analysis_detail(analysis_id: str, db: Db, user: CurrentUser) -> Analysis:
    return owned_analysis(analysis_id, db, user)


SORT_COLUMNS = {
    "priority": FileResult.file_priority_score,
    "complexity": FileResult.complexity,
    "churn": FileResult.code_churn,
    "commits": FileResult.commit_count,
    "name": FileResult.file_path,
}


@app.get("/analyses/{analysis_id}/files", response_model=PaginatedFiles)
def analysis_files(
    analysis_id: str,
    db: Db,
    user: CurrentUser,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    sort_by: Literal["priority", "complexity", "churn", "commits", "name"] = "priority",
    order: Literal["asc", "desc"] = "desc",
) -> PaginatedFiles:
    owned_analysis(analysis_id, db, user)
    column = SORT_COLUMNS[sort_by]
    ordering = asc(column) if order == "asc" else desc(column)
    total = (
        db.scalar(
            select(func.count())
            .select_from(FileResult)
            .where(FileResult.analysis_id == analysis_id)
        )
        or 0
    )
    items = list(
        db.scalars(
            select(FileResult)
            .where(FileResult.analysis_id == analysis_id)
            .order_by(ordering)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return PaginatedFiles(
        items=[FileResultOut.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get("/analyses/{analysis_id}/files/{file_id}", response_model=FileResultOut)
def file_detail(analysis_id: str, file_id: str, db: Db, user: CurrentUser) -> FileResult:
    owned_analysis(analysis_id, db, user)
    item = db.scalar(
        select(FileResult).where(FileResult.id == file_id, FileResult.analysis_id == analysis_id)
    )
    if not item:
        raise HTTPException(404, "File result not found")
    return item


@app.get("/models/current/metrics", response_model=ModelMetrics)
def model_metrics(_: CurrentUser) -> dict:
    return ModelService(settings.model_path).metrics()


@app.post("/predict", response_model=PredictionOut)
def predict(payload: PredictionInput, _: CurrentUser) -> PredictionOut:
    prediction = ModelService(settings.model_path).predict(payload.model_dump())
    return PredictionOut(
        change_risk_probability=prediction.probability,
        risk_level=risk_level(prediction.probability),
        model_version=prediction.model_version,
        explanations=[Explanation(**item) for item in prediction.explanations],
    )
