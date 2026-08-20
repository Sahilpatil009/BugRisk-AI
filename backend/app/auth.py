import base64
import hashlib
import secrets
from typing import Annotated
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .database import get_db
from .models import User


def _fernet(settings: Settings) -> Fernet:
    if settings.token_encryption_key:
        return Fernet(settings.token_encryption_key.encode())
    key = base64.urlsafe_b64encode(hashlib.sha256(settings.session_secret.encode()).digest())
    return Fernet(key)


def encrypt_token(token: str, settings: Settings) -> str:
    return _fernet(settings).encrypt(token.encode()).decode()


def decrypt_token(token: str | None, settings: Settings) -> str | None:
    return _fernet(settings).decrypt(token.encode()).decode() if token else None


def ensure_demo_user(db: Session) -> User:
    user = db.scalar(select(User).where(User.github_id == "demo-user"))
    if user:
        return user
    user = User(github_id="demo-user", github_username="demo-developer", email="demo@bugrisk.local")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def current_user(request: Request, db: Annotated[Session, Depends(get_db)]) -> User:
    settings = get_settings()
    user_id = request.session.get("user_id")
    if user_id:
        user = db.get(User, user_id)
        if user:
            set_rls_user(db, user.id)
            return user
    if settings.demo_mode:
        user = ensure_demo_user(db)
        set_rls_user(db, user.id)
        return user
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def set_rls_user(db: Session, user_id: str) -> None:
    if db.get_bind().dialect.name == "postgresql":
        db.execute(
            text("select set_config('app.current_user_id', :user_id, true)"),
            {"user_id": user_id},
        )


def begin_oauth(request: Request, settings: Settings) -> str:
    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(status_code=503, detail="GitHub OAuth is not configured")
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    query = urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": f"{settings.backend_url}/auth/github/callback",
            "scope": "read:user user:email repo",
            "state": state,
        }
    )
    return f"https://github.com/login/oauth/authorize?{query}"


async def finish_oauth(
    code: str, state: str, request: Request, db: Session, settings: Settings
) -> User:
    expected = request.session.pop("oauth_state", None)
    if not expected or not secrets.compare_digest(expected, state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
            },
        )
        token_response.raise_for_status()
        token = token_response.json().get("access_token")
        if not token:
            raise HTTPException(status_code=400, detail="GitHub did not return an access token")
        profile_response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        profile_response.raise_for_status()
        profile = profile_response.json()
    user = db.scalar(select(User).where(User.github_id == str(profile["id"])))
    if not user:
        user = User(
            github_id=str(profile["id"]),
            github_username=profile["login"],
            email=profile.get("email"),
        )
        db.add(user)
    user.encrypted_github_token = encrypt_token(token, settings)
    db.commit()
    db.refresh(user)
    request.session["user_id"] = user.id
    return user
