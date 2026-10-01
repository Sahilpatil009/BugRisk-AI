import logging

import httpx
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .risk import ExplanationData

logger = logging.getLogger(__name__)


class RewriteResponse(BaseModel):
    recommendations: list[str] = Field(min_length=1, max_length=3)


def rewrite_recommendations(
    actions: list[str], explanations: list[ExplanationData], settings: Settings
) -> list[str]:
    """Optionally rewrite actions; the request and response contain no writable score field."""
    if not settings.llm_rewrite_url:
        return actions
    try:
        response = httpx.post(
            settings.llm_rewrite_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"}
            if settings.llm_api_key
            else {},
            json={
                "task": "Rewrite these evidence-backed review actions without changing meaning.",
                "actions": actions,
                "factors": explanations,
                "response_schema": {"recommendations": ["string"]},
            },
            timeout=8,
        )
        response.raise_for_status()
        parsed = RewriteResponse.model_validate(response.json())
        return [item.strip() for item in parsed.recommendations if item.strip()]
    except (httpx.HTTPError, ValidationError, ValueError):
        logger.warning("recommendation_rewrite_failed", exc_info=True)
        return actions
