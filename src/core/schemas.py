"""Typed request/response formats shared by every agent.

Why Pydantic models and not plain dicts? A dict accepts anything, so a typo like
resp["sorces"] fails silently or crashes far from the cause. A model checks types
the moment it is created and fails loudly right there.
"""
from pydantic import BaseModel, Field


class Source(BaseModel):
    """One citation: where a piece of information came from."""
    id: str
    title: str
    url: str | None = None


class AgentRequest(BaseModel):
    # min_length rejects empty messages; max_length stops huge inputs (cost + abuse).
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"


class AgentResponse(BaseModel):
    text: str
    agent_name: str
    sources: list[Source] = Field(default_factory=list)
    disclaimer: str = ""
    # Free-form extras: latency, tokens, cost, whether a guardrail fired, etc.
    metadata: dict = Field(default_factory=dict)
