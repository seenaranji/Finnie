"""Load settings: non-secret values from config.yaml, secrets from environment (.env).

Why two sources? config.yaml is committed to git and safe to share. API keys must
never be, so they come from environment variables that python-dotenv loads from .env.
"""
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class MissingSecretError(RuntimeError):
    """Raised when code needs a secret that is not set. Message says how to fix it."""


# --- Typed sections of config.yaml. A typo or wrong type fails loudly at load time. ---
class AppConfig(BaseModel):
    log_level: str = "INFO"


class Price(BaseModel):
    input: float = 0.0   # $ per 1M input tokens
    output: float = 0.0  # $ per 1M output tokens


class LLMConfig(BaseModel):
    answer_model: str
    router_model: str
    timeout_seconds: float = 30
    max_retries: int = 3
    prices_per_million_tokens: dict[str, Price] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    embedding_model: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.0
    chroma_path: str = "chroma_db"           # folder where Chroma saves vectors (git-ignored)
    collection_name: str = "finnie_knowledge_base"


class CacheConfig(BaseModel):
    quote_ttl_seconds: int = 60
    history_ttl_seconds: int = 900


class Secrets(BaseSettings):
    """Read from environment variables / .env. Missing is allowed here (None);
    code that actually needs a key calls the require_* method for a clear error."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    openai_api_key: str | None = None
    alpha_vantage_api_key: str | None = None

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise MissingSecretError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key


class Settings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    rag: RAGConfig
    cache: CacheConfig
    secrets: Secrets

    def masked(self) -> dict:
        """Settings as a dict that is safe to print: secrets show only last 4 chars."""
        data = self.model_dump(exclude={"secrets"})
        data["secrets"] = {
            name: ("<not set>" if not v else "****" + v[-4:])
            for name, v in self.secrets.model_dump().items()
        }
        return data


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(path) as config_file:
        raw = yaml.safe_load(config_file)
    return Settings(**raw, secrets=Secrets())
