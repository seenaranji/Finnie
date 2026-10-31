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
    candidate_k: int = 20                    # chunks fetched from the store before filtering and reranking
    rerank: bool = False                     # second-stage re-ordering by an LLM (off until Step 8 shows it helps)
    dedupe_overlap: float = 0.8              # drop a chunk if this share of its words appears in a better chunk
    # --- Step 7c: hybrid search. Default stays "dense" so Step 7 behaviour is unchanged until Step 8 measures. ---
    retrieval_mode: str = "dense"            # dense | bm25 | hybrid
    keyword_index_path: str = "keyword_index"   # where the BM25 index is saved (git-ignored)
    rrf_k: int = 60                          # RRF damping constant
    candidates_per_retriever: int = 20       # how many chunks each retriever contributes to the fusion
    bm25_min_score: float | None = None      # a BM25 score at/above this lets a chunk in even if its cosine is below the threshold; None = BM25 cannot open the gate


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
