"""Turn text into vectors ("embeddings").

An Embedder interface lets the rest of the code (vector store, retriever) stay the same
if we swap the model. IMPORTANT: documents and questions must be embedded with the SAME
model, otherwise their vectors live in different "meaning spaces" and cannot be compared.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.core.config import Settings, load_settings
from src.core.llm import TRANSIENT_ERRORS
from src.core.logging import get_logger
from src.core.metrics import timer

log = get_logger(__name__)


@dataclass
class EmbeddingResult:
    vectors: np.ndarray   # shape (number of texts, dimensions)
    model: str
    input_tokens: int
    cost_usd: float
    latency_s: float
    batches: int


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Return one vector per text, in the same order."""


class OpenAIEmbedder(Embedder):
    def __init__(self, settings: Settings | None = None, batch_size: int = 100) -> None:
        self.settings = settings or load_settings()
        self.batch_size = batch_size   # texts per API call; our choice, not an API rule
        self.model = self.settings.rag.embedding_model
        self._client = OpenAI(
            api_key=self.settings.secrets.require_openai_key(),
            timeout=self.settings.llm.timeout_seconds,
            max_retries=0,   # we retry ourselves so retries are visible, as in llm.py
        )

    def _embed_batch(self, batch: list[str]):
        retrying = Retrying(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(self.settings.llm.max_retries + 1),
            wait=wait_exponential_jitter(initial=1, max=20),
            reraise=True,
        )
        return retrying(self._client.embeddings.create, model=self.model, input=batch)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            raise ValueError("embed() needs at least one text")
        vectors, tokens, batches = [], 0, 0
        with timer() as timing:
            for batch_start in range(0, len(texts), self.batch_size):
                response = self._embed_batch(texts[batch_start:batch_start + self.batch_size])
                # The API labels each result with its position; sort so order always matches input.
                vectors += [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
                tokens += response.usage.prompt_tokens
                batches += 1
        price = self.settings.llm.prices_per_million_tokens["embedding_model"]
        result = EmbeddingResult(np.array(vectors, dtype=np.float32), self.model, tokens,
                                 tokens * price.input / 1_000_000, timing["seconds"], batches)
        log.info("embedded %d texts model=%s dims=%d tokens=%d cost=$%.6f latency=%.2fs batches=%d",
                 len(texts), self.model, result.vectors.shape[1], tokens, result.cost_usd,
                 result.latency_s, batches)
        return result


def cosine_matrix(vectors: np.ndarray, other_vectors: np.ndarray | None = None) -> np.ndarray:
    """cosine(x, y) = (x . y) / (|x| * |y|), for every pair of rows. Range -1..1."""
    other_vectors = vectors if other_vectors is None else other_vectors
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    other_unit_vectors = other_vectors / np.linalg.norm(other_vectors, axis=1, keepdims=True)
    return unit_vectors @ other_unit_vectors.T
