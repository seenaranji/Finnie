"""Step 6: build the vector index from the knowledge base (full rebuild).

Pipeline:  load articles -> chunk -> embed -> save in Chroma.

Run:  .venv/bin/python -m scripts.index_knowledge_base
"""
import json
from datetime import datetime
from pathlib import Path

from src.core.config import load_settings
from src.core.logging import get_logger, setup_logging
from src.core.metrics import timer
from src.rag.chunker import chunk_structured
from src.rag.embedder import OpenAIEmbedder
from src.rag.loader import load_documents
from src.rag.vector_store import ChromaStore

settings = load_settings()
setup_logging(settings.app.log_level)
log = get_logger("index_knowledge_base")

with timer() as total_timing:
    documents = load_documents()
    chunks = [chunk for document in documents
              for chunk in chunk_structured(document, settings.rag.chunk_size)]
    log.info("loaded %d articles -> %d chunks (size=%d)", len(documents), len(chunks), settings.rag.chunk_size)

    embedding = OpenAIEmbedder(settings).embed([chunk.text for chunk in chunks])

    store = ChromaStore.from_settings(settings)
    store.reset()   # full rebuild: never leave old chunks of edited articles behind
    store.add(chunks, embedding.vectors)

summary = {
    "articles": len(documents), "chunks": len(chunks), "stored": store.count(),
    "chunk_size": settings.rag.chunk_size, "embedding_model": embedding.model,
    "dimensions": int(embedding.vectors.shape[1]), "input_tokens": embedding.input_tokens,
    "embedding_cost_usd": round(embedding.cost_usd, 8),
    "embedding_seconds": round(embedding.latency_s, 2), "total_seconds": round(total_timing["seconds"], 2),
}
print(json.dumps(summary, indent=2))

output_file = Path(f"benchmarks/results/step6_index_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(summary, indent=2))
print(f"saved: {output_file}")
print("\n*** INDEXING FINISHED ***")
