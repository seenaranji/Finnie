"""Step 7c: build (or rebuild) the BM25 keyword index WITHOUT re-embedding.

Chunks the knowledge base exactly like scripts.index_knowledge_base, then checks that the chunk
ids match what Chroma holds. If they differ, the two indexes would disagree, so it stops.

Run:  .venv/bin/python -m scripts.build_keyword_index
"""
from src.core.config import load_settings
from src.core.logging import get_logger, setup_logging
from src.core.metrics import timer
from src.rag.chunker import chunk_structured
from src.rag.keyword_index import KeywordIndex
from src.rag.loader import load_documents
from src.rag.vector_store import ChromaStore

settings = load_settings()
setup_logging(settings.app.log_level)
log = get_logger("build_keyword_index")

with timer() as timing:
    chunks = [chunk for document in load_documents()
              for chunk in chunk_structured(document, settings.rag.chunk_size)]
    store = ChromaStore.from_settings(settings)
    if store.count() != len(chunks):
        raise SystemExit(f"vector store has {store.count()} chunks but the knowledge base makes {len(chunks)}. "
                         f"Run scripts.index_knowledge_base first.")
    index = KeywordIndex.build(chunks)
    index.save(settings.rag.keyword_index_path)

print(f"keyword index: {len(index)} chunks, fingerprint {index.fingerprint}, "
      f"saved to {settings.rag.keyword_index_path}/ in {timing['seconds']:.3f} s")
print("\n*** KEYWORD INDEXING FINISHED ***")
