"""Step 6: search the saved index.

Run:  .venv/bin/python -m scripts.search_knowledge_base "your question" [--top-k 5] [--category etfs]
"""
import argparse

from src.core.config import load_settings
from src.rag.embedder import OpenAIEmbedder
from src.rag.vector_store import ChromaStore

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--top-k", type=int, default=None, help="default: rag.top_k from config.yaml")
parser.add_argument("--category", default=None, help="only search articles in this category")
arguments = parser.parse_args()

settings = load_settings()
store = ChromaStore.from_settings(settings)
if store.count() == 0:
    raise SystemExit("The index is empty. Run: .venv/bin/python -m scripts.index_knowledge_base")

query_vector = OpenAIEmbedder(settings).embed([arguments.question]).vectors[0]
where = {"category": arguments.category} if arguments.category else None
hits = store.search(query_vector, arguments.top_k or settings.rag.top_k, where)

print(f"question: {arguments.question!r}   (index holds {store.count()} chunks)\n")
for rank, hit in enumerate(hits, 1):
    print(f"{rank}. similarity={hit.similarity:.3f}  {hit.chunk_id}  "
          f"[{hit.metadata['category']}] section={hit.metadata['section']!r}")
    print(f"   {hit.text[:160].replace(chr(10), ' ')}...\n")
