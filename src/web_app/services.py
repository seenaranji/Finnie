"""The objects the API endpoints use, built lazily and once, so importing the app is instant and tests can swap any part.

Building the retriever and the graph loads indexes from disk, so it happens on first use, not at import. A lock makes two simultaneous first requests build one copy.
"""
import json
import threading

from src.core.config import PROJECT_ROOT, Settings, load_settings
from src.data.factory import build_market_data_service


class Services:
    def __init__(self, settings: Settings | None = None, market=None, retriever=None, graph=None, llm=None, articles=None, news=None, portfolios=None, auth=None) -> None:
        self.settings = settings or load_settings()
        self._market, self._retriever, self._graph, self._llm, self._metadata, self._articles = market, retriever, graph, llm, None, articles
        self._news, self._portfolios, self._auth = news, portfolios, auth
        self._lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}

    @property
    def market(self):
        with self._lock:
            self._market = self._market or build_market_data_service(self.settings)
            return self._market

    @property
    def news(self):
        with self._lock:
            if self._news is None:
                from src.data.news import NewsService
                from src.data.factory import build_cache
                self._news = NewsService(self.settings.news, build_cache(self.settings))
            return self._news

    @property
    def portfolios(self):
        """The saved-portfolio repository (PostgreSQL). Built on first use, so the API starts and serves everything else even if the database is down."""
        with self._lock:
            if self._portfolios is None:
                from src.data.portfolio_repository import PortfolioRepository
                self._portfolios = PortfolioRepository.from_settings(self.settings)
            return self._portfolios

    @property
    def auth(self):
        """Register / log in / verify tokens. Built on first use; needs the database (users) and JWT_SECRET."""
        with self._lock:
            if self._auth is None:
                from src.auth.service import AuthService
                self._auth = AuthService(None, self.settings)
            return self._auth

    @property
    def retriever(self):
        with self._lock:
            if self._retriever is None:
                from src.rag.hybrid_retriever import HybridRetriever
                self._retriever = HybridRetriever(settings=self.settings)
            return self._retriever

    @property
    def llm(self):
        with self._lock:
            if self._llm is None:
                from src.core.llm import LLMClient
                self._llm = LLMClient(self.settings)
            return self._llm

    @property
    def graph(self):
        with self._lock:
            if self._graph is None:
                from src.workflow.factory import build_default_graph
                self._graph = build_default_graph(settings=self.settings, llm=self._llm, retriever=self._retriever, market_service=self._market, news_service=self._news,
                                                  repository_getter=lambda: self.portfolios)
            return self._graph

    @property
    def metadata(self) -> dict:
        with self._lock:
            self._metadata = self._metadata or json.loads((PROJECT_ROOT / "src" / "data" / "ticker_metadata.json").read_text())
            return self._metadata

    def ready(self) -> dict[str, bool]:
        """Which parts are already built. Reading this never builds anything and never calls an outside service."""
        return {"market": self._market is not None, "retriever": self._retriever is not None, "graph": self._graph is not None}

    def session_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            return self._session_locks.setdefault(session_id, threading.Lock())   # grows by one small lock per session, like the in-memory conversation store

    def articles(self) -> list:
        """The knowledge base documents, read from disk once."""
        with self._lock:
            if self._articles is None:
                from src.rag.loader import load_documents
                self._articles = load_documents(PROJECT_ROOT / "knowledge_base")
            return self._articles
