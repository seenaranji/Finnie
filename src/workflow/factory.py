"""Build the whole Finnie graph from config.yaml, so the demo, the evaluations and (later) the API construct it the same way.

Anything can be replaced for a test or a fault-injection run: pass `llm`, `router_llm` (the router's model client), `retriever`, `market_service`
or `news_service`; the rest is built from settings.
"""
import json

from langgraph.checkpoint.memory import InMemorySaver

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.goal_agent import GoalPlanningAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.market_tools import MarketTools
from src.agents.news_agent import NewsAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.tax_agent import TaxEducationAgent
from src.core.config import PROJECT_ROOT, load_settings
from src.core.llm import LLMClient
from src.data.factory import build_market_data_service
from src.data.news import NewsService
from src.rag.hybrid_retriever import HybridRetriever
from src.workflow.graph import Agents, build_graph
from src.workflow.router import Router


def build_default_graph(checkpointer="memory", settings=None, llm=None, router_llm=None, retriever=None, market_service=None, news_service=None,
                        router_prompt=None, router=None):
    """checkpointer: "memory" (default, an in-process InMemorySaver), None (no memory), or any LangGraph checkpointer.
    `router` replaces the router entirely (used to simulate a world without the scope gate)."""
    settings = settings or load_settings()
    llm = llm or LLMClient(settings)
    retriever = retriever or HybridRetriever(settings=settings)
    market_service = market_service or build_market_data_service(settings)
    news_service = news_service or NewsService(settings.news)
    metadata = json.loads((PROJECT_ROOT / "src" / "data" / "ticker_metadata.json").read_text())
    agents = Agents(finance_qa=FinanceQAAgent(llm, retriever), market=MarketAnalysisAgent(llm, MarketTools(market_service)),
                    portfolio=PortfolioAnalysisAgent(llm, market_service, metadata), goal=GoalPlanningAgent(llm, settings.goal_planning),
                    news=NewsAgent(llm, news_service, defense="full", settings=settings), tax=TaxEducationAgent(llm, retriever))
    saver = InMemorySaver() if checkpointer == "memory" else checkpointer
    router = router or Router(router_llm or llm, **({"prompt_version": router_prompt} if router_prompt else {}))   # default: the Router's own default prompt
    return build_graph(llm, router, agents, settings.workflow, checkpointer=saver)


def config_for(session_id: str) -> dict:
    """The LangGraph config that selects a conversation: same session id, same saved state."""
    return {"configurable": {"thread_id": session_id}}
