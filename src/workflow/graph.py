"""The Finnie graph, part 1 (Step 20):   START -> router -> (conditional edge) -> finance_qa | market -> END

A NODE is a plain function: it receives the whole state and returns a partial update. An AGENT is a class with its own prompt, tools and
guardrails (Steps 9 to 19). Here each agent node is a thin wrapper that calls an agent and writes its answer into the state.
"""
import time

from langgraph.graph import END, START, StateGraph

from src.agents.base_agent import BaseAgent
from src.core.schemas import AgentRequest
from src.workflow.router import Router
from src.workflow.state import FinnieState


def build_graph(router: Router, finance_qa: BaseAgent, market: BaseAgent):
    def router_node(state: FinnieState) -> dict:
        decision = router.classify(state["message"])
        return {"route": decision.route, "router": decision.as_dict(),
                "trace": [{"node": "router", "route": decision.route, "seconds": round(decision.latency_s, 3),
                           "tokens": decision.input_tokens + decision.output_tokens, "cost_usd": decision.cost_usd, "fell_back": decision.fell_back}]}

    def agent_node(node_name: str, agent: BaseAgent):
        def run(state: FinnieState) -> dict:
            start = time.perf_counter()
            response = agent.handle(AgentRequest(message=state["message"], session_id=state.get("session_id", "default")))
            meta = response.metadata
            return {"response": response.model_dump(mode="json"),
                    "trace": [{"node": node_name, "agent": response.agent_name, "seconds": round(time.perf_counter() - start, 3),
                               "model": meta.get("model"), "cost_usd": meta.get("cost_usd"), "tools": meta.get("tool_names")}]}
        return run

    def pick_route(state: FinnieState) -> str:
        """The conditional edge: LangGraph calls this after the router node and follows the name it returns."""
        return state["route"]

    graph = StateGraph(FinnieState)
    graph.add_node("router", router_node)
    graph.add_node("finance_qa", agent_node("finance_qa", finance_qa))
    graph.add_node("market", agent_node("market", market))
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", pick_route, {"finance_qa": "finance_qa", "market": "market"})
    graph.add_edge("finance_qa", END)
    graph.add_edge("market", END)
    return graph.compile()
