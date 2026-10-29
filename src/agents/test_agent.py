"""A trivial agent used to exercise the base-class pipeline. Not a real Finnie agent."""
from src.agents.base_agent import BaseAgent
from src.core.llm import LLMResult
from src.core.schemas import AgentRequest


class TutorTestAgent(BaseAgent):
    name = "tutor_test"
    system_prompt = (
        "You are Finnie, a friendly finance tutor for beginners. Explain simply, "
        "under 120 words. You teach; you never give personal investment advice."
    )

    temperature = 0.2  # evals set this to 0 for repeatability

    def answer(self, request: AgentRequest, system: str) -> LLMResult:
        return self.llm.chat(request.message, system=system, temperature=self.temperature)
