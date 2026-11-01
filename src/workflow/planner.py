"""The planner (Step 21b): turns a message that needs several agents into a small plan of tasks with dependencies.

Output is a strict JSON schema, then CHECKED IN CODE (the model can still produce a bad plan): known agents, unique ids, dependencies that exist, no cycles,
between 1 and max_tasks tasks. `levels()` groups the tasks into waves: everything in a wave can run at the same time, a wave starts when the previous one is done.
A plan with ONE task is legal and means "a single agent is enough": the graph then returns that agent's answer unchanged.
"""
import json
from dataclasses import asdict, dataclass, field

from src.core.llm import LLMClient

AGENTS = ("finance_qa", "market", "portfolio", "goal", "news", "tax")
# The only result another task can truly NEED in this system is the user's own portfolio (its holdings or value feed a news, market or goal task). The planner
# sometimes invents other dependencies ("news after market" for two independent parts, about 10% of plans in the Step 21b benchmark); each one silently serializes
# work that could overlap. So dependencies on any other agent are dropped in code, and counted.
DEPENDENCY_SOURCES = ("portfolio",)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"tasks": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "agent": {"type": "string", "enum": list(AGENTS)}, "question": {"type": "string"},
                       "depends_on": {"type": "array", "items": {"type": "string"}}},
        "required": ["id", "agent", "question", "depends_on"], "additionalProperties": False}}},
    "required": ["tasks"], "additionalProperties": False,
}
PLAN_FORMAT = {"type": "json_schema", "json_schema": {"name": "plan", "strict": True, "schema": PLAN_SCHEMA}}

SYSTEM_PROMPT = """You plan how to answer a user's message that needs more than one specialist agent. Split it into tasks.

Agents:
- finance_qa: explains a concept from the knowledge base
- market: live or recent prices and performance, moving averages, volatility, 52-week range for named tickers or indexes
- portfolio: analyzes the user's OWN uploaded holdings
- goal: savings goal planning (how much to save for a target by a date)
- news: recent news headlines for a ticker or the markets
- tax: tax rules for investing and retirement accounts

Rules:
- One task per separate thing the message asks. Do not add tasks the message does not ask for. At most 4 tasks. If ONE agent can answer everything, return ONE task.
- Each task's question must be self-contained: name the ticker or topic explicitly and never use pronouns like "it" or "that".
- Give each task a short id (t1, t2, ...). depends_on lists the ids of tasks whose RESULT this task needs before it can start. Leave it empty if the task can start immediately.
- Use a dependency ONLY when a task truly needs another task's output (for example "the news on my smallest holding" needs the portfolio result to know which holding). Tasks that merely share a topic are independent.

Examples:
"Explain what a REIT is and tell me how VNQ did this week" -> t1 finance_qa "What is a REIT?" (no dependencies); t2 market "How did VNQ do this week?" (no dependencies)
"Find my smallest holding and tell me the news on it" -> t1 portfolio "Which of my holdings is the smallest?" (no dependencies); t2 news "What is the latest news on the smallest holding named in the portfolio result?" (depends_on t1)
"What is the tax on interest income and how is JPM doing?" -> t1 tax "How is interest income taxed?"; t2 market "How is JPM doing today?" (no dependencies)"""


class PlanError(ValueError):
    """The model's plan is unusable (unknown agent, missing dependency, cycle, too many tasks). The graph answers with the fixed fallback."""


@dataclass(frozen=True)
class Task:
    id: str
    agent: str
    question: str
    depends_on: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self) | {"depends_on": list(self.depends_on)}


def validate_plan(raw_tasks: list[dict], max_tasks: int) -> list[Task]:
    if not 1 <= len(raw_tasks) <= max_tasks:
        raise PlanError(f"a plan needs 1 to {max_tasks} tasks (got {len(raw_tasks)})")
    tasks = [Task(str(t["id"]).strip(), t["agent"], str(t["question"]).strip(), tuple(t["depends_on"])) for t in raw_tasks]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids) or not all(ids):
        raise PlanError("task ids must be unique and non-empty")
    for t in tasks:
        if t.agent not in AGENTS:
            raise PlanError(f"unknown agent {t.agent!r}")
        if not t.question:
            raise PlanError(f"task {t.id} has no question")
        for dep in t.depends_on:
            if dep not in ids or dep == t.id:
                raise PlanError(f"task {t.id} depends on {dep!r}, which is not another task of this plan")
    levels(tasks)   # raises PlanError on a cycle
    return tasks


def drop_unjustified_dependencies(tasks: list[Task]) -> tuple[list[Task], list[str]]:
    """Keep a dependency only if its source task is in DEPENDENCY_SOURCES. Returns the cleaned tasks and a readable list of what was dropped."""
    agent_of, cleaned, dropped = {t.id: t.agent for t in tasks}, [], []
    for t in tasks:
        keep = tuple(d for d in t.depends_on if agent_of[d] in DEPENDENCY_SOURCES)
        dropped += [f"{t.agent} after {agent_of[d]}" for d in t.depends_on if d not in keep]
        cleaned.append(Task(t.id, t.agent, t.question, keep))
    return cleaned, dropped


def levels(tasks: list[Task]) -> list[list[Task]]:
    """Waves of tasks that can run together. Raises PlanError if the dependencies contain a cycle."""
    done, remaining, waves = set(), list(tasks), []
    while remaining:
        wave = [t for t in remaining if all(dep in done for dep in t.depends_on)]
        if not wave:
            raise PlanError("the tasks depend on each other in a circle")
        waves.append(wave)
        done |= {t.id for t in wave}
        remaining = [t for t in remaining if t.id not in done]
    return waves


@dataclass
class Plan:
    tasks: list[Task]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    dropped: list[str] = field(default_factory=list)   # dependencies the model asked for that the code rule removed


class Planner:
    def __init__(self, llm: LLMClient, max_tasks: int = 4, model_key: str = "router_model") -> None:
        self.llm, self.max_tasks, self.model_key = llm, max_tasks, model_key

    def plan(self, message: str) -> Plan:
        result = self.llm.chat(f"MESSAGE: {message}", system=SYSTEM_PROMPT, model_key=self.model_key, temperature=0, max_output_tokens=500, response_format=PLAN_FORMAT)
        try:
            tasks = validate_plan(json.loads(result.text)["tasks"], self.max_tasks)
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise PlanError(f"unusable plan: {type(error).__name__}") from error
        tasks, dropped = drop_unjustified_dependencies(tasks)
        return Plan(tasks, result.model, result.input_tokens, result.output_tokens, result.cost_usd, result.latency_s, dropped)
