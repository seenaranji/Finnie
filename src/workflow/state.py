"""The shared state that flows through the Finnie graph (Steps 20-21).

One typed dictionary. Every node receives the WHOLE state and returns only the keys it wants to change; LangGraph merges that update in.
How the merge works is decided per key:
  - a plain key (message, route, response ...) is OVERWRITTEN: the last write wins
  - `trace` and `history` have a REDUCER, operator.add: a node's list is APPENDED to what is already there. Without it the router's trace entry would be
    erased by the agent's. Reducers are also what make parallel branches safe later: two branches can both append and neither loses its entry.
With a CHECKPOINTER the state is saved after every step under a thread id (the session id). The next call with the same id starts from the saved state,
so `history`, `goal_state` and `portfolio_csv` carry over between turns, while keys every turn rewrites (route, response, error) are replaced.
Everything stored here is plain data (dicts, lists, numbers, strings) so it can be saved.
"""
import operator
from typing import Annotated, TypedDict


class FinnieState(TypedDict, total=False):
    message: str                                   # the user's message this turn (input)
    session_id: str                                # which conversation (input; also the checkpointer's thread id)
    portfolio_csv: str                             # the user's holdings (input; supplied once, remembered by the checkpointer)
    turn: int                                      # 1 for the first user message of the session, 2 for the second ...
    resolved_message: str                          # the message rewritten as a standalone question ("it" -> "VOO"); equals message when nothing needed resolving
    route: str                                     # set by the router: which node answers (an agent, clarify, or fallback)
    router: dict                                   # the router's decision: confidence, tokens, cost, latency, whether the policy overrode it
    response: dict                                 # the answer as a plain dict (AgentResponse)
    error: dict | None                             # set when a node failed and a safe fallback answer was given; None otherwise
    goal_state: dict                               # the goal agent's partly filled slots, kept between turns until a plan is made
    history: Annotated[list[dict], operator.add]   # the conversation so far: {"role": "user"|"assistant", "content": ..., "route": ...}
    trace: Annotated[list[dict], operator.add]     # one entry per node that ran, for every turn (REDUCER: appended, never overwritten)
