"""The shared state that flows through the Finnie graph (Step 20).

One typed dictionary. Every node receives the WHOLE state and returns only the keys it wants to change; LangGraph merges that update in.
How the merge works is decided per key:
  - a plain key (message, route, response ...) is OVERWRITTEN: the last write wins
  - `trace` has a REDUCER, operator.add: a node's list is APPENDED to what is already there. Without it, the router's trace entry would be
    erased by the agent's. Reducers are also what make parallel branches safe later: two branches can both append, and neither loses its entry.
Everything stored here is plain data (dicts, lists, numbers, strings), so a checkpointer can save it later (Step 21).
"""
import operator
from typing import Annotated, TypedDict


class FinnieState(TypedDict, total=False):
    message: str                                   # the user's message (input)
    session_id: str                                # which conversation (input; used for memory in Step 21)
    route: str                                     # set by the router: which agent should answer
    router: dict                                   # the router's full decision: model, tokens, cost, latency, whether it fell back
    response: dict                                 # the chosen agent's AgentResponse as a plain dict
    trace: Annotated[list[dict], operator.add]     # one entry per node that ran (REDUCER: appended, never overwritten)
