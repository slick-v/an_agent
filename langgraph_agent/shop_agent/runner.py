"""Running one user message through the graph, including approval pauses.

No terminal code here: how to ask for approval and how to show progress are
passed in, so the same runner can later sit behind an API or a web UI.
"""
import sqlite3
from contextlib import closing, contextmanager
from typing import Callable

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from .config import CHECKPOINT_DB, MAX_STEPS
from .database import init_db
from .graph import build_graph
from .tracing import langfuse_callbacks, trace_turn

STOPPED = "Stopped — hit max steps without finishing."


@contextmanager
def open_graph():
    """shop data + memory database + compiled graph, closed cleanly afterwards"""
    init_db()
    with closing(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)) as conn:
        yield build_graph(checkpointer=SqliteSaver(conn))


def run_agent(
    graph,
    user_task: str,
    approver: Callable[[dict], str],
    on_update: Callable[[dict], None] = lambda update: None,
    thread_id: str = "default",
    max_steps: int = MAX_STEPS,
):
    config = {
        "configurable": {"thread_id": thread_id},  # which saved conversation to continue
        "run_name": "graph",
        "callbacks": langfuse_callbacks(),
        # a step can be agent + approve + tools, plus one for the final answer
        "recursion_limit": 3 * max_steps + 1,
    }
    inputs = {"messages": [HumanMessage(user_task)]}  # the checkpointer adds the history
    approvals = []

    with trace_turn(user_task, thread_id) as span:
        try:
            while True:
                for update in graph.stream(inputs, config, stream_mode="updates"):
                    on_update(update)
                state = graph.get_state(config)
                if not state.interrupts:
                    break  # finished
                # paused at the approve node -> ask, then resume from the same point
                request = state.interrupts[0].value
                decision = approver(request)
                approvals.append({"actions": request["actions"], "decision": decision})
                inputs = Command(resume=decision)
        except GraphRecursionError:
            span.update(output=STOPPED, level="WARNING", metadata={"approvals": approvals})
            return STOPPED

        answer = graph.get_state(config).values["messages"][-1].content
        span.update(output=answer, metadata={"approvals": approvals})
        return answer
