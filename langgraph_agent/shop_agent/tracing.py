"""Langfuse tracing.

Two layers:
  trace_turn()          one parent trace per user message — the pause for approval
                        and the resume end up in the SAME trace
  langfuse_callbacks()  the LangChain callback that records every node, LLM call
                        (tokens, model) and tool call inside that trace
"""
from contextlib import contextmanager

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from .config import TRACE_NAME, TRACE_TAGS

langfuse = get_client()  # reads LANGFUSE_* from .env


@contextmanager
def trace_turn(user_message: str, thread_id: str):
    # session_id groups every turn of a conversation into one Langfuse session
    with propagate_attributes(trace_name=TRACE_NAME, session_id=thread_id, tags=TRACE_TAGS):
        with langfuse.start_as_current_observation(
            name=TRACE_NAME, as_type="agent", input=user_message
        ) as span:
            yield span


def langfuse_callbacks():
    return [CallbackHandler()]


def flush():
    """send everything before the process exits"""
    langfuse.flush()
