"""The LangGraph part: state, nodes, edges, and building the graph.

    START --> agent --(read-only tools)--> tools --> agent ... --> END
                |                            ^
                +--(write/notify tools)--> approve --(yes)--+
                                             |
                                             +--(no)--> agent   (told: "user denied")
"""
import json
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage, trim_messages
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .config import FALLBACK_MODEL, LLM_MAX_RETRIES, LLM_TIMEOUT, MAX_CONTEXT_TOKENS, MODEL
from .prompts import SYSTEM_PROMPT
from .tools import SENSITIVE_TOOLS, TOOLS


# ---------------------------------------------------------------------------
# STATE — `add_messages` appends new messages instead of replacing the list.
# With a checkpointer this list is saved per thread_id: that's the memory.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------

def make_llm(model: str):
    # timeout + max_retries: one slow/rate-limited request won't hang or kill the run
    return ChatGroq(model=model, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES).bind_tools(TOOLS)


# if the main model still fails after its retries, the fallback model answers
llm = make_llm(MODEL).with_fallbacks([make_llm(FALLBACK_MODEL)])


def drop_unanswered_tool_calls(messages: list[AnyMessage]):
    """
    If a turn stopped early (step limit, or an approval that was never answered),
    history can hold tool calls that never ran. The API rejects that.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    cleaned = []
    for m in messages:
        if isinstance(m, AIMessage) and any(tc["id"] not in answered for tc in m.tool_calls):
            m = AIMessage(content=m.content or "(these actions were not run)")
        cleaned.append(m)
    return cleaned


def agent_node(state: AgentState):
    """The brain: ask the LLM to answer or pick tools."""
    history = drop_unanswered_tool_calls(state["messages"])
    # only the recent part of memory is sent; the full history stays saved
    history = trim_messages(
        history,
        max_tokens=MAX_CONTEXT_TOKENS,
        token_counter="approximate",
        strategy="last",
        start_on="human",
    )
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + history)
    return {"messages": [response]}


def approve_node(state: AgentState) -> Command[Literal["tools", "agent"]]:
    """
    ASK PERMISSION. interrupt() pauses the graph and saves it in the checkpointer.
    The caller asks the user and resumes with Command(resume="yes"/"no").
    On resume this node runs again from the top, and interrupt() returns the answer.
    """
    last = state["messages"][-1]
    decision = interrupt({
        "question": "The assistant wants to run these actions. Allow?",
        "actions": [{"tool": tc["name"], "args": tc["args"]} for tc in last.tool_calls],
    })

    # strip a stray byte-order mark (Windows pipes can add one)
    if str(decision).replace("﻿", "").strip().lower() in {"y", "yes"}:
        return Command(goto="tools")

    # denied: answer every tool call so the history stays valid, then let the LLM respond
    denied = [
        ToolMessage(
            content=json.dumps({"error": "The user denied permission. Nothing was changed."}),
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in last.tool_calls
    ]
    return Command(goto="agent", update={"messages": denied})


# ToolNode runs the tool calls; handle_tool_errors=True sends exceptions
# (bad input, business-rule errors) back to the model instead of crashing
tool_node = ToolNode(TOOLS, handle_tool_errors=True)


# ---------------------------------------------------------------------------
# EDGES
# ---------------------------------------------------------------------------

def route_after_agent(state: AgentState) -> Literal["approve", "tools", "__end__"]:
    last = state["messages"][-1]
    if not last.tool_calls:
        return END
    if any(tc["name"] in SENSITIVE_TOOLS for tc in last.tool_calls):
        return "approve"  # code decides this, not the LLM
    return "tools"


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """interrupt() needs a checkpointer: the paused state is saved there until answered"""
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("approve", approve_node)  # its exits come from the Command[...] type hint
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)
