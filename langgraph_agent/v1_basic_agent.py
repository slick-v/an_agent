"""
v2 — LangGraph agent with memory, new tools and reliability fixes
(v1 = plain LangGraph rebuild of agent.py — see git history, commit "lgraphv1")

What v2 adds on top of v1:
  MEMORY        SqliteSaver checkpointer + thread_id -> the agent remembers earlier
                turns of a conversation, even after the script restarts
  CHAT          interactive loop (or one-shot with -q), resume any thread with --thread
  TOOLS         list_notes, get_current_date, real web_search (Tavily)
  RELIABILITY   LLM timeout + retries + fallback model,
                old messages trimmed to fit the context window,
                long tool output truncated
  LANGFUSE      every turn of a thread is grouped into one Langfuse *session*

The graph is the same as v1 — memory lives in the checkpointer, not in the graph:

    START --> agent --(has tool calls?)--> tools --+
                ^                                  |
                +----------------------------------+
                |
                +--(no tool calls)--> END

Run:
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py                 # chat, thread "default"
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py --thread study  # chat in another thread
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py -q "list my notes"
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py --graph         # print Mermaid diagram
"""
import argparse
import os
import sqlite3
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

load_dotenv()  # finds an_agent/.env (searches parent folders)

# ---------------------------------------------------------------------------
# 0. SETTINGS
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # an_agent/
NOTES_DIR = os.path.join(ROOT_DIR, "notes")
DATA_DIR = os.path.join(ROOT_DIR, "data")  # checkpoint database lives here (git-ignored)
os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.3-70b-versatile"  # used only if MODEL keeps failing
LLM_TIMEOUT = 60           # seconds per LLM request
LLM_MAX_RETRIES = 3        # retries on timeouts / rate limits / 5xx
MAX_CONTEXT_TOKENS = 6000  # history sent to the LLM is trimmed to roughly this size
MAX_TOOL_CHARS = 8000      # longer tool output is cut off

langfuse = get_client()  # reads LANGFUSE_* from .env

SYSTEM_PROMPT = (
    "You are a knowledge assistant that reads and updates local notes. "
    "Always search/read existing notes before writing, so you don't "
    "overwrite useful content blindly. Explain your plan briefly before acting. "
    "Filenames are relative to the notes folder, e.g. 'agentic-ai.md' (no 'notes/' prefix). "
    "Use web_search only for information that is not in the notes, and "
    "get_current_date whenever you need today's date."
)


# ---------------------------------------------------------------------------
# 1. TOOLS
# The @tool decorator builds the JSON schema for the LLM from the function
# name, type hints and docstring.
# ---------------------------------------------------------------------------

def safe_path(filename: str):
    """resolve filename inside NOTES_DIR, rejecting anything that escapes it"""
    # tolerate the model prefixing "notes/" itself
    name = filename.replace("\\", "/")
    if name.startswith("notes/"):
        name = name[len("notes/"):]
    path = os.path.realpath(os.path.join(NOTES_DIR, name))
    if os.path.commonpath([path, os.path.realpath(NOTES_DIR)]) != os.path.realpath(NOTES_DIR):
        raise ValueError(f"{filename} is outside the notes folder")
    return path


def truncate(text: str, limit: int = MAX_TOOL_CHARS):
    """keep tool output small enough for the context window"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"


@tool
def list_notes():
    """List all note files with their size in characters"""
    notes = []
    for fname in sorted(os.listdir(NOTES_DIR)):
        path = os.path.join(NOTES_DIR, fname)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                notes.append({"filename": fname, "chars": len(f.read())})
    return {"notes": notes} if notes else {"notes": [], "note": "the notes folder is empty"}


@tool
def search_files(query: str):
    """Search local notes for a keyword in filename or content"""
    results = []
    for fname in os.listdir(NOTES_DIR):
        path = os.path.join(NOTES_DIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if query.lower() in fname.lower() or query.lower() in content.lower():
            results.append(fname)

    return {"matches": results} if results else {"matches": [], "note": "no files found"}


@tool
def read_file(filename: str):
    """Read the full content of a specific note file"""
    path = safe_path(filename)
    if not os.path.exists(path):
        return {"error": f"{filename} does not exist"}
    with open(path, "r", encoding="utf-8") as f:
        return {"content": truncate(f.read())}


@tool
def write_file(filename: str, content: str, mode: Literal["overwrite", "append"] = "overwrite"):
    """Write or append content to a note file"""
    path = safe_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_mode = "a" if mode == "append" else "w"
    with open(path, file_mode, encoding="utf-8") as f:
        f.write(content)
    return {"status": "success", "filename": filename, "mode": mode}


@tool
def web_search(query: str):
    """Search the web for current information not in local notes"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "web search is not configured: add TAVILY_API_KEY to .env"}
    response = TavilyClient(api_key=api_key).search(query, max_results=3, timeout=30)
    return {
        "results": [
            {"title": r["title"], "url": r["url"], "content": truncate(r["content"], 1000)}
            for r in response.get("results", [])
        ]
    }


@tool
def get_current_date():
    """Get today's date and weekday (the model does not know it otherwise)"""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A")}


TOOLS = [list_notes, search_files, read_file, write_file, web_search, get_current_date]


# ---------------------------------------------------------------------------
# 2. STATE
# `add_messages` is a *reducer*: a node returning {"messages": [msg]} appends
# to the list. With a checkpointer, this list is saved after every node, per
# thread_id — that saved list IS the agent's short-term memory.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# 3. NODES
# ---------------------------------------------------------------------------

def make_llm(model: str):
    # timeout + max_retries: one slow/rate-limited request won't hang or kill the run
    return ChatGroq(model=model, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES).bind_tools(TOOLS)


# if the main model still fails after its retries, the fallback model answers
llm = make_llm(MODEL).with_fallbacks([make_llm(FALLBACK_MODEL)])


def drop_unanswered_tool_calls(messages: list[AnyMessage]):
    """
    If a previous turn hit the step limit, history can end with an AI message
    whose tool calls never ran. The API rejects that, so strip those calls.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    cleaned = []
    for m in messages:
        if isinstance(m, AIMessage) and any(tc["id"] not in answered for tc in m.tool_calls):
            m = AIMessage(content=m.content or "(stopped before running tools: step limit reached)")
        cleaned.append(m)
    return cleaned


def agent_node(state: AgentState):
    """Ask the LLM what to do next (answer, or call tools)."""
    history = drop_unanswered_tool_calls(state["messages"])
    # memory grows every turn, so only send the most recent part to the LLM.
    # start_on="human" keeps tool results together with the call that made them.
    # (trimming only affects this request — the full history stays saved)
    history = trim_messages(
        history,
        max_tokens=MAX_CONTEXT_TOKENS,
        token_counter="approximate",
        strategy="last",
        start_on="human",
    )
    response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + history)
    return {"messages": [response]}


# ToolNode runs the tool calls on the last AI message; handle_tool_errors=True
# sends exceptions (bad args, safe_path errors) back to the model instead of crashing
tool_node = ToolNode(TOOLS, handle_tool_errors=True)


# ---------------------------------------------------------------------------
# 4. EDGES
# ---------------------------------------------------------------------------

def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END  # no tool calls = the model thinks it's done


# ---------------------------------------------------------------------------
# 5. BUILD + COMPILE THE GRAPH (with a checkpointer = memory)
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")  # after tools, always go back to the LLM

# SqliteSaver stores the state after every node in data/checkpoints.sqlite.
# (MemorySaver would do the same in RAM, but forgets everything on exit.)
conn = sqlite3.connect(os.path.join(DATA_DIR, "checkpoints.sqlite"), check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 6. RUN
# ---------------------------------------------------------------------------

def run_agent(user_task: str, thread_id: str = "default", max_steps: int = 5):
    config = {
        # thread_id picks which saved conversation to load and continue
        "configurable": {"thread_id": thread_id},
        "run_name": "agent_run",  # name of the root span in Langfuse
        "callbacks": [CallbackHandler()],
        # one agent "step" = agent + tools node, plus one for the final answer
        "recursion_limit": 2 * max_steps + 1,
    }
    # only the NEW message is passed in — the checkpointer adds the history
    inputs = {"messages": [HumanMessage(user_task)]}

    final = "Stopped — hit max steps without finishing."
    step = 0
    # session_id groups all turns of this thread into one Langfuse session
    with propagate_attributes(
        trace_name="agent_run",
        session_id=thread_id,
        tags=["knowledge-assistant", "langgraph-v2"],
    ):
        try:
            for update in graph.stream(inputs, config, stream_mode="updates"):
                for node_name, node_output in update.items():
                    if node_name == "agent":
                        step += 1
                        ai_msg = node_output["messages"][-1]
                        for tc in ai_msg.tool_calls:
                            print(f"\n🔧 Step {step}: calling {tc['name']}({tc['args']})")
                        if not ai_msg.tool_calls:
                            final = ai_msg.content
                            print(f"\n✅ FINAL ANSWER:\n{final}")
                    elif node_name == "tools":
                        for tool_msg in node_output["messages"]:
                            print(f"   → result: {truncate(tool_msg.content, 300)}")
        except GraphRecursionError:
            print(f"\n⛔ {final}")

    return final


def history_size(thread_id: str):
    """how many messages are saved for this thread"""
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    return len(state.values.get("messages", []))


def chat(thread_id: str):
    print(f"💬 thread '{thread_id}' — {history_size(thread_id)} saved messages. "
          "Type 'exit' to quit.")
    while True:
        try:
            task = input("\nyou> ").strip()
        except EOFError:
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break
        run_agent(task, thread_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph notes agent (v2: memory)")
    parser.add_argument("--thread", default="default", help="conversation id to start or resume")
    parser.add_argument("-q", "--query", help="ask one question and exit")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    try:
        if args.graph:
            # paste into https://mermaid.live to see it
            print(graph.get_graph().draw_mermaid())
        elif args.query:
            run_agent(args.query, args.thread)
        else:
            chat(args.thread)
    finally:
        langfuse.flush()  # make sure all traces are sent before the script exits
        conn.close()
