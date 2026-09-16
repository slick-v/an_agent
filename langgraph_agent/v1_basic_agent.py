"""
v1 — agent.py rebuilt with LangGraph (single agent, ReAct loop)

What changes compared to agent.py:
  agent.py (hand-written)                 LangGraph (this file)
  -------------------------------------   ---------------------------------------
  messages = [...] list                   State (a TypedDict) that flows through the graph
  client.chat.completions.create(...)     "agent" node  -> calls the LLM
  for tool_call in msg.tool_calls: ...    "tools" node  -> ToolNode runs the tools
  if not msg.tool_calls: return           conditional edge -> route_after_agent()
  for step in range(max_steps)            recursion_limit in the run config
  manual Langfuse generation spans        Langfuse CallbackHandler traces every node

The graph:

    START --> agent --(has tool calls?)--> tools --+
                ^                                  |
                +----------------------------------+
                |
                +--(no tool calls)--> END

Run:  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py
"""
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()  # finds an_agent/.env (searches parent folders)

MODEL = "openai/gpt-oss-120b"
# shared notes folder: an_agent/notes (one level above this file)
NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes")
os.makedirs(NOTES_DIR, exist_ok=True)

langfuse = get_client()  # reads LANGFUSE_* from .env

SYSTEM_PROMPT = (
    "You are a knowledge assistant that reads and updates local notes. "
    "Always search/read existing notes before writing, so you don't "
    "overwrite useful content blindly. Explain your plan briefly before acting. "
    "Filenames are relative to the notes folder, e.g. 'agentic-ai.md' (no 'notes/' prefix)."
)


# ---------------------------------------------------------------------------
# 1. TOOLS
# Same logic as agent.py. The @tool decorator builds the JSON schema for the
# LLM from the function name, type hints and docstring — no hand-written
# TOOLS list or TOOL_MAP needed.
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
        return {"content": f.read()}


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
    # Stub — plug in a real search API (Tavily, SerpAPI, Bing) here.
    return {"results": f"[stub] Would search the web for: {query}"}


TOOLS = [search_files, read_file, write_file, web_search]


# ---------------------------------------------------------------------------
# 2. STATE
# The data every node reads and updates. `add_messages` is a *reducer*: when a
# node returns {"messages": [new_msg]}, LangGraph appends it to the list
# instead of replacing the list. (This is exactly what langgraph's built-in
# MessagesState does — written out here so you can see it.)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# 3. NODES
# A node is just a function: state in -> partial state update out.
# ---------------------------------------------------------------------------

# bind_tools sends the tool schemas along with every LLM request
llm = ChatGroq(model=MODEL).bind_tools(TOOLS)


def agent_node(state: AgentState):
    """Ask the LLM what to do next (answer, or call tools)."""
    messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}  # appended thanks to add_messages


# ToolNode reads the tool_calls on the last AI message, runs them, and returns
# one ToolMessage per call. handle_tool_errors=True sends exceptions (e.g. the
# safe_path ValueError, bad arguments) back to the model instead of crashing.
tool_node = ToolNode(TOOLS, handle_tool_errors=True)


# ---------------------------------------------------------------------------
# 4. EDGES
# A conditional edge is a function that returns the name of the next node.
# (langgraph.prebuilt.tools_condition does the same thing.)
# ---------------------------------------------------------------------------

def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END  # no tool calls = the model thinks it's done


# ---------------------------------------------------------------------------
# 5. BUILD + COMPILE THE GRAPH
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")  # after tools, always go back to the LLM

graph = builder.compile()


# ---------------------------------------------------------------------------
# 6. RUN
# ---------------------------------------------------------------------------

def run_agent(user_task: str, max_steps: int = 5):
    config = {
        "run_name": "agent_run",  # name of the root span in Langfuse
        # the Langfuse handler traces every node, LLM call and tool call
        "callbacks": [CallbackHandler()],
        # every node execution counts as one super-step;
        # one agent "step" = agent + tools, plus one for the final answer
        "recursion_limit": 2 * max_steps + 1,
    }
    inputs = {"messages": [HumanMessage(user_task)]}

    final = "Stopped — hit max steps without finishing."
    step = 0
    try:
        # stream_mode="updates" yields {node_name: what_that_node_returned}
        # after every node, so we can print progress like agent.py did
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
                        print(f"   → result: {tool_msg.content}")
    except GraphRecursionError:
        print(f"\n⛔ {final}")

    return final


if __name__ == "__main__":
    # print the graph structure (paste into https://mermaid.live to see it)
    print(graph.get_graph().draw_mermaid())

    with propagate_attributes(trace_name="agent_run", tags=["knowledge-assistant", "langgraph-v1"]):
        run_agent(
            "Search my notes for anything about 'science'. If a file exists, "
            "read it. Then create or update science.md with a short "
            "summary of what's there, adding a section on ReAct pattern if missing."
        )

    langfuse.flush()  # make sure all traces are sent before the script exits
