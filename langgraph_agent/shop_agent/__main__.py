"""Terminal interface.

Run from the an_agent folder:
  .venv\\Scripts\\python.exe -m langgraph_agent.shop_agent                  # chat
  .venv\\Scripts\\python.exe -m langgraph_agent.shop_agent -q "top 3 products by revenue"
  .venv\\Scripts\\python.exe -m langgraph_agent.shop_agent --thread demo    # another conversation
  .venv\\Scripts\\python.exe -m langgraph_agent.shop_agent --reset-db       # rebuild sample data
  .venv\\Scripts\\python.exe -m langgraph_agent.shop_agent --graph          # Mermaid diagram
"""
import argparse
import json

from .config import SHOP_DB
from .database import init_db
from .graph import build_graph
from .runner import STOPPED, open_graph, run_agent
from .tracing import flush
from .utils import truncate


def ask_in_terminal(request: dict):
    print(f"\n🔐 {request['question']}")
    for action in request["actions"]:
        print(f"   • {action['tool']}({json.dumps(action['args'], ensure_ascii=False)})")
    try:
        return input("   approve? [y/N] ").strip() or "no"
    except EOFError:
        return "no"  # no one to ask -> deny


class ProgressPrinter:
    """prints each node's output as the graph runs"""

    def __init__(self):
        self.step = 0

    def __call__(self, update: dict):
        for node_name, node_output in update.items():
            if node_name == "agent":
                self.step += 1
                ai_msg = node_output["messages"][-1]
                for tc in ai_msg.tool_calls:
                    print(f"\n🔧 Step {self.step}: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
                if not ai_msg.tool_calls:
                    print(f"\n✅ FINAL ANSWER:\n{ai_msg.content}")
            elif node_name == "tools":
                for tool_msg in node_output["messages"]:
                    print(f"   → {truncate(str(tool_msg.content), 300)}")
            elif node_name == "approve" and node_output:
                print("   ✋ denied — nothing was changed")


def ask(graph, task: str, thread_id: str):
    answer = run_agent(graph, task, approver=ask_in_terminal,
                       on_update=ProgressPrinter(), thread_id=thread_id)
    if answer == STOPPED:
        print(f"\n⛔ {STOPPED}")


def chat(graph, thread_id: str):
    print(f"🛒 Shop assistant — thread '{thread_id}'. Type 'exit' to quit.")
    while True:
        try:
            task = input("\nyou> ").strip()
        except EOFError:
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break
        ask(graph, task, thread_id)


def main():
    parser = argparse.ArgumentParser(description="LangGraph shop assistant (v3)")
    parser.add_argument("--thread", default="default", help="conversation id to start or resume")
    parser.add_argument("-q", "--query", help="ask one question and exit")
    parser.add_argument("--reset-db", action="store_true", help="delete and re-create the sample shop data")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    if args.reset_db:
        init_db(reset=True)
        print(f"sample data re-created in {SHOP_DB}")
        return
    if args.graph:
        print(build_graph().get_graph().draw_mermaid())  # paste into https://mermaid.live
        return

    try:
        with open_graph() as graph:
            if args.query:
                ask(graph, args.query, args.thread)
            else:
                chat(graph, args.thread)
    finally:
        flush()


if __name__ == "__main__":
    main()
