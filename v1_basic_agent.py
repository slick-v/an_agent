"""
v3 — Shop assistant: real-world tools on a SQLite practice database
(v1 = LangGraph rebuild — commit "lgraphv1"; v2 = memory + notes tools — commit "checkpointer")

A small practice shop lives in data/shop.db (created + seeded automatically):
  customers(id, name, email, city)
  products(id, name, category, price, stock)
  orders(id, customer_id, product_id, quantity, unit_price, discount, total, status, created_at)
  notifications(id, customer_id, channel, message, status, created_at)

Tools, by capability:
  READ        get_schema, query_database (read-only SQL)
  SUMMARIZE   summarize_sales
  COMPUTE     quote_order (price x qty, discount, tax, stock check)
  CALCULATE   calculator (safe maths)
  WRITE       create_order, update_order_status          <- need approval
  NOTIFY      send_notification (saved to an outbox)     <- needs approval
  UTILITY     get_current_date
  ASK PERMISSION  the "approve" node pauses the graph with interrupt()
                  before any tool that changes data, and waits for yes/no

The graph:

    START --> agent --(read-only tools)--> tools --> agent ... --> END
                |                            ^
                +--(write/notify tools)--> approve --(yes)--+
                                             |
                                             +--(no)--> agent   (told: "user denied")

Kept from v2: SQLite checkpointer memory (thread_id), retries/timeout/fallback
model, history trimming, tool-output truncation, Langfuse sessions.

Run:
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py                  # chat
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py -q "top 3 products by revenue"
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py --thread demo    # another conversation
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py --reset-db       # rebuild sample data
  .venv\\Scripts\\python.exe langgraph_agent\\v1_basic_agent.py --graph          # Mermaid diagram
"""
import argparse
import ast
import json
import math
import operator
import os
import random
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional, TypedDict

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
from langgraph.types import Command, interrupt

load_dotenv()  # finds an_agent/.env (searches parent folders)

# ---------------------------------------------------------------------------
# 0. SETTINGS
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # an_agent/
DATA_DIR = os.path.join(ROOT_DIR, "data")  # git-ignored
os.makedirs(DATA_DIR, exist_ok=True)
SHOP_DB = os.path.join(DATA_DIR, "shop.db")                 # the practice business data
CHECKPOINT_DB = os.path.join(DATA_DIR, "checkpoints.sqlite")  # the agent's memory

MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.3-70b-versatile"  # used only if MODEL keeps failing
LLM_TIMEOUT = 60           # seconds per LLM request
LLM_MAX_RETRIES = 3        # retries on timeouts / rate limits / 5xx
MAX_CONTEXT_TOKENS = 6000  # history sent to the LLM is trimmed to roughly this size
MAX_TOOL_CHARS = 8000      # longer tool output is cut off
MAX_QUERY_ROWS = 50        # query_database never returns more rows than this

TAX_RATE = 0.18
DISCOUNT_CODES = {"SAVE10": 0.10, "WELCOME5": 0.05}
ORDER_STATUSES = ("pending", "shipped", "delivered", "cancelled")

# tools that change data or contact people -> must pass the approve node first
SENSITIVE_TOOLS = {"create_order", "update_order_status", "send_notification"}

langfuse = get_client()  # reads LANGFUSE_* from .env

SYSTEM_PROMPT = (
    "You are an operations assistant for a small online shop in India backed by a SQLite database. "
    "All prices are in Indian rupees (₹). "
    "Call get_schema before writing SQL, and use query_database for questions about the data. "
    "Never do arithmetic in your head: use calculator, or quote_order for order prices. "
    "Always quote_order before create_order. "
    "Actions that change data or notify customers are paused for the user's approval "
    "automatically, so do not ask for confirmation yourself — just call the tool. "
    "If the user denies an action, do not retry it unless they ask again. "
    "Never tell a customer something happened unless a tool result confirms it. "
    "Use get_current_date for anything involving 'today' or recent dates. "
    "Keep answers short; use tables for lists."
)


# ---------------------------------------------------------------------------
# 1. PRACTICE DATABASE
# ---------------------------------------------------------------------------

def connect(read_only: bool = False):
    """new connection per call (tools may run in parallel threads)"""
    if read_only:
        # mode=ro: SQLite itself refuses INSERT/UPDATE/DELETE/DROP on this connection
        conn = sqlite3.connect(f"file:{SHOP_DB}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(SHOP_DB)
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_dicts(rows):
    return [dict(r) for r in rows]


def init_db(reset: bool = False):
    """create the tables and fill them with sample data (only if empty)"""
    if reset and os.path.exists(SHOP_DB):
        os.remove(SHOP_DB)

    with closing(connect()) as conn, conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                city TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL CHECK (price >= 0),
                stock INTEGER NOT NULL CHECK (stock >= 0)
            );
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL,
                discount REAL NOT NULL DEFAULT 0,
                total REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                channel TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

        if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
            return  # already seeded

        conn.executemany(
            "INSERT INTO customers (name, email, city) VALUES (?, ?, ?)",
            [
                ("Asha Rao", "asha@example.com", "Bengaluru"),
                ("Ravi Kumar", "ravi@example.com", "Hyderabad"),
                ("Meera Iyer", "meera@example.com", "Chennai"),
                ("Arjun Singh", "arjun@example.com", "Delhi"),
                ("Priya Nair", "priya@example.com", "Kochi"),
                ("Karan Mehta", "karan@example.com", "Mumbai"),
            ],
        )
        conn.executemany(
            "INSERT INTO products (name, category, price, stock) VALUES (?, ?, ?, ?)",
            [
                ("Wireless Mouse", "electronics", 799.0, 40),
                ("Mechanical Keyboard", "electronics", 3499.0, 15),
                ("USB-C Hub", "electronics", 1899.0, 25),
                ("Laptop Stand", "accessories", 1299.0, 30),
                ("Notebook (pack of 3)", "stationery", 249.0, 120),
                ("Desk Lamp", "home", 1599.0, 3),
                ("Water Bottle", "home", 499.0, 0),
            ],
        )

        # 25 orders spread over the last 60 days (fixed seed = same data every reset)
        rng = random.Random(42)
        prices = dict(conn.execute("SELECT id, price FROM products").fetchall())
        now = datetime.now()
        orders = []
        for _ in range(25):
            product_id = rng.randint(1, 6)
            quantity = rng.randint(1, 3)
            subtotal = prices[product_id] * quantity
            total = round(subtotal * (1 + TAX_RATE), 2)
            created = now - timedelta(days=rng.randint(0, 60), hours=rng.randint(0, 23))
            status = rng.choice(["pending", "shipped", "delivered", "delivered", "cancelled"])
            orders.append((rng.randint(1, 6), product_id, quantity, prices[product_id],
                           0.0, total, status, created.strftime("%Y-%m-%d %H:%M:%S")))
        conn.executemany(
            "INSERT INTO orders (customer_id, product_id, quantity, unit_price, discount,"
            " total, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            orders,
        )


# ---------------------------------------------------------------------------
# 2. TOOLS
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int = MAX_TOOL_CHARS):
    """keep tool output small enough for the context window"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"


# ----- READ -----

@tool
def get_schema():
    """Show the database tables and their columns. Call this before writing SQL."""
    with closing(connect(read_only=True)) as conn:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return {"tables": {t["name"]: t["sql"] for t in tables},
            "order_statuses": list(ORDER_STATUSES)}


@tool
def query_database(sql: str):
    """Run a read-only SQL SELECT query on the shop database and return the rows."""
    if not sql.lstrip().lower().startswith(("select", "with")):
        return {"error": "only SELECT queries are allowed; use the write tools to change data"}
    with closing(connect(read_only=True)) as conn:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)
    result = {"rows": rows_to_dicts(rows[:MAX_QUERY_ROWS])}
    if len(rows) > MAX_QUERY_ROWS:
        result["note"] = f"only the first {MAX_QUERY_ROWS} rows are shown; add LIMIT or aggregate"
    return truncate(json.dumps(result, default=str))


# ----- SUMMARIZE -----

@tool
def summarize_sales(start_date: str, end_date: str):
    """Summarize sales between two dates (YYYY-MM-DD, inclusive): revenue, order counts
    by status, top products and top customers. Cancelled orders are excluded from revenue."""
    for d in (start_date, end_date):
        datetime.strptime(d, "%Y-%m-%d")  # raises on a bad date -> error goes back to the model
    period = (start_date, end_date + " 23:59:59")

    with closing(connect(read_only=True)) as conn:
        totals = conn.execute(
            "SELECT COUNT(*) AS orders, COALESCE(ROUND(SUM(total), 2), 0) AS revenue,"
            " COALESCE(ROUND(AVG(total), 2), 0) AS avg_order_value"
            " FROM orders WHERE created_at BETWEEN ? AND ? AND status != 'cancelled'",
            period,
        ).fetchone()
        by_status = conn.execute(
            "SELECT status, COUNT(*) AS orders FROM orders"
            " WHERE created_at BETWEEN ? AND ? GROUP BY status",
            period,
        ).fetchall()
        top_products = conn.execute(
            "SELECT p.name, SUM(o.quantity) AS units, ROUND(SUM(o.total), 2) AS revenue"
            " FROM orders o JOIN products p ON p.id = o.product_id"
            " WHERE o.created_at BETWEEN ? AND ? AND o.status != 'cancelled'"
            " GROUP BY p.id ORDER BY revenue DESC LIMIT 3",
            period,
        ).fetchall()
        top_customers = conn.execute(
            "SELECT c.name, COUNT(*) AS orders, ROUND(SUM(o.total), 2) AS spent"
            " FROM orders o JOIN customers c ON c.id = o.customer_id"
            " WHERE o.created_at BETWEEN ? AND ? AND o.status != 'cancelled'"
            " GROUP BY c.id ORDER BY spent DESC LIMIT 3",
            period,
        ).fetchall()

    return {
        "period": {"from": start_date, "to": end_date},
        **dict(totals),
        "orders_by_status": {r["status"]: r["orders"] for r in by_status},
        "top_products": rows_to_dicts(top_products),
        "top_customers": rows_to_dicts(top_customers),
    }


# ----- COMPUTE -----

def build_quote(conn, product_id: int, quantity: int, discount_code: Optional[str]):
    """business rules shared by quote_order and create_order"""
    if quantity <= 0:
        raise ValueError("quantity must be at least 1")
    product = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    if product is None:
        raise ValueError(f"product {product_id} does not exist")

    rate = 0.0
    if discount_code:
        code = discount_code.strip().upper()
        if code not in DISCOUNT_CODES:
            raise ValueError(f"unknown discount code {discount_code!r}")
        rate = DISCOUNT_CODES[code]

    subtotal = product["price"] * quantity
    discount = round(subtotal * rate, 2)
    tax = round((subtotal - discount) * TAX_RATE, 2)
    return {
        "product_id": product_id,
        "product": product["name"],
        "unit_price": product["price"],
        "quantity": quantity,
        "subtotal": round(subtotal, 2),
        "discount": discount,
        "tax": tax,
        "total": round(subtotal - discount + tax, 2),
        "in_stock": product["stock"],
        "can_fulfil": product["stock"] >= quantity,
    }


@tool
def quote_order(product_id: int, quantity: int, discount_code: Optional[str] = None):
    """Compute the price of an order before placing it: subtotal, discount, 18% tax,
    total, and whether there is enough stock. Does not change anything."""
    with closing(connect(read_only=True)) as conn:
        return build_quote(conn, product_id, quantity, discount_code)


# ----- CALCULATE -----

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCTIONS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}


def _evaluate(node):
    # walk the parsed expression and allow only numbers, maths operators and a few
    # functions — unlike eval(), nothing else can run
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*[_evaluate(a) for a in node.args])
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


@tool
def calculator(expression: str):
    """Evaluate a maths expression exactly, e.g. '(3499 * 2) * 0.9' or 'round(1234.567, 2)'.
    Supports + - * / // % ** and round, abs, min, max, sqrt."""
    result = _evaluate(ast.parse(expression, mode="eval").body)
    return {"expression": expression, "result": result}


# ----- WRITE (approval required) -----

@tool
def create_order(customer_id: int, product_id: int, quantity: int, discount_code: Optional[str] = None):
    """Place a new order: checks stock, applies the discount and tax, reduces stock.
    Requires user approval."""
    with closing(connect()) as conn, conn:  # `with conn` = one transaction, rolled back on error
        if conn.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone() is None:
            raise ValueError(f"customer {customer_id} does not exist")
        quote = build_quote(conn, product_id, quantity, discount_code)
        if not quote["can_fulfil"]:
            raise ValueError(f"not enough stock: {quote['in_stock']} left, {quantity} requested")

        conn.execute("UPDATE products SET stock = stock - ? WHERE id = ?", (quantity, product_id))
        cursor = conn.execute(
            "INSERT INTO orders (customer_id, product_id, quantity, unit_price, discount,"
            " total, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (customer_id, product_id, quantity, quote["unit_price"], quote["discount"],
             quote["total"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
    return {"status": "created", "order_id": cursor.lastrowid,
            **quote, "in_stock": quote["in_stock"] - quantity}


@tool
def update_order_status(order_id: int, status: Literal["pending", "shipped", "delivered", "cancelled"]):
    """Change an order's status. Cancelling puts the items back in stock.
    Requires user approval."""
    with closing(connect()) as conn, conn:
        order = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if order is None:
            raise ValueError(f"order {order_id} does not exist")
        if order["status"] == status:
            return {"status": "unchanged", "order_id": order_id, "order_status": status}
        if order["status"] == "cancelled":
            raise ValueError("a cancelled order cannot be changed")
        if order["status"] == "delivered" and status != "cancelled":
            raise ValueError("a delivered order can only be cancelled (refund)")

        conn.execute("UPDATE orders SET status = ? WHERE id = ?", (status, order_id))
        if status == "cancelled":
            conn.execute("UPDATE products SET stock = stock + ? WHERE id = ?",
                         (order["quantity"], order["product_id"]))
    return {"status": "updated", "order_id": order_id, "from": order["status"], "to": status}


# ----- NOTIFY (approval required) -----

@tool
def send_notification(customer_id: int, message: str, channel: Literal["email", "sms"] = "email"):
    """Send a message to a customer by email or SMS. Requires user approval."""
    with closing(connect()) as conn, conn:
        customer = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if customer is None:
            raise ValueError(f"customer {customer_id} does not exist")
        # practice mode: saved to an "outbox" table instead of really sending.
        # In production a worker would read this table and call an email/SMS API.
        cursor = conn.execute(
            "INSERT INTO notifications (customer_id, channel, message, status, created_at)"
            " VALUES (?, ?, ?, 'queued', ?)",
            (customer_id, channel, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
    return {"status": "queued", "notification_id": cursor.lastrowid,
            "to": customer["email"] if channel == "email" else customer["name"], "channel": channel}


# ----- UTILITY -----

@tool
def get_current_date():
    """Get today's date and weekday (the model does not know it otherwise)"""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A")}


TOOLS = [
    get_schema, query_database, summarize_sales, quote_order, calculator,
    create_order, update_order_status, send_notification, get_current_date,
]


# ---------------------------------------------------------------------------
# 3. STATE
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# 4. NODES
# ---------------------------------------------------------------------------

def make_llm(model: str):
    return ChatGroq(model=model, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES).bind_tools(TOOLS)


llm = make_llm(MODEL).with_fallbacks([make_llm(FALLBACK_MODEL)])


def drop_unanswered_tool_calls(messages: list[AnyMessage]):
    """
    If a turn stopped early (step limit, or a pending approval that was never
    answered), history can hold tool calls that never ran. The API rejects that.
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
    The caller shows the request to the user and resumes with Command(resume="yes"/"no").
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


tool_node = ToolNode(TOOLS, handle_tool_errors=True)


# ---------------------------------------------------------------------------
# 5. EDGES
# ---------------------------------------------------------------------------

def route_after_agent(state: AgentState) -> Literal["approve", "tools", "__end__"]:
    last = state["messages"][-1]
    if not last.tool_calls:
        return END
    if any(tc["name"] in SENSITIVE_TOOLS for tc in last.tool_calls):
        return "approve"  # code decides this, not the LLM
    return "tools"


# ---------------------------------------------------------------------------
# 6. BUILD + COMPILE THE GRAPH
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("approve", approve_node)  # its exits come from the Command[...] type hint
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")

# interrupt() needs a checkpointer: the paused state is saved here until you answer
memory_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(memory_conn))


# ---------------------------------------------------------------------------
# 7. RUN
# ---------------------------------------------------------------------------

def ask_in_terminal(request: dict):
    print(f"\n🔐 {request['question']}")
    for action in request["actions"]:
        print(f"   • {action['tool']}({json.dumps(action['args'], ensure_ascii=False)})")
    try:
        return input("   approve? [y/N] ").strip() or "no"
    except EOFError:
        return "no"  # no one to ask -> deny


def print_updates(stream, step: list):
    for update in stream:
        for node_name, node_output in update.items():
            if node_name == "agent":
                step[0] += 1
                ai_msg = node_output["messages"][-1]
                for tc in ai_msg.tool_calls:
                    print(f"\n🔧 Step {step[0]}: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
                if not ai_msg.tool_calls:
                    print(f"\n✅ FINAL ANSWER:\n{ai_msg.content}")
            elif node_name == "tools":
                for tool_msg in node_output["messages"]:
                    print(f"   → {truncate(str(tool_msg.content), 300)}")
            elif node_name == "approve" and node_output:
                print("   ✋ denied — nothing was changed")


def run_agent(user_task: str, thread_id: str = "default", max_steps: int = 6, approver=ask_in_terminal):
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "agent_run",
        "callbacks": [CallbackHandler()],
        # a step can be agent + approve + tools, plus one for the final answer
        "recursion_limit": 3 * max_steps + 1,
    }
    inputs = {"messages": [HumanMessage(user_task)]}
    step = [0]

    with propagate_attributes(
        trace_name="agent_run",
        session_id=thread_id,
        tags=["shop-assistant", "langgraph-v3"],
    ):
        try:
            while True:
                print_updates(graph.stream(inputs, config, stream_mode="updates"), step)
                state = graph.get_state(config)
                if not state.interrupts:
                    break  # finished
                # paused at approve node -> ask, then resume from the same point
                decision = approver(state.interrupts[0].value)
                inputs = Command(resume=decision)
        except GraphRecursionError:
            print("\n⛔ Stopped — hit max steps without finishing.")
            return "Stopped — hit max steps without finishing."

    last = graph.get_state(config).values["messages"][-1]
    return last.content


def chat(thread_id: str):
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
        run_agent(task, thread_id)


init_db()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph shop assistant (v3)")
    parser.add_argument("--thread", default="default", help="conversation id to start or resume")
    parser.add_argument("-q", "--query", help="ask one question and exit")
    parser.add_argument("--reset-db", action="store_true", help="delete and re-create the sample shop data")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    try:
        if args.reset_db:
            init_db(reset=True)
            print(f"sample data re-created in {SHOP_DB}")
        elif args.graph:
            print(graph.get_graph().draw_mermaid())
        elif args.query:
            run_agent(args.query, args.thread)
        else:
            chat(args.thread)
    finally:
        langfuse.flush()
        memory_conn.close()
