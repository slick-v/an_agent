"""READ tools: look at the database without changing it."""
import json
from contextlib import closing

from langchain_core.tools import tool

from ..config import MAX_QUERY_ROWS, ORDER_STATUSES
from ..database import connect, rows_to_dicts
from ..utils import truncate


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
    # the read-only connection is the real guard; the check above just gives a clearer message
    with closing(connect(read_only=True)) as conn:
        rows = conn.execute(sql).fetchmany(MAX_QUERY_ROWS + 1)
    result = {"rows": rows_to_dicts(rows[:MAX_QUERY_ROWS])}
    if len(rows) > MAX_QUERY_ROWS:
        result["note"] = f"only the first {MAX_QUERY_ROWS} rows are shown; add LIMIT or aggregate"
    return truncate(json.dumps(result, default=str))
