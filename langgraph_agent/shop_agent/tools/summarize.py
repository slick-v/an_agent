"""SUMMARIZE tools: business reports built from the data."""
from contextlib import closing
from datetime import datetime

from langchain_core.tools import tool

from ..database import connect, rows_to_dicts


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
