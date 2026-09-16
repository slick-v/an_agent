"""WRITE tools: change orders and stock. These always pass the approve node first."""
from contextlib import closing
from datetime import datetime
from typing import Literal, Optional

from langchain_core.tools import tool

from ..database import connect
from .compute import build_quote


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
