"""NOTIFY tools: contact customers. These always pass the approve node first."""
from contextlib import closing
from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

from ..database import connect


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
