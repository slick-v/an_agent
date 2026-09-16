"""COMPUTE tools: business calculations with the shop's pricing rules."""
from contextlib import closing
from typing import Optional

from langchain_core.tools import tool

from ..config import DISCOUNT_CODES, TAX_RATE
from ..database import connect


def build_quote(conn, product_id: int, quantity: int, discount_code: Optional[str]):
    """pricing rules shared by quote_order and create_order"""
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
