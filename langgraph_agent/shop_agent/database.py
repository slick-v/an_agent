"""The practice shop database: connections, tables and sample data.

  customers(id, name, email, city)
  products(id, name, category, price, stock)
  orders(id, customer_id, product_id, quantity, unit_price, discount, total, status, created_at)
  notifications(id, customer_id, channel, message, status, created_at)
"""
import os
import random
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta

from .config import SHOP_DB, TAX_RATE


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
