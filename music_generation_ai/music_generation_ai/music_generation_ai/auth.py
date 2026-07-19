"""
auth.py
-------
Minimal username/password authentication backed by a local SQLite database
(users.db, created automatically on first run). No external services needed.
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

import config

DB_PATH = os.path.join(config.BASE_DIR, "users.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_user(username, password):
    """Returns (success: bool, message: str)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password))
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


def verify_user(username, password):
    """Returns True if the username/password combo is valid."""
    conn = get_db()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?", (username.strip(),)
    ).fetchone()
    conn.close()
    if row is None:
        return False
    return check_password_hash(row["password_hash"], password)
