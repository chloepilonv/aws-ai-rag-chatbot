"""
SQLite logging module for chatbot conversations and feedback.
Stores all Q&A interactions for analysis and improvement.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Database path - configurable via environment variable
DB_PATH = os.getenv("DB_PATH", "/data/feedback.db")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_db():
    """Initialize the database schema if it doesn't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Conversations table - stores all Q&A interactions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                user_id TEXT,
                feedback INTEGER DEFAULT NULL,
                feedback_comment TEXT,
                response_time_ms INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON conversations(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback
            ON conversations(feedback)
        """)


def log_conversation(
    question: str,
    answer: str,
    sources: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    response_time_ms: Optional[int] = None
) -> int:
    """
    Log a conversation to the database.

    Args:
        question: User's question
        answer: Bot's answer
        sources: List of source URLs used
        user_id: Optional user identifier
        response_time_ms: Response time in milliseconds

    Returns:
        conversation_id: The ID of the inserted record
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Convert sources list to JSON string
        sources_json = json.dumps(sources) if sources else None

        cursor.execute("""
            INSERT INTO conversations
            (timestamp, question, answer, sources, user_id, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            question,
            answer,
            sources_json,
            user_id,
            response_time_ms
        ))

        return cursor.lastrowid


def add_feedback(conversation_id: int, feedback: int, comment: Optional[str] = None):
    """
    Add user feedback to a conversation.

    Args:
        conversation_id: ID of the conversation
        feedback: 1 for thumbs up, -1 for thumbs down
        comment: Optional text comment
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations
            SET feedback = ?, feedback_comment = ?
            WHERE id = ?
        """, (feedback, comment, conversation_id))


def get_conversations(
    limit: int = 100,
    offset: int = 0,
    feedback_only: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve conversations from the database.

    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        feedback_only: Filter by feedback (1 for positive, -1 for negative)

    Returns:
        List of conversation dictionaries
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM conversations"
        params = []

        if feedback_only is not None:
            query += " WHERE feedback = ?"
            params.append(feedback_only)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)

        # Convert rows to dictionaries
        rows = cursor.fetchall()
        conversations = []
        for row in rows:
            conv = dict(row)
            # Parse sources JSON back to list
            if conv['sources']:
                conv['sources'] = json.loads(conv['sources'])
            conversations.append(conv)

        return conversations


def get_stats() -> Dict[str, Any]:
    """
    Get statistics about conversations.

    Returns:
        Dictionary with stats (total, positive feedback, negative feedback, etc.)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total = cursor.fetchone()[0]

        # Positive feedback
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE feedback = 1")
        positive = cursor.fetchone()[0]

        # Negative feedback
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE feedback = -1")
        negative = cursor.fetchone()[0]

        # Average response time
        cursor.execute("SELECT AVG(response_time_ms) FROM conversations WHERE response_time_ms IS NOT NULL")
        avg_response_time = cursor.fetchone()[0]

        return {
            "total_conversations": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "no_feedback": total - positive - negative,
            "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else None
        }


# Initialize database on module import
try:
    init_db()
    print(f"[INFO] Database initialized at {DB_PATH}")
except Exception as e:
    print(f"[WARNING] Failed to initialize database: {e}")
