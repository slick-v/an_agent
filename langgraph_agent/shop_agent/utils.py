"""Small helpers shared by several modules."""
from .config import MAX_TOOL_CHARS


def truncate(text: str, limit: int = MAX_TOOL_CHARS):
    """keep text small enough for the context window / terminal"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"
