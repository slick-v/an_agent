"""UTILITY tools."""
from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_date():
    """Get today's date and weekday (the model does not know it otherwise)"""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A")}
