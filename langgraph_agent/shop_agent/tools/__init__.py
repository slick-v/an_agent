"""
All tools the agent can use, grouped by capability.
To add a tool: write it in the right module, then add it to TOOLS below
(and to SENSITIVE_TOOLS if it changes data or contacts someone).
"""
from .calculate import calculator
from .compute import quote_order
from .notify import send_notification
from .read import get_schema, query_database
from .summarize import summarize_sales
from .utility import get_current_date
from .write import create_order, update_order_status

TOOLS = [
    get_schema, query_database,          # read
    summarize_sales,                     # summarize
    quote_order,                         # compute
    calculator,                          # calculate
    create_order, update_order_status,   # write   (approval)
    send_notification,                   # notify  (approval)
    get_current_date,                    # utility
]

# tools that change data or contact people -> must pass the approve node first
SENSITIVE_TOOLS = {"create_order", "update_order_status", "send_notification"}
