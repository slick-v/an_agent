"""All settings in one place."""
import os

from dotenv import load_dotenv

load_dotenv()  # finds an_agent/.env (searches parent folders)

# an_agent/  (this file is an_agent/langgraph_agent/shop_agent/config.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")  # git-ignored
os.makedirs(DATA_DIR, exist_ok=True)

SHOP_DB = os.path.join(DATA_DIR, "shop.db")                   # the practice business data
CHECKPOINT_DB = os.path.join(DATA_DIR, "checkpoints.sqlite")  # the agent's memory

# --- LLM ---
MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.3-70b-versatile"  # used only if MODEL keeps failing
LLM_TIMEOUT = 60           # seconds per LLM request
LLM_MAX_RETRIES = 3        # retries on timeouts / rate limits / 5xx

# --- limits ---
MAX_CONTEXT_TOKENS = 6000  # history sent to the LLM is trimmed to roughly this size
MAX_TOOL_CHARS = 8000      # longer tool output is cut off
MAX_QUERY_ROWS = 50        # query_database never returns more rows than this
MAX_STEPS = 6              # agent steps per user message

# --- business rules ---
TAX_RATE = 0.18
DISCOUNT_CODES = {"SAVE10": 0.10, "WELCOME5": 0.05}
ORDER_STATUSES = ("pending", "shipped", "delivered", "cancelled")

# --- tracing ---
TRACE_NAME = "agent_run"
TRACE_TAGS = ["shop-assistant", "langgraph-v3"]
