import os
import json
from dotenv import load_dotenv
from groq import Groq

from langfuse import get_client, observe, propagate_attributes


load_dotenv()


MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
langfuse = get_client()  # reads LANGFUSE_* from .env
NOTES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes")
os.makedirs(NOTES_DIR, exist_ok=True)

# models = client.models.list()
# for m in models.data:
#     print(m.id)


def safe_path(filename: str):
    """resolve filename inside NOTES_DIR, rejecting anything that escapes it"""
    # tolerate the model prefixing "notes/" itself
    name = filename.replace("\\", "/")
    if name.startswith("notes/"):
        name = name[len("notes/"):]
    path = os.path.realpath(os.path.join(NOTES_DIR, name))
    if os.path.commonpath([path, os.path.realpath(NOTES_DIR)]) != os.path.realpath(NOTES_DIR):
        raise ValueError(f"{filename} is outside the notes folder")
    return path


# tool implementation
@observe(as_type="tool", name="search_files")
def search_files(query:str):
    """search filename + coontent in the notes folder for a keyword"""

    results = []
    for fname in os.listdir(NOTES_DIR):
        path = os.path.join(NOTES_DIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r",encoding="utf-8") as f:
            content = f.read()
        if query.lower() in fname.lower() or query.lower() in content.lower():
            results.append(fname)

    return {"matches": results} if results else {"matches": [], "note": "no files found"}

@observe(as_type="tool", name="read_file")
def read_file(filename:str):
    path = safe_path(filename)
    if not os.path.exists(path):
        return {"error": f"{filename} does not exist"}
    with open(path, "r", encoding="utf-8") as f:
        return {"content": f.read()}

    

@observe(as_type="tool", name="write_file")
def write_file(filename: str, content: str, mode: str = "overwrite"):
    path = safe_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_mode = "a" if mode == "append" else "w"
    with open(path, file_mode, encoding="utf-8") as f:
        f.write(content)
    return {"status": "success", "filename": filename, "mode": mode}

@observe(as_type="tool", name="web_search")
def web_search(query: str):
    # Stub — plug in a real search API (Tavily, SerpAPI, Bing) here.
    # Keeping it stubbed so the loop logic is the focus.
    return {"results": f"[stub] Would search the web for: {query}"}


# toool schemassss
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search local notes for a keyword in filename or content",
        "parameters": {
            "type": "object",
            "properties":{"query":{"type":"string"}},
            "required": ["query"],
        },
    },
},
{
    "type":"function",
    "function": {
        "name": "read_file",
        "description": "Read the full content of a specific note file",
        "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
        },
    },
},

{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or append content to a note file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["overwrite", "append"]},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information not in local notes",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {
    "search_files": search_files,
    "read_file": read_file,
    "write_file": write_file,
    "web_search": web_search,
}

# the agent loop

@observe(name="agent_run", as_type="agent")
def run_agent(user_task:str, max_steps:int = 8):

    messages = [

        {
            "role": "system",
            "content": (
                "You are a knowledge assistant that reads and updates local notes. "
                "Always search/read existing notes before writing, so you don't "
                "overwrite useful content blindly. Explain your plan briefly before acting. "
                "Filenames are relative to the notes folder, e.g. 'agentic-ai.md' (no 'notes/' prefix)."
            ),
        },
        {"role": "user", "content": user_task},

    ]

    for step in range(max_steps):
        # log each LLM call as its own "generation" nested under agent_run
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=f"llm_call_step_{step+1}",
            model=MODEL,
            input=list(messages),
        ) as generation:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            generation.update(
                output=response.choices[0].message.content or str(response.choices[0].message.tool_calls),
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
            )
        msg = response.choices[0].message
        messages.append(msg)


         # No tool call = model thinks it's done
        if not msg.tool_calls:
            print(f"\n✅ FINAL ANSWER:\n{msg.content}")
            return msg.content

        # Execute each requested tool call
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            # send failures back to the model so it can correct itself
            try:
                fn_args = json.loads(tool_call.function.arguments or "{}")
                print(f"\n🔧 Step {step+1}: calling {fn_name}({fn_args})")
                if fn_name not in TOOL_MAP:
                    raise ValueError(f"unknown tool: {fn_name}")
                result = TOOL_MAP[fn_name](**fn_args)

            except Exception as e:
                print(f"\n🔧 Step {step+1}: {fn_name} failed")
                result = {"error": f"{type(e).__name__}: {e}"}
            print(f"   → result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": json.dumps(result),
            })
    return "Stopped — hit max steps without finishing."


if __name__ == "__main__":
    # tags must wrap the traced call so they apply to the whole trace
    with propagate_attributes(tags=["knowledge-assistant"]):
        run_agent(
            "Search my notes for anything about 'agentic AI'. If a file exists, "
            "read it. Then create or update agentic-ai.md with a short "
            "summary of what's there, adding a section on ReAct pattern if missing."
        )

    langfuse.flush()   # makes sure all traces are sent before the script exits