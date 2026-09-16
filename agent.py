import os
import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv() 



client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
NOTES_DIR = "notes"

# models = client.models.list()
# for m in models.data:
#     print(m.id)

# tool implementation
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

def read_file(filename:str):
    path = os.path.join(NOTES_DIR,filename)
    if not os.path.exists(path):
        return {"error": f"{filename} does not exist"}
    with open(path, "r", encoding="utf-8") as f:
        return {"content": f.read()}

def write_file(filename: str, content: str, mode: str = "overwrite"):
    path = os.path.join(NOTES_DIR, filename)
    file_mode = "a" if mode == "append" else "w"
    with open(path, file_mode, encoding="utf-8") as f:
        f.write(content)
    return {"status": "success", "filename": filename, "mode": mode}

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


def run_agent(user_task:str, max_steps:int = 8):
    messages = [

        {
            "role": "system",
            "content": (
                "You are a knowledge assistant that reads and updates local notes. "
                "Always search/read existing notes before writing, so you don't "
                "overwrite useful content blindly. Explain your plan briefly before acting."
            ),
        },
        {"role": "user", "content": user_task},

    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
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
            fn_args = json.loads(tool_call.function.arguments)
            print(f"\n🔧 Step {step+1}: calling {fn_name}({fn_args})")

            result = TOOL_MAP[fn_name](**fn_args)
            print(f"   → result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": json.dumps(result),
            })

    return "Stopped — hit max steps without finishing."


if __name__ == "__main__":
    run_agent(
        "Search my notes for anything about 'agentic AI'. If a file exists, "
        "read it. Then create or update notes/agentic-ai.md with a short "
        "summary of what's there, adding a section on ReAct pattern if missing."
    )