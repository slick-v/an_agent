"""
Shop assistant — LangGraph agent on a SQLite practice database (v3).

  config.py     settings (paths, model, limits, business rules, tracing tags)
  prompts.py    system prompt
  database.py   SQLite connection, tables, sample data
  tools/        one module per capability: read, summarize, compute,
                calculate, write, notify, utility
  graph.py      LangGraph: state, nodes (agent, approve, tools), edges, build_graph()
  tracing.py    Langfuse: one trace per user message + callback handler
  runner.py     run_agent(): streams the graph and handles approval pauses
  __main__.py   terminal chat / CLI  ->  python -m langgraph_agent.shop_agent
"""
