# Agentic AI

## Short Summary
Agentic AI refers to autonomous systems that pursue defined goals by planning, reasoning, maintaining internal state, and interacting with external tools or APIs. They differ from purely reactive models by incorporating self‑reflection loops that allow them to evaluate progress and adapt actions.

## ReAct Pattern
The **ReAct** (Reason + Act) pattern structures agentic AI by alternating reasoning steps with concrete actions and observations. This loop enables transparent, reliable execution of complex tasks such as web browsing, tool use, or multi‑step problem solving.

**Typical ReAct flow:**
1. *Thought*: generate a natural‑language reasoning step.
2. *Action*: invoke a tool, API, or code execution.
3. *Observation*: ingest the result and feed it back into the next reasoning step.

*Example:*
```
Thought: I need the current weather for Paris.
Action: call_weather_api(location="Paris")
Observation: {"temp": 18, "condition": "cloudy"}
Thought: The weather is mild; suggest a light jacket.
Answer: It's 18°C and cloudy in Paris. A light jacket should be comfortable.
```

The ReAct pattern is widely adopted in modern LLM‑based agents (e.g., OpenAI function calling, LangChain agents) to provide clear reasoning traces and improve task reliability.

*This file condenses the earlier detailed notes on Agentic AI while preserving the essential ReAct pattern description.*