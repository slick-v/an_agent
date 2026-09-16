# Agentic AI

## Brief Summary

- **Agentic AI** are autonomous systems that pursue explicit goals using perception, planning, reasoning, and learning, with a focus on safety and alignment.
- This note outlines core aspects such as goal‑oriented behavior, world modeling, adaptive learning, and safety considerations.
- It also details the **ReAct (Reason+Act) pattern**, which interleaves reasoning steps with tool‑based actions, providing transparency, modularity, and robust error handling. The pattern’s workflow, benefits, and an example are included, noting its adoption in AutoGPT, BabyAGI, and LangChain agents.

**Short Summary:**

- Agentic AI are autonomous systems that pursue explicit goals using perception, planning, reasoning, and learning.
- Core aspects include goal‑oriented behavior, world modeling, adaptive learning, and alignment safety.
- The note details the **ReAct (Reason+Act) pattern**, which interleaves reasoning steps with tool‑based actions, providing transparency, modularity, and robust error handling.
- Includes an illustrative example and mentions adoption in AutoGPT, BabyAGI, and LangChain agents.

*This note provides an overview of agentic AI and details the ReAct pattern for building such agents.*

# Agentic AI

*This note currently contains a brief overview of the concept of agentic AI, summarizing existing content and adding new information about the ReAct pattern.*

## Summary

Agentic AI refers to artificial intelligence systems designed to act autonomously toward goals, often incorporating reasoning, planning, and decision‑making capabilities. Such agents can perceive their environment, maintain internal state, and execute actions that influence outcomes without direct human intervention. Key aspects include:

- **Goal‑oriented behavior**: Agents have explicit objectives or utility functions they strive to maximize.
- **Perception and world model**: They process inputs (e.g., language, sensor data) to build an internal representation of the environment.
- **Planning and reasoning**: Using models of cause and effect, agents generate plans or sequences of actions.
- **Learning and adaptation**: Many agents improve over time via reinforcement learning, meta‑learning, or continual learning.
- **Safety and alignment**: Ensuring that autonomous actions remain aligned with human values is a central research focus.

## ReAct Pattern

*If not already present, this section introduces the ReAct (Reason+Act) pattern, a framework for building agentic AI that interleaves reasoning steps with actions.*

The ReAct pattern combines **reasoning** (thought) and **action** (execution) in a loop, allowing language models or other reasoning components to decide when to act and what action to take. The typical workflow is:

1. **Prompt** the model with a task description.
2. The model **generates a reasoning step** (e.g., "I need to look up the latest price of Bitcoin").
3. The model **outputs an action command** (e.g., `search["latest Bitcoin price"]`).
4. The system **executes the action**, retrieves the result, and feeds it back to the model.
5. The model **continues reasoning** with the new information, possibly generating further actions, until it arrives at a final answer.

### Benefits
- **Transparency**: Each reasoning step is explicit, making the agent’s thought process observable.
- **Modularity**: Actions can be defined as separate tools (search, calculator, database query), enabling easy extension.
- **Error handling**: If an action fails, the model can reason about the failure and retry or choose an alternative.

### Example
```
User: What is the current population of Tokyo and how does it compare to New York?

Assistant:
Thought: I need the latest population figures for both cities.
Action: search["Tokyo population 2024"]
Observation: Tokyo population is 14 million.
Thought: Now I need New York's population.
Action: search["New York City population 2024"]
Observation: New York City population is 8.5 million.
Thought: Compare the two.
Answer: Tokyo has about 5.5 million more residents than New York City.
```

The ReAct pattern is widely used in recent agentic AI systems such as AutoGPT, BabyAGI, and various LangChain agents.

---
*Last updated: 2026-09-16*

## Quick Summary

- **Agentic AI**: Autonomous AI systems with goals, perception, planning, learning, and safety considerations.
- **ReAct pattern**: Interleaves reasoning (thought) and actions (tool calls) in a loop, enabling transparent, modular, and error‑resilient agents.

---
*Summary added on 2026-09-16*

## Overview

This note provides a concise overview of **Agentic AI**, describing its core characteristics such as goal‑oriented autonomous behavior, perception, planning, learning, and safety alignment. It also includes a detailed explanation of the **ReAct (Reason+Act) pattern**, which interleaves reasoning steps with tool‑based actions to build transparent and modular AI agents. The ReAct section covers its workflow, benefits, and a concrete example, and notes its adoption in systems like AutoGPT and BabyAGI.
