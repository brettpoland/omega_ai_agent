# Minimal Coding Agent

This repository contains a simple coding agent inspired by the article
["how to build a coding agent"](https://ghuntley.com/agent/).

The agent is roughly 150 lines of Python and demonstrates the core idea
from the article: a loop that repeatedly plans, acts using tools, and
observes the results.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
export OPENAI_API_KEY=sk-...
python agent.py
```

When run without an `AGENT_GOAL` environment variable, the agent
asks you for a goal interactively. During execution it may call the
`ask` tool to request additional input from you.

You can customise the model with `OPENAI_MODEL`.
