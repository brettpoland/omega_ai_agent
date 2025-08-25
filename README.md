# Minimal Coding Agent

This repository contains a simple coding agent inspired by the article
["how to build a coding agent"](https://ghuntley.com/agent/).

The agent is roughly 150 lines of Python and demonstrates the core idea
from the article: a loop that repeatedly plans, acts using tools, and
observes the results. It now ships with additional tools for file search,
git commands and running tests, a simple planning step and persistent
memory between sessions.

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

### Tools

* `run` – execute shell commands
* `read` / `write` – file operations
* `ask` – interactively query the user
* `search` – search project files with ripgrep
* `git` – run git commands
* `test` – execute `pytest`

### Configuration

* `OPENAI_MODEL` – choose the model
* `AGENT_SYSTEM_PROMPT` – customise the agent persona
* `AGENT_MEMORY_FILE` – path to store conversation memory

You can customise the model with `OPENAI_MODEL` and the system prompt
with `AGENT_SYSTEM_PROMPT`.
