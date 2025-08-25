#!/usr/bin/env python3
"""Minimal coding agent based on ideas from
https://ghuntley.com/agent/

The agent runs a simple perceive-think-act loop powered by a
chat completion model. Tools allow the model to interact with the
local environment. 300 lines of code?  Nah, this example is even
shorter but captures the same spirit.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List

import requests


MEMORY_FILE = os.getenv("AGENT_MEMORY_FILE", "agent_memory.json")


@dataclass
class Tool:
    """A callable that the language model can use."""

    name: str
    description: str
    func: Callable[[str], str]


def run_shell(command: str) -> str:
    """Execute a shell command and return combined output."""
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=60
    )
    return result.stdout + result.stderr


def read_file(path: str) -> str:
    """Return the contents of ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(args: str) -> str:
    """Write content to a file.

    Expected ``args`` format: ``"<path> <content>"``.
    """
    path, _, content = args.partition(" ")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


def ask_user(question: str) -> str:
    """Prompt the human for input and return their response."""
    return input(question + "\n")


def search_files(pattern: str) -> str:
    """Search project files for ``pattern`` using ripgrep."""
    result = subprocess.run(
        ["rg", "--line-number", pattern], capture_output=True, text=True, timeout=60
    )
    if result.returncode == 1:
        return "No matches found."
    return result.stdout + result.stderr


def run_git(args: str) -> str:
    """Run a git command and return its output."""
    return run_shell(f"git {args}")


def run_tests(_: str) -> str:
    """Execute the project's test suite using pytest."""
    return run_shell("pytest -q")


def control_desktop(action: str) -> str:
    """Capture the desktop and perform a simple automation.

    The function saves a screenshot to ``desktop.png`` and performs a couple of
    very small automations using ``pyautogui``. Currently it understands the
    phrases "click start" and "open settings" which, when combined, will open
    the system settings menu on most operating systems.

    Parameters
    ----------
    action: str
        Natural language description of the desired automation.
    """

    import pyautogui  # type: ignore

    screenshot = pyautogui.screenshot()
    path = "desktop.png"
    screenshot.save(path)

    lowered = action.lower()
    if "click start" in lowered:
        width, height = pyautogui.size()
        pyautogui.moveTo(10, height - 10)
        pyautogui.click()
    if "open settings" in lowered:
        pyautogui.write("settings")
        pyautogui.press("enter")
    return f"Screenshot saved to {path}"


class OpenAILLM:
    """Small wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY env var is required. Please set it in your environment.")

    def complete(self, messages: List[Dict[str, str]]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class Agent:
    """A minimal perceive-think-act loop."""

    def __init__(self, tools: List[Tool], llm: OpenAILLM, system_prompt: str) -> None:
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.memory_file = MEMORY_FILE
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    past = json.load(f)
                if isinstance(past, list):
                    self.history.extend(past)
            except json.JSONDecodeError:
                pass

    def run(self, goal: str, max_steps: int = 8) -> str:
        """Attempt to satisfy ``goal`` using the available tools."""
        self.history.append({"role": "user", "content": goal})
        plan_prompt = [
            {"role": "system", "content": "Devise a short plan to accomplish the user's goal."},
            {"role": "user", "content": goal},
        ]
        plan = self.llm.complete(plan_prompt)
        self.history.append({"role": "assistant", "content": f"Plan:\n{plan}"})
        for _ in range(max_steps):
            reply = self.llm.complete(self.history + [self._tool_instructions()])
            try:
                action = json.loads(reply.strip())
            except json.JSONDecodeError:
                self.history.append({"role": "assistant", "content": reply})
                continue
            if action.get("tool") == "finish":
                answer = action.get("answer", "")
                self._save_memory()
                return answer
            tool_name = action.get("tool")
            tool_input = action.get("input", "")
            observation = self._use_tool(tool_name, tool_input)
            self.history.append({"role": "assistant", "content": reply})
            self.history.append({"role": "user", "content": f"Observation: {observation}"})
        self._save_memory()
        return "Reached max steps without finishing"

    def _tool_instructions(self) -> Dict[str, str]:
        tool_desc = "\n".join(
            f"{t.name}: {t.description}" for t in self.tools.values()
        )
        prompt = (
            "You may use the following tools:\n"
            f"{tool_desc}\n"
            "Respond with JSON: {\"tool\": <name>, \"input\": <args>}\n"
            "Use {\"tool\": \"finish\", \"answer\": <text>} to end."
        )
        return {"role": "user", "content": prompt}

    def _use_tool(self, name: str, arg: str) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Unknown tool {name!r}"
        try:
            return tool.func(arg)
        except Exception as exc:  # pragma: no cover - debugging aid
            return f"Tool {name} failed: {exc}"

    def _save_memory(self) -> None:
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.history[-50:], f)
        except OSError:
            pass


def build_default_agent() -> Agent:
    tools = [
        Tool("run", "Execute a shell command", run_shell),
        Tool("read", "Read a file", read_file),
        Tool("write", "Write to a file. Usage: '<path> <content>'", write_file),
        Tool("ask", "Ask the user for input", ask_user),
        Tool("search", "Search files for a pattern", search_files),
        Tool("git", "Run git commands", run_git),
        Tool("test", "Run tests with pytest", run_tests),
        Tool(
            "desktop",
            "Take a screenshot and perform simple desktop automations",
            control_desktop,
        ),
    ]
    llm = OpenAILLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    system_prompt = os.getenv("AGENT_SYSTEM_PROMPT", "You are a helpful coding agent.")
    return Agent(tools, llm, system_prompt)


def main() -> None:
    goal = os.getenv("AGENT_GOAL") or input("Enter your goal: ")
    agent = build_default_agent()
    result = agent.run(goal)
    print(result)


if __name__ == "__main__":
    main()
