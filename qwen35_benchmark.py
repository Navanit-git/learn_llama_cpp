#!/usr/bin/env python3
"""
Qwen3.5-4B Model Capability Benchmark
======================================
Tests a llama.cpp server (OpenAI-compatible API) across multiple capability
categories and writes results to a timestamped CSV + summary.

Categories tested:
  1. Basic chat / instruction following
  2. Reasoning & logic
  3. Code generation
  4. Summarisation
  5. Structured output (JSON)
  6. Math
  7. Multi-turn conversation
  8. Tool / function calling readiness
  9. Long-context handling
 10. Creativity
 11. Agentic: single tool call (OpenAI function-calling format)
 12. Agentic: parallel tool calls
 13. Agentic: tool-result loop (multi-step)
 14. Agentic: complex planning & decomposition
 15. Agentic: error handling & recovery
 16. Agentic: constraint satisfaction

Usage:
  python3 qwen35_benchmark.py                     # defaults
  SERVER_URL=http://host:port python3 qwen35_benchmark.py
  python3 qwen35_benchmark.py --out-dir ./my_results --model qwen35
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("Installing requests …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ── Configuration ────────────────────────────────────────────────────────────

CFG = {
    "server_url": os.environ.get("SERVER_URL", "http://172.31.16.1:1234"),
    "model": os.environ.get("BENCH_MODEL", "qwen3.5-4b@q5_k_s"),
    "timeout": int(os.environ.get("BENCH_TIMEOUT", "120")),
    "native_tools": os.environ.get("BENCH_NATIVE_TOOLS", "0").strip().lower() in {"1", "true", "yes", "on"},
}

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    category: str
    name: str
    messages: list[dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.1          # low temp for reproducibility
    check: str = ""                   # callable name in CHECKS dict
    check_args: dict = field(default_factory=dict)
    json_schema: dict | None = None   # if set, request structured output
    tools: list[dict] | None = None   # OpenAI-format tool definitions
    tool_choice: str | dict | None = None  # "auto", "required", or specific


@dataclass
class TestResult:
    id: str
    category: str
    name: str
    passed: bool
    score: float           # 0.0 – 1.0
    reason: str
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    tokens_per_sec: float
    error: str = ""


def _to_serializable(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable values for debug logs."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return str(value)


def append_debug_record(debug_path: Path | None, record: dict[str, Any]) -> None:
    """Append one debug record to a pretty-printed JSON array file."""
    if not debug_path:
        return

    serializable_record = _to_serializable(record)
    existing: list[Any] = []

    if debug_path.exists():
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(serializable_record)

    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ── Checking / scoring helpers ───────────────────────────────────────────────

def check_contains_any(text: str, *, keywords: list[str], **_kw) -> tuple[bool, float, str]:
    """Pass if response contains at least one keyword (case-insensitive)."""
    lower = text.lower()
    found = [k for k in keywords if k.lower() in lower]
    if found:
        return True, 1.0, f"found: {found}"
    return False, 0.0, f"none of {keywords} found"


def check_contains_all(text: str, *, keywords: list[str], **_kw) -> tuple[bool, float, str]:
    """Pass if response contains ALL keywords."""
    lower = text.lower()
    found = [k for k in keywords if k.lower() in lower]
    missing = [k for k in keywords if k.lower() not in lower]
    score = len(found) / len(keywords) if keywords else 0.0
    if not missing:
        return True, 1.0, "all keywords present"
    return score >= 0.5, score, f"missing: {missing}"


def check_valid_json(text: str, *, keys: list[str] | None = None, **_kw) -> tuple[bool, float, str]:
    """Pass if response is valid JSON (optionally with required keys)."""
    # Try to extract JSON from markdown code blocks or raw text
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    raw = json_match.group(1).strip() if json_match else text.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return False, 0.0, "invalid JSON"
    if keys:
        if isinstance(obj, dict):
            present = [k for k in keys if k in obj]
            missing = [k for k in keys if k not in obj]
            score = len(present) / len(keys)
            if missing:
                return score >= 0.5, score, f"missing keys: {missing}"
        elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            present = [k for k in keys if k in obj[0]]
            missing = [k for k in keys if k not in obj[0]]
            score = len(present) / len(keys)
            if missing:
                return score >= 0.5, score, f"missing keys in first item: {missing}"
    return True, 1.0, "valid JSON"


def check_code_block(text: str, *, lang: str = "", keywords: list[str] | None = None, **_kw) -> tuple[bool, float, str]:
    """Pass if response contains a code block, optionally with keywords."""
    has_block = "```" in text or "def " in text or "function " in text or "class " in text
    if not has_block:
        return False, 0.0, "no code block found"
    if keywords:
        lower = text.lower()
        found = [k for k in keywords if k.lower() in lower]
        score = len(found) / len(keywords)
        if score < 0.5:
            return False, score, f"code present but missing: {[k for k in keywords if k.lower() not in lower]}"
        return True, score, f"code with keywords: {found}"
    return True, 1.0, "code block present"


def check_math_answer(text: str, *, answer: str, **_kw) -> tuple[bool, float, str]:
    """Pass if the expected numeric answer appears in the response."""
    # Normalize spaces and look for the answer
    normalized = re.sub(r"[,\s]+", "", text)
    if answer.replace(",", "") in normalized:
        return True, 1.0, f"correct answer {answer} found"
    return False, 0.0, f"expected {answer}, not found in response"


def check_not_empty(text: str, *, min_words: int = 10, **_kw) -> tuple[bool, float, str]:
    """Pass if response has at least min_words words."""
    words = len(text.split())
    if words >= min_words:
        return True, 1.0, f"{words} words (>= {min_words})"
    return False, words / min_words, f"only {words} words (need {min_words})"


def check_multi_turn(text: str, *, keywords: list[str], **_kw) -> tuple[bool, float, str]:
    """For multi-turn: check the model remembers context from earlier turns."""
    return check_contains_any(text, keywords=keywords)


def check_tool_call(text: str, *, tool_name: str = "", arg_keys: list[str] | None = None,
                    _tool_calls: list[dict] | None = None, **_kw) -> tuple[bool, float, str]:
    """Pass if the response contains a tool call with the expected function name and argument keys.
    Checks both structured tool_calls (from API) and text-based fallback."""
    calls = _tool_calls or []
    if calls:
        names = [c.get("function", {}).get("name", "") for c in calls]
        if tool_name and tool_name not in names:
            return False, 0.3, f"tool calls present ({names}) but expected '{tool_name}'"
        # Check argument keys if requested
        if arg_keys:
            for c in calls:
                try:
                    args = json.loads(c["function"]["arguments"]) if isinstance(c["function"]["arguments"], str) else c["function"]["arguments"]
                except (json.JSONDecodeError, KeyError):
                    args = {}
                found = [k for k in arg_keys if k in args]
                if len(found) == len(arg_keys):
                    return True, 1.0, f"tool '{c['function']['name']}' with args {found}"
            return True, 0.7, f"tool call present but missing some arg keys: {arg_keys}"
        return True, 1.0, f"tool call(s): {names}"
    # Fallback: check if text contains tool-call-like JSON
    if tool_name and tool_name.lower() in text.lower():
        return True, 0.7, f"tool name '{tool_name}' found in text (no structured tool_calls)"
    return False, 0.0, "no tool calls found"


def check_parallel_tool_calls(text: str, *, expected_tools: list[str],
                              _tool_calls: list[dict] | None = None, **_kw) -> tuple[bool, float, str]:
    """Pass if multiple tool calls are emitted in a single response."""
    calls = _tool_calls or []
    if len(calls) < 2:
        # Fallback: check text for multiple function references
        lower = text.lower()
        found = [t for t in expected_tools if t.lower() in lower]
        if len(found) >= 2:
            return True, 0.6, f"found {len(found)} tool refs in text: {found} (not structured)"
        return False, len(found) / max(len(expected_tools), 1), \
            f"expected {len(expected_tools)} parallel calls, got {len(calls)} structured + {len(found)} text refs"
    names = [c.get("function", {}).get("name", "") for c in calls]
    found = [t for t in expected_tools if t in names]
    score = len(found) / len(expected_tools)
    if score >= 0.8:
        return True, score, f"parallel calls: {names}"
    return False, score, f"expected tools {expected_tools}, got {names}"


def check_agent_plan(text: str, *, min_steps: int = 3, required_keywords: list[str] | None = None,
                     **_kw) -> tuple[bool, float, str]:
    """Pass if response contains a numbered/bulleted multi-step plan."""
    # Count numbered steps or bullet points
    steps = re.findall(r'(?:^|\n)\s*(?:\d+[.)\-]|[-*•])\s+\S', text)
    step_count = len(steps)
    score_steps = min(step_count / min_steps, 1.0)

    score_kw = 1.0
    kw_msg = ""
    if required_keywords:
        lower = text.lower()
        found = [k for k in required_keywords if k.lower() in lower]
        score_kw = len(found) / len(required_keywords)
        kw_msg = f", keywords {len(found)}/{len(required_keywords)}"

    score = (score_steps * 0.6) + (score_kw * 0.4)
    passed = step_count >= min_steps and score_kw >= 0.5
    return passed, round(score, 2), f"{step_count} steps (need {min_steps}){kw_msg}"


def check_error_recovery(text: str, *, error_keywords: list[str], fix_keywords: list[str],
                         **_kw) -> tuple[bool, float, str]:
    """Pass if model identifies the error AND proposes a fix."""
    lower = text.lower()
    err_found = [k for k in error_keywords if k.lower() in lower]
    fix_found = [k for k in fix_keywords if k.lower() in lower]

    # Semantic recovery signals to reduce brittle exact-keyword dependence.
    semantic_fix_patterns = [
        r"\bcorrect(?:ed|ion)?\b",
        r"\bclean(?:ing)?\b",
        r"\bfix(?:ed|ing)?\b",
        r"\bhandle\b",
        r"\breplace\b",
        r"\bremove\b",
        r"\bimput(?:e|ation)\b",
        r"\bfill(?: in)?\b",
        r"\bvalidate\b",
        r"\bconvert\b",
        r"\bstandardiz(?:e|ation)\b",
        r"\bask for clarif(?:ication|y)\b",
        r"\bprovide (?:the )?correct(?:ed)? (?:data|values?)\b",
    ]
    semantic_fix_count = sum(1 for pat in semantic_fix_patterns if re.search(pat, lower))

    err_score = len(err_found) / max(len(error_keywords), 1)
    fix_score_lexical = len(fix_found) / max(len(fix_keywords), 1)
    fix_score_semantic = min(semantic_fix_count / 4, 1.0)
    fix_score = max(fix_score_lexical, fix_score_semantic)
    score = (err_score * 0.4) + (fix_score * 0.6)
    passed = err_score >= 0.5 and fix_score >= 0.5
    return passed, round(score, 2), (
        f"errors: {err_found}, fixes: {fix_found}, semantic_fix_hits: {semantic_fix_count}"
    )


def check_constraint_satisfaction(text: str, *, constraints: list[dict], **_kw) -> tuple[bool, float, str]:
    """Check multiple constraints. Each: {"type": "contains"|"not_contains"|"max_words"|"min_words", ...}"""
    met = 0
    details = []
    for c in constraints:
        ctype = c["type"]
        if ctype == "contains":
            # Treat common bullet markers as equivalent for robustness.
            if c["value"] == "•":
                has_bullet = bool(re.search(r"(?:^|\n)\s*[-*•]\s+", text))
                if has_bullet:
                    met += 1
                    details.append("✓ contains bullet list")
                else:
                    details.append("✗ missing bullet list")
            elif c["value"].lower() in text.lower():
                met += 1
                details.append(f"✓ contains '{c['value']}'")
            else:
                details.append(f"✗ missing '{c['value']}'")
        elif ctype == "not_contains":
            if c["value"].lower() not in text.lower():
                met += 1
                details.append(f"✓ avoids '{c['value']}'")
            else:
                details.append(f"✗ contains forbidden '{c['value']}'")
        elif ctype == "max_words":
            wc = len(text.split())
            if wc <= c["value"]:
                met += 1
                details.append(f"✓ {wc} words <= {c['value']}")
            else:
                details.append(f"✗ {wc} words > {c['value']}")
        elif ctype == "min_words":
            wc = len(text.split())
            if wc >= c["value"]:
                met += 1
                details.append(f"✓ {wc} words >= {c['value']}")
            else:
                details.append(f"✗ {wc} words < {c['value']}")
    score = met / max(len(constraints), 1)
    passed = score >= 0.7
    return passed, round(score, 2), "; ".join(details)


CHECKS = {
    "contains_any": check_contains_any,
    "contains_all": check_contains_all,
    "valid_json": check_valid_json,
    "code_block": check_code_block,
    "math_answer": check_math_answer,
    "not_empty": check_not_empty,
    "multi_turn": check_multi_turn,
    "tool_call": check_tool_call,
    "parallel_tool_calls": check_parallel_tool_calls,
    "agent_plan": check_agent_plan,
    "error_recovery": check_error_recovery,
    "constraint_satisfaction": check_constraint_satisfaction,
}


# ── Test case definitions ────────────────────────────────────────────────────

TEST_CASES: list[TestCase] = [
    # ── 1. Basic chat ────────────────────────────────────────────────────
    TestCase(
        id="basic_01", category="basic_chat", name="simple greeting",
        messages=[{"role": "user", "content": "Say hello and introduce yourself in 2 sentences."}],
        max_tokens=128, check="not_empty", check_args={"min_words": 5},
    ),
    TestCase(
        id="basic_02", category="basic_chat", name="instruction following",
        messages=[
            {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
            {"role": "user", "content": "What is the weather like today?"},
        ],
        max_tokens=128, check="contains_any", check_args={"keywords": ["arr", "matey", "ye", "sail", "ahoy", "seas", "ship", "treasure", "aye", "cap"]},
    ),

    # ── 2. Reasoning & logic ─────────────────────────────────────────────
    TestCase(
        id="reason_01", category="reasoning", name="logical deduction",
        messages=[{"role": "user", "content": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain your reasoning step by step."}],
        max_tokens=300, check="contains_any", check_args={"keywords": ["cannot conclude", "no", "not necessarily", "does not follow", "invalid"]},
    ),
    TestCase(
        id="reason_02", category="reasoning", name="word puzzle",
        messages=[{"role": "user", "content": "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?"}],
        max_tokens=128, check="contains_any", check_args={"keywords": ["echo"]},
    ),

    # ── 3. Code generation ───────────────────────────────────────────────
    TestCase(
        id="code_01", category="code_generation", name="python function",
        messages=[{"role": "user", "content": "Write a Python function called `fibonacci(n)` that returns the n-th Fibonacci number using iteration (not recursion). Include a docstring."}],
        max_tokens=400, check="code_block", check_args={"keywords": ["def fibonacci", "return", "for", "docstring"]},
    ),
    TestCase(
        id="code_02", category="code_generation", name="bash one-liner",
        messages=[{"role": "user", "content": "Give me a bash one-liner to find all .py files modified in the last 7 days under the current directory."}],
        max_tokens=200, check="contains_all", check_args={"keywords": ["find", "-mtime", ".py"]},
    ),
    TestCase(
        id="code_03", category="code_generation", name="bug fix",
        messages=[{"role": "user", "content": "This Python code has a bug. Fix it and explain:\n\ndef average(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total / len(numbers)\n\nprint(average([]))"}],
        max_tokens=400, check="contains_any", check_args={"keywords": ["ZeroDivisionError", "division by zero", "empty", "len(numbers) == 0", "not numbers", "if len"]},
    ),

    # ── 4. Summarisation ─────────────────────────────────────────────────
    TestCase(
        id="summary_01", category="summarisation", name="text summary",
        messages=[{"role": "user", "content": "Summarise the following paragraph in exactly 2 sentences:\n\nThe Great Wall of China is one of the greatest wonders of the world. It was built over many centuries by different Chinese dynasties to protect against invasions from the north. The wall stretches over 13,000 miles and is made of stone, brick, tamped earth, and other materials. It is a UNESCO World Heritage Site and attracts millions of visitors each year. Despite common myths, it is not visible from space with the naked eye."}],
        max_tokens=200, check="not_empty", check_args={"min_words": 10},
    ),

    # ── 5. Structured output (JSON) ──────────────────────────────────────
    TestCase(
        id="json_01", category="structured_output", name="JSON extraction",
        messages=[{"role": "user", "content": "Extract the following into a JSON object with keys \"name\", \"age\", \"city\":\n\n\"John Smith is 34 years old and lives in San Francisco.\""}],
        max_tokens=200, check="valid_json", check_args={"keys": ["name", "age", "city"]},
    ),
    TestCase(
        id="json_02", category="structured_output", name="JSON list",
        messages=[{"role": "user", "content": "Return a JSON array of 3 objects, each with keys \"language\" and \"year_created\". Include Python, JavaScript, and Rust."}],
        max_tokens=300, check="valid_json", check_args={"keys": ["language", "year_created"]},
    ),

    # ── 6. Math ──────────────────────────────────────────────────────────
    TestCase(
        id="math_01", category="math", name="arithmetic",
        messages=[{"role": "user", "content": "What is 347 × 23? Show your work and give the final answer."}],
        max_tokens=300, check="math_answer", check_args={"answer": "7981"},
    ),
    TestCase(
        id="math_02", category="math", name="word problem",
        messages=[{"role": "user", "content": "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. What is the total distance traveled?"}],
        max_tokens=300, check="math_answer", check_args={"answer": "270"},
    ),

    # ── 7. Multi-turn conversation ───────────────────────────────────────
    TestCase(
        id="multi_01", category="multi_turn", name="context recall",
        messages=[
            {"role": "user", "content": "My name is Alice and I work as a software engineer at Google."},
            {"role": "assistant", "content": "Nice to meet you, Alice! It's great to hear you're a software engineer at Google. How can I help you today?"},
            {"role": "user", "content": "What is my name and where do I work?"},
        ],
        max_tokens=128, check="multi_turn", check_args={"keywords": ["Alice", "Google"]},
    ),
    TestCase(
        id="multi_02", category="multi_turn", name="follow-up clarification",
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is its population approximately?"},
        ],
        max_tokens=200, check="contains_any", check_args={"keywords": ["2 million", "2.1 million", "2.2 million", "million", "11 million", "12 million"]},
    ),

    # ── 8. Tool / function calling readiness ─────────────────────────────
    TestCase(
        id="tool_01", category="tool_calling", name="tool call format",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to functions. When the user asks something that requires a function, output a JSON object with keys \"function\" and \"arguments\"."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        max_tokens=200, check="valid_json", check_args={"keys": ["function", "arguments"]},
    ),
    TestCase(
        id="tool_02", category="tool_calling", name="multi-step plan",
        messages=[{"role": "user", "content": "I want to build a REST API with Python. Break this down into 5 numbered steps."}],
        max_tokens=400, check="contains_all", check_args={"keywords": ["1.", "2.", "3.", "4.", "5."]},
    ),

    # ── 11. Agentic: single tool call ─────────────────────────────────────
    TestCase(
        id="agent_tool_01", category="agentic_tool_call", name="weather tool call",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
            {"role": "user", "content": "What is the current weather in London?"},
        ],
        tools=[
            {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
        ],
        tool_choice="auto",
        max_tokens=200, check="tool_call", check_args={"tool_name": "get_weather", "arg_keys": ["city"]},
    ),
    TestCase(
        id="agent_tool_02", category="agentic_tool_call", name="search tool call",
        messages=[
            {"role": "system", "content": "You are an AI assistant with tools. Call the right tool for the user."},
            {"role": "user", "content": "Find information about the latest Python release"},
        ],
        tools=[
            {"type": "function", "function": {"name": "web_search", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
        ],
        tool_choice="auto",
        max_tokens=200, check="tool_call", check_args={"tool_name": "web_search", "arg_keys": ["query"]},
    ),
    TestCase(
        id="agent_tool_03", category="agentic_tool_call", name="correct tool selection",
        messages=[
            {"role": "system", "content": "You have tools available. Use the most appropriate one."},
            {"role": "user", "content": "Calculate 15% tip on a $85.50 restaurant bill"},
        ],
        tools=[
            {"type": "function", "function": {"name": "calculator", "description": "Perform math calculations", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
            {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "send_email", "description": "Send an email", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
        ],
        tool_choice="required",
        max_tokens=200, check="tool_call", check_args={"tool_name": "calculator"},
    ),

    # ── 12. Agentic: parallel tool calls ──────────────────────────────────
    TestCase(
        id="agent_parallel_01", category="agentic_parallel", name="parallel weather calls",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. When the user asks about multiple things, call all necessary tools in parallel in a single response."},
            {"role": "user", "content": "What's the weather in Tokyo, London, and New York right now?"},
        ],
        tools=[
            {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
        ],
        tool_choice="auto",
        max_tokens=400, check="parallel_tool_calls", check_args={"expected_tools": ["get_weather", "get_weather", "get_weather"]},
    ),
    TestCase(
        id="agent_parallel_02", category="agentic_parallel", name="parallel mixed tools",
        messages=[
            {"role": "system", "content": "You are an assistant with tools. Use multiple tools in parallel when the user's request requires it."},
            {"role": "user", "content": "I need the weather in Paris and also search for 'best restaurants in Paris'"},
        ],
        tools=[
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
            {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        ],
        tool_choice="auto",
        max_tokens=400, check="parallel_tool_calls", check_args={"expected_tools": ["get_weather", "web_search"]},
    ),

    # ── 13. Agentic: tool result loop (multi-step) ────────────────────────
    TestCase(
        id="agent_loop_01", category="agentic_loop", name="use tool result in answer",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "What's the weather in Berlin?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_001", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Berlin"}'}}]},
            {"role": "tool", "tool_call_id": "call_001", "content": '{"temperature": 18, "condition": "partly cloudy", "humidity": 65}'},
        ],
        max_tokens=200, check="contains_all", check_args={"keywords": ["18", "cloudy"]},
    ),
    TestCase(
        id="agent_loop_02", category="agentic_loop", name="multi-tool results synthesis",
        messages=[
            {"role": "system", "content": "You are a travel assistant. Synthesize tool results into a helpful response."},
            {"role": "user", "content": "Compare the weather in Tokyo and Sydney for my trip planning."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_101", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}},
                {"id": "call_102", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Sydney"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_101", "content": '{"temperature": 12, "condition": "rainy", "humidity": 80}'},
            {"role": "tool", "tool_call_id": "call_102", "content": '{"temperature": 25, "condition": "sunny", "humidity": 45}'},
        ],
        max_tokens=300, check="contains_all", check_args={"keywords": ["tokyo", "sydney"]},
    ),
    TestCase(
        id="agent_loop_03", category="agentic_loop", name="chain: search then summarise",
        messages=[
            {"role": "system", "content": "You are a research assistant. Use tool results to provide accurate answers."},
            {"role": "user", "content": "What are the key features of Rust programming language?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_201", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "Rust programming language key features"}'}}]},
            {"role": "tool", "tool_call_id": "call_201", "content": '{"results": [{"title": "Rust Features", "snippet": "Rust offers memory safety without garbage collection, zero-cost abstractions, fearless concurrency, pattern matching, and a strong type system. It prevents null pointer dereferences and data races at compile time."}]}'},
        ],
        max_tokens=400, check="contains_all", check_args={"keywords": ["memory safety", "concurrency"]},
    ),

    # ── 14. Agentic: complex planning & decomposition ─────────────────────
    TestCase(
        id="agent_plan_01", category="agentic_planning", name="deploy ML model plan",
        messages=[
            {"role": "system", "content": "You are a senior ML engineer assistant. Break down complex tasks into actionable steps with specific tool calls or commands."},
            {"role": "user", "content": "I have a trained PyTorch model (model.pt) and I need to deploy it as a REST API on AWS. Give me a step-by-step plan with specific commands and tools needed."},
        ],
        max_tokens=800, check="agent_plan",
        check_args={"min_steps": 4, "required_keywords": ["docker", "api", "endpoint", "deploy"]},
    ),
    TestCase(
        id="agent_plan_02", category="agentic_planning", name="debug production issue",
        messages=[
            {"role": "system", "content": "You are a DevOps agent. When given a production incident, provide a structured debugging plan."},
            {"role": "user", "content": "Our API server is returning 502 errors intermittently. Response times have increased from 200ms to 5s. The server runs on Kubernetes with 3 pods. CPU usage is normal but memory is at 85%. What's your debugging plan?"},
        ],
        max_tokens=800, check="agent_plan",
        check_args={"min_steps": 4, "required_keywords": ["memory", "logs", "pod"]},
    ),
    TestCase(
        id="agent_plan_03", category="agentic_planning", name="data pipeline design",
        messages=[
            {"role": "user", "content": "Design a data pipeline that: 1) Scrapes product prices from 3 e-commerce sites every hour, 2) Stores data in a database, 3) Sends alerts when prices drop more than 20%, 4) Generates a weekly report. Give me the architecture with specific tools/libraries for each component."},
        ],
        max_tokens=800, check="agent_plan",
        check_args={"min_steps": 4, "required_keywords": ["scrape", "database", "alert", "report"]},
    ),

    # ── 15. Agentic: error handling & recovery ────────────────────────────
    TestCase(
        id="agent_err_01", category="agentic_error", name="handle tool failure",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with tools. When a tool returns an error, explain the problem and suggest alternatives."},
            {"role": "user", "content": "Send an email to john@example.com about the meeting."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_301", "type": "function", "function": {"name": "send_email", "arguments": '{"to": "john@example.com", "subject": "Meeting", "body": "Reminder about our upcoming meeting."}'}}]},
            {"role": "tool", "tool_call_id": "call_301", "content": '{"error": "SMTP connection failed: timeout after 30s. Server smtp.example.com is unreachable."}'},
        ],
        max_tokens=300, check="error_recovery",
        check_args={"error_keywords": ["smtp", "timeout", "unreachable", "fail", "error", "connection"], "fix_keywords": ["retry", "later", "alternative", "check", "try again", "connection", "server"]},
    ),
    TestCase(
        id="agent_err_02", category="agentic_error", name="handle invalid input",
        messages=[
            {"role": "system", "content": "You are a data analysis assistant. If the user provides incomplete or invalid data, identify the issue and ask for clarification."},
            {"role": "user", "content": "Analyze this dataset and find trends:\n\ndate,sales,region\n2024-01-01,1500,North\n2024-01-02,,South\n2024-01-03,2100,\n2024-01-04,abc,East\ninvalid_date,1800,West"},
        ],
        max_tokens=400, check="error_recovery",
        check_args={"error_keywords": ["missing", "invalid", "empty", "abc", "date"], "fix_keywords": ["clean", "fix", "handle", "replace", "remove", "impute", "correct"]},
    ),
    TestCase(
        id="agent_err_03", category="agentic_error", name="ambiguous request handling",
        messages=[
            {"role": "system", "content": "You are a precise assistant. If a request is ambiguous, identify the ambiguities and ask clarifying questions before proceeding."},
            {"role": "user", "content": "Delete the file."},
        ],
        max_tokens=300, check="contains_any",
        check_args={"keywords": ["which file", "what file", "specify", "clarify", "more information", "file name", "path", "which one"]},
    ),

    # ── 16. Agentic: constraint satisfaction ──────────────────────────────
    TestCase(
        id="agent_constraint_01", category="agentic_constraint", name="format constraints",
        messages=[
            {"role": "system", "content": "You MUST follow these rules exactly:\n1. Respond in exactly 3 bullet points\n2. Each bullet must start with an action verb\n3. Do not use the word 'simply'"},
            {"role": "user", "content": "How do I set up a Python virtual environment?"},
        ],
        max_tokens=300, check="constraint_satisfaction",
        check_args={"constraints": [
            {"type": "contains", "value": "•"},
            {"type": "not_contains", "value": "simply"},
            {"type": "min_words", "value": 15},
            {"type": "max_words", "value": 200},
        ]},
    ),
    TestCase(
        id="agent_constraint_02", category="agentic_constraint", name="output schema compliance",
        messages=[{"role": "user", "content": "Give me exactly 3 tasks for a Kanban board in JSON format. Each task must have: \"id\" (integer), \"title\" (string), \"priority\" (\"high\", \"medium\", or \"low\"), and \"estimated_hours\" (number). Nothing else."}],
        max_tokens=400, check="valid_json", check_args={"keys": ["id", "title", "priority", "estimated_hours"]},
    ),

    # ── 17. Agentic: complex multi-part queries ──────────────────────────
    TestCase(
        id="agent_complex_01", category="agentic_complex", name="code review agent",
        messages=[
            {"role": "system", "content": "You are a senior code reviewer. For each issue found, provide: 1) severity (critical/warning/info), 2) line reference, 3) explanation, 4) suggested fix. Output as JSON array."},
            {"role": "user", "content": "Review this Python code:\n\nimport os\nimport pickle\n\ndef load_user_data(filename):\n    with open(filename, 'rb') as f:\n        data = pickle.load(f)\n    return data\n\ndef save_config(config, path):\n    with open(path, 'w') as f:\n        f.write(str(config))\n\npassword = 'admin123'\ndb_url = f'postgresql://admin:{password}@localhost/mydb'\nprint(db_url)"},
        ],
        max_tokens=800, check="contains_all",
        check_args={"keywords": ["pickle", "password", "security"]},
    ),
    TestCase(
        id="agent_complex_02", category="agentic_complex", name="SQL agent simulation",
        messages=[
            {"role": "system", "content": "You are a database agent. Given a schema and question, write the SQL query, explain your reasoning, and warn about any performance concerns."},
            {"role": "user", "content": "Schema:\n  users(id, name, email, created_at)\n  orders(id, user_id, amount, status, created_at)\n  products(id, name, price, category)\n  order_items(order_id, product_id, quantity)\n\nQuestion: Find the top 5 customers by total spending in the last 30 days, along with their most purchased product category."},
        ],
        max_tokens=800, check="contains_all",
        check_args={"keywords": ["select", "join", "group by", "order by"]},
    ),
    TestCase(
        id="agent_complex_03", category="agentic_complex", name="multi-API orchestration",
        messages=[
            {"role": "system", "content": "You are an agent that coordinates multiple API calls. Given the user request, describe what API calls need to be made, in what order, and what data flows between them. If calls can be parallelised, indicate that."},
            {"role": "user", "content": "Book a trip: fly from New York to Paris on March 15, find a hotel near the Eiffel Tower for 3 nights, and rent a compact car for the duration. My budget is $3000 total."},
        ],
        max_tokens=800, check="agent_plan",
        check_args={"min_steps": 3, "required_keywords": ["flight", "hotel", "car", "parallel"]},
    ),
    TestCase(
        id="agent_complex_04", category="agentic_complex", name="ReAct-style reasoning",
        messages=[
            {"role": "system", "content": "You are an agent that uses the Thought/Action/Observation pattern. For each step, write:\nThought: your reasoning\nAction: the tool to call\nAction Input: the input\nThen wait for an Observation. Solve the user's question step by step."},
            {"role": "user", "content": "What is the population of the country where the Eiffel Tower is located, divided by 1000?"},
        ],
        max_tokens=500, check="contains_all",
        check_args={"keywords": ["thought", "action", "france"]},
    ),

    # ── 9. Long context handling ─────────────────────────────────────────
    TestCase(
        id="long_01", category="long_context", name="needle in haystack",
        messages=[{"role": "user", "content":
            "Read the following carefully:\n\n" +
            "The sky is blue. " * 50 +
            "The secret code is PINEAPPLE42. " +
            "The sky is blue. " * 50 +
            "\n\nWhat is the secret code mentioned in the text above?"
        }],
        max_tokens=64, check="contains_any", check_args={"keywords": ["PINEAPPLE42"]},
    ),

    # ── 10. Creativity ───────────────────────────────────────────────────
    TestCase(
        id="creative_01", category="creativity", name="short story",
        messages=[{"role": "user", "content": "Write a 3-sentence short story about a robot who discovers music for the first time."}],
        max_tokens=300, check="not_empty", check_args={"min_words": 15},
    ),
    TestCase(
        id="creative_02", category="creativity", name="analogy",
        messages=[{"role": "user", "content": "Explain quantum computing to a 10-year-old using an analogy."}],
        max_tokens=300, check="not_empty", check_args={"min_words": 20},
    ),
]


# ── API interaction ──────────────────────────────────────────────────────────

def call_chat_completion(
    messages: list[dict],
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> dict[str, Any]:
    """Call LM Studio endpoints, using chat/completions for tool-enabled requests."""

    # Prepare flat text once for fallback /v1/responses path.
    normalized_lines: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if content is None:
            content = ""
        normalized_lines.append(f"{role}: {content}")
        if m.get("tool_calls"):
            normalized_lines.append(f"{role}_tool_calls: {json.dumps(m['tool_calls'], ensure_ascii=False)}")
        if role == "tool" and m.get("tool_call_id"):
            normalized_lines.append(f"tool_call_id: {m['tool_call_id']}")
    input_text = "\n".join(normalized_lines) if normalized_lines else ""

    # If tools are provided, use the OpenAI-compatible tool-calling path.
    if tools:
        url = f"{CFG['server_url']}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
        }
        if tool_choice:
            payload["tool_choice"] = tool_choice

        resp = requests.post(url, json=payload, timeout=CFG['timeout'])
        if resp.status_code >= 400:
            try:
                err_payload = resp.json()
            except Exception:
                err_payload = resp.text
            raise RuntimeError(f"HTTP {resp.status_code} from {url}: {err_payload}")

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content") or ""
        tool_calls_parsed = message.get("tool_calls", []) or []

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        prompt_tokens = usage.get("prompt_tokens") or len(input_text.split())
        completion_tokens = usage.get("completion_tokens") or len(str(content).split())
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

        return {
            "choices": [{
                "message": {
                    "content": content,
                    "tool_calls": tool_calls_parsed,
                }
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "_debug": {
                "endpoint": url,
                "request_payload": payload,
                "raw_response": data,
                "parse_notes": ["chat_completions"],
            },
        }

    # No tools: use /v1/responses (faster + works well for plain generations).
    url = f"{CFG['server_url']}/v1/responses"
    payload = {
        "model": model,
        "input": input_text,
    }

    resp = requests.post(url, json=payload, timeout=CFG['timeout'])
    if resp.status_code >= 400:
        try:
            err_payload = resp.json()
        except Exception:
            err_payload = resp.text
        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {err_payload}")
    data = resp.json()

    parse_notes: list[str] = []
    text_parts: list[str] = []
    tool_calls_parsed: list[dict[str, Any]] = []

    if isinstance(data, str):
        text_parts.append(data)
        parse_notes.append("root:string")
    elif isinstance(data, dict):
        if isinstance(data.get("output_text"), str):
            text_parts.append(data["output_text"])
            parse_notes.append("output_text")

        output_items = data.get("output", [])
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")
                if item_type == "function_call":
                    tool_calls_parsed.append({
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    })
                    parse_notes.append("output:function_call")

                if item_type == "message":
                    content_list = item.get("content", [])
                    if isinstance(content_list, list):
                        for chunk in content_list:
                            if not isinstance(chunk, dict):
                                continue
                            ctype = chunk.get("type", "")
                            if ctype in {"output_text", "text", "input_text"}:
                                text = chunk.get("text")
                                if isinstance(text, str) and text.strip():
                                    text_parts.append(text)
                                    parse_notes.append(f"message:{ctype}")

                legacy_calls = item.get("tool_calls")
                if isinstance(legacy_calls, list):
                    for c in legacy_calls:
                        if isinstance(c, dict):
                            tool_calls_parsed.append(c)
                    parse_notes.append("output:tool_calls")

        if isinstance(data.get("response"), str):
            text_parts.append(data["response"])
            parse_notes.append("response")

        if not text_parts and isinstance(data.get("content"), str):
            text_parts.append(data["content"])
            parse_notes.append("content")

    content = "\n".join([t for t in text_parts if isinstance(t, str) and t.strip()]).strip()
    if not content:
        content = json.dumps(data, ensure_ascii=False)
        parse_notes.append("fallback:raw_json")

    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or len(input_text.split())
    )
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or len(content.split())
    )
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    # Return in OpenAI-like format for compatibility
    return {
        "choices": [{
            "message": {
                "content": content,
                "tool_calls": tool_calls_parsed,
            }
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "_debug": {
            "endpoint": url,
            "request_payload": payload,
            "raw_response": data,
            "parse_notes": parse_notes,
        },
    }


def health_check() -> bool:
    """Check if the server is up by trying a simple request."""
    try:
        payload = {"model": CFG["model"], "input": "Hello"}
        r = requests.post(f"{CFG['server_url']}/v1/responses", json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ── Runner ───────────────────────────────────────────────────────────────────

def run_test(tc: TestCase, model: str, debug_path: Path | None = None) -> TestResult:
    """Execute a single test case and evaluate its result."""
    start = time.perf_counter()
    try:
        data = call_chat_completion(
            messages=tc.messages,
            model=model,
            max_tokens=tc.max_tokens,
            temperature=tc.temperature,
            tools=tc.tools,
            tool_choice=tc.tool_choice,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = data["choices"][0]
        # Handle reasoning_content (think tags) — extract main content
        content = choice["message"].get("content", "") or ""
        reasoning = choice["message"].get("reasoning_content", "")

        # Extract structured tool calls if present
        tool_calls_raw = choice["message"].get("tool_calls", []) or []

        debug_meta = data.get("_debug", {})

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        tps = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

        # Run the check — inject _tool_calls for tool-aware checkers
        check_fn = CHECKS.get(tc.check, check_not_empty)
        extra_args = {**tc.check_args}
        if tool_calls_raw:
            extra_args["_tool_calls"] = tool_calls_raw
            # If content is empty (pure tool call), build a text repr for fallback checks
            if not content:
                content = json.dumps([{"function": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in tool_calls_raw])
        passed, score, reason = check_fn(content, **extra_args)

        append_debug_record(debug_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "test_id": tc.id,
            "category": tc.category,
            "name": tc.name,
            "check": tc.check,
            "check_args": tc.check_args,
            "passed": passed,
            "score": score,
            "reason": reason,
            "latency_ms": round(elapsed_ms, 1),
            "tokens_per_sec": round(tps, 2),
            "response_excerpt": content[:1000],
            "tool_calls": tool_calls_raw,
            "debug": debug_meta,
        })

        return TestResult(
            id=tc.id, category=tc.category, name=tc.name,
            passed=passed, score=score, reason=reason,
            response_text=content[:500],  # truncate for CSV
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(elapsed_ms, 1),
            tokens_per_sec=round(tps, 2),
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        append_debug_record(debug_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "test_id": tc.id,
            "category": tc.category,
            "name": tc.name,
            "check": tc.check,
            "check_args": tc.check_args,
            "passed": False,
            "score": 0.0,
            "reason": "error",
            "latency_ms": round(elapsed_ms, 1),
            "error": str(e),
        })
        return TestResult(
            id=tc.id, category=tc.category, name=tc.name,
            passed=False, score=0.0, reason="error",
            response_text="", prompt_tokens=0, completion_tokens=0,
            total_tokens=0, latency_ms=round(elapsed_ms, 1),
            tokens_per_sec=0.0, error=str(e),
        )


def run_all(model: str, out_dir: Path) -> list[TestResult]:
    """Run all test cases sequentially, printing progress."""
    results: list[TestResult] = []
    total = len(TEST_CASES)
    debug_path = out_dir / "debug.json"
    if debug_path.exists():
        debug_path.unlink()

    print(f"\n{'='*70
               }")
    print(f"  Qwen3.5-4B Capability Benchmark")
    print(f"  Server  : {CFG['server_url']}")
    print(f"  Model   : {model}")
    print(f"  Tests   : {total}")
    print(f"  Output  : {out_dir}")
    print(f"  Debug   : {debug_path}")
    print(f"{'='*70}\n")

    for i, tc in enumerate(TEST_CASES, 1):
        tag = f"[{i:2d}/{total}]"
        print(f"{tag} {tc.category:20s} | {tc.name:30s} ", end="", flush=True)
        result = run_test(tc, model, debug_path=debug_path)
        status = "PASS ✓" if result.passed else "FAIL ✗"
        print(f"| {status} | {result.score:.0%} | {result.latency_ms:7.0f}ms | {result.tokens_per_sec:5.1f} t/s")
        if result.error:
            print(f"       ERROR: {result.error[:100]}")
        results.append(result)

    return results


# ── Output ───────────────────────────────────────────────────────────────────

def write_csv(results: list[TestResult], path: Path):
    """Write all results to CSV."""
    fieldnames = [
        "id", "category", "name", "passed", "score", "reason",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "latency_ms", "tokens_per_sec", "error", "response_text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # Truncate response for readability
            row["response_text"] = row["response_text"][:300].replace("\n", " ")
            writer.writerow(row)
    print(f"\nCSV written: {path}")


def write_summary(results: list[TestResult], path: Path):
    """Write a human-readable summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / total if total else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total else 0
    avg_tps = sum(r.tokens_per_sec for r in results) / total if total else 0

    # Per-category breakdown
    categories: dict[str, list[TestResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    lines = []
    lines.append("=" * 70)
    lines.append("  BENCHMARK SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Server        : {CFG['server_url']}")
    lines.append(f"  Total tests   : {total}")
    lines.append(f"  Passed        : {passed}/{total} ({passed/total:.0%})")
    lines.append(f"  Avg score     : {avg_score:.2f}")
    lines.append(f"  Avg latency   : {avg_latency:.0f} ms")
    lines.append(f"  Avg tokens/s  : {avg_tps:.1f}")
    lines.append("")
    lines.append("  Per-Category Results:")
    lines.append(f"  {'Category':<22s} {'Pass':>6s} {'Score':>7s} {'Avg ms':>8s} {'Avg t/s':>8s}")
    lines.append("  " + "-" * 55)
    for cat, cat_results in categories.items():
        cat_pass = sum(1 for r in cat_results if r.passed)
        cat_total = len(cat_results)
        cat_score = sum(r.score for r in cat_results) / cat_total
        cat_lat = sum(r.latency_ms for r in cat_results) / cat_total
        cat_tps = sum(r.tokens_per_sec for r in cat_results) / cat_total
        lines.append(f"  {cat:<22s} {cat_pass}/{cat_total:>3d}  {cat_score:>6.0%}  {cat_lat:>7.0f}  {cat_tps:>7.1f}")

    lines.append("")
    lines.append("  Failed Tests:")
    failed = [r for r in results if not r.passed]
    if not failed:
        lines.append("    (none)")
    else:
        for r in failed:
            lines.append(f"    [{r.id}] {r.name}: {r.reason}")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary written: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B Model Capability Benchmark")
    parser.add_argument("--server-url", default=CFG["server_url"], help="Server base URL")
    parser.add_argument("--model", default=CFG["model"], help="Model name for API requests")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: auto-timestamped)")
    parser.add_argument("--timeout", type=int, default=CFG["timeout"], help="Per-request timeout in seconds")
    parser.add_argument(
        "--native-tools",
        action="store_true",
        help="Send native tools/tool_choice fields to /v1/responses (disabled by default)",
    )
    args = parser.parse_args()

    CFG["server_url"] = args.server_url
    CFG["timeout"] = args.timeout
    CFG["native_tools"] = args.native_tools or CFG.get("native_tools", False)

    # Health check
    print(f"Checking server at {CFG['server_url']} …")
    print(f"Native tools mode: {'ON' if CFG.get('native_tools') else 'OFF (text fallback)'}")
    if not health_check():
        print(f"ERROR: Server not reachable at {CFG['server_url']}")
        print("Start it with: ./qwen35_server.sh &")
        sys.exit(1)
    print("Server is healthy ✓")

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).parent / f"benchmark_qwen35_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run
    results = run_all(args.model, out_dir)

    # Write outputs
    write_csv(results, out_dir / "results.csv")
    write_summary(results, out_dir / "summary.txt")

    # Quick exit code: fail if < 50% pass
    passed = sum(1 for r in results if r.passed)
    if passed < len(results) * 0.5:
        print(f"\n⚠  Only {passed}/{len(results)} tests passed — model may not be suitable for agent use.")
        sys.exit(1)
    else:
        print(f"\n✓  {passed}/{len(results)} tests passed — model looks good for agent use.")
        sys.exit(0)


if __name__ == "__main__":
    main()

# cd /home/nav_wsl/code/learn_llama_cpp && python3 qwen35_benchmark.py --model qwen35 --timeout 300