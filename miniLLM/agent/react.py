"""ReAct format: prompt templates, trajectory parsing, and data structures.

The agent follows a Thought -> Action -> Observation loop:
  Thought: reasoning about what to do
  Action: execute_sql["SELECT ..."]
  Observation: (injected by environment)
  ...
  Answer: final SQL query
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..sql_eval import extract_sql

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """\
You are a Text-to-SQL agent. Given a database schema and a natural language question, \
solve it step by step using the tools available to you.

## Available Tools
- execute_sql[\"<SQL query>\"]: Execute a SQL query against the database and observe results.

## Format
You MUST follow this format exactly:

Thought: <your reasoning>
Action: execute_sql["<SQL query>"]
Observation: <result from execution, provided by the system>
... (you may repeat Thought/Action/Observation as needed)
Thought: <final reasoning>
Answer: <final SQL query>

## Rules
1. Always start with a Thought.
2. You may execute exploratory queries (e.g., check table columns, sample rows) before writing the final query.
3. If a query returns an error, analyze the error and try a corrected query.
4. End with Answer: containing ONLY the final SQL query, no explanation."""


def build_react_user_prompt(schema: str, question: str) -> str:
    return (
        f"### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}"
    )


def build_react_messages(
    schema: str,
    question: str,
    trajectory: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for ReAct format.

    If trajectory is provided, include it as the assistant response (for SFT).
    Otherwise, return prompt-only messages (for inference / GRPO).
    """
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": build_react_user_prompt(schema, question)},
    ]
    if trajectory is not None:
        messages.append({"role": "assistant", "content": trajectory})
    return messages


def build_react_sft_text(
    schema: str, question: str, trajectory: str, tokenizer
) -> str:
    """Build a complete SFT training string with the tokenizer's chat template."""
    messages = build_react_messages(schema, question, trajectory)
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_react_inference_prompt(schema: str, question: str, tokenizer) -> str:
    """Build inference prompt (no assistant turn, with generation prompt)."""
    messages = build_react_messages(schema, question)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

_ACTION_PATTERN = re.compile(
    r'Action:\s*execute_sql\["(.+?)"\]', re.DOTALL
)
_ANSWER_PATTERN = re.compile(
    r"Answer:\s*(.+?)(?=\n(?:Thought:|Action:|Observation:)|$)", re.DOTALL
)
_SQL_START_PATTERN = re.compile(r"\b(select|with)\b", re.IGNORECASE)


@dataclass
class TrajectoryStep:
    thought: str = ""
    action_sql: str | None = None
    observation: str | None = None


@dataclass
class ParsedTrajectory:
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str | None = None
    raw: str = ""


def parse_trajectory(text: str) -> ParsedTrajectory:
    """Parse a ReAct trajectory string into structured steps."""
    result = ParsedTrajectory(raw=text)

    # Extract final answer
    answer_m = _ANSWER_PATTERN.search(text)
    if answer_m:
        result.final_answer = answer_m.group(1).strip()

    # Split by Thought: markers
    parts = re.split(r"(?=Thought:)", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        step = TrajectoryStep()

        # Extract thought
        thought_m = re.match(r"Thought:\s*(.+?)(?=Action:|Answer:|$)", part, re.DOTALL)
        if thought_m:
            step.thought = thought_m.group(1).strip()

        # Extract action
        action_m = _ACTION_PATTERN.search(part)
        if action_m:
            step.action_sql = action_m.group(1).strip()

        # Extract observation
        obs_m = re.search(r"Observation:\s*(.+?)(?=Thought:|Answer:|$)", part, re.DOTALL)
        if obs_m:
            step.observation = obs_m.group(1).strip()

        if step.thought or step.action_sql:
            result.steps.append(step)

    return result


def _looks_like_sql(text: str) -> bool:
    return bool(_SQL_START_PATTERN.match(text.strip()))


def _strip_outer_wrappers(text: str) -> str:
    cleaned = text.strip()
    wrappers = [('"', '"'), ("'", "'"), ("`", "`")]
    changed = True
    while changed and cleaned:
        changed = False
        for left, right in wrappers:
            if cleaned.startswith(left) and cleaned.endswith(right):
                inner = cleaned[len(left):-len(right)].strip()
                if not inner:
                    continue
                if _looks_like_sql(inner) or _SQL_START_PATTERN.search(inner):
                    cleaned = inner
                    changed = True
    return cleaned


def _clean_sql_candidate(text: str) -> str | None:
    candidate = extract_sql(text).strip()
    if not candidate:
        return None

    marker_split = re.split(r"\n(?:Thought:|Action:|Observation:)", candidate, maxsplit=1)
    candidate = marker_split[0].strip()
    candidate = re.sub(r"(?is)^sql\s*[:\-]\s*", "", candidate).strip()
    candidate = _strip_outer_wrappers(candidate)

    keyword_match = _SQL_START_PATTERN.search(candidate)
    if keyword_match and keyword_match.start() > 0:
        candidate = candidate[keyword_match.start():].strip()

    candidate = _strip_outer_wrappers(candidate)
    candidate = candidate.strip().strip("`").strip()

    for quote in ("'", '"', "`"):
        while candidate.startswith(quote):
            trimmed = candidate[1:].lstrip()
            if candidate.count(quote) % 2 == 1 and _looks_like_sql(trimmed):
                candidate = trimmed
                continue
            break
        while candidate.endswith(quote):
            trimmed = candidate[:-1].rstrip()
            if candidate.count(quote) % 2 == 1 and _looks_like_sql(trimmed):
                candidate = trimmed
                continue
            break

    if ";" in candidate:
        head, tail = candidate.split(";", 1)
        if head.strip() and tail.strip():
            candidate = head.strip()

    return candidate.strip() or None


def extract_final_sql(text: str) -> str | None:
    """Extract the final SQL answer from a ReAct trajectory."""
    parsed = parse_trajectory(text)
    if parsed.final_answer:
        return _clean_sql_candidate(parsed.final_answer)
    # Fallback: last action SQL
    for step in reversed(parsed.steps):
        if step.action_sql:
            return _clean_sql_candidate(step.action_sql)
    return None
