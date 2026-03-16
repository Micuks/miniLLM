from __future__ import annotations

from typing import Dict, List


SYSTEM_PROMPT = (
    "You are a precise Text-to-SQL assistant. "
    "Given database DDL (CREATE TABLE ...) and a natural language question, "
    "return only the SQL query that answers the question."
)


def _build_messages(schema: str, question: str, answer: str | None = None) -> List[Dict[str, str]]:
    """Build a chat-style message list for Text-to-SQL."""
    user_content = (
        f"### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def build_supervised_chat(sample: Dict[str, str], tokenizer) -> str:
    """Return a fully-formatted SFT training string using the tokenizer's native chat template."""
    messages = _build_messages(
        schema=sample.get("context", ""),
        question=sample.get("question", ""),
        answer=sample.get("answer", ""),
    )
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_inference_prompt(schema: str, question: str, tokenizer) -> str:
    """Return an inference prompt with add_generation_prompt=True."""
    messages = _build_messages(schema=schema, question=question)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
