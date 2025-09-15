from typing import Dict


SYSTEM_PROMPT = (
    "You are a precise Text-to-SQL assistant. "
    "Given database DDL (CREATE TABLE ...) and a natural language question, "
    "return only the SQL query that answers the question."
)


def build_supervised_chat(sample: Dict[str, str]) -> str:
    """Return a single-string prompt for supervised fine-tuning.

    The returned text follows an instruct-style conversation format used by modern chat models.
    """
    context = sample.get("context", "").strip()
    question = sample.get("question", "").strip()
    answer = sample.get("answer", "").strip()

    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"### Database Schema:\n{context}\n\n"
        f"### Question:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{answer}<|eot_id|>"
    )


def build_inference_prompt(schema: str, question: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"### Database Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )



