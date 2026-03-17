"""Spider benchmark data loader with difficulty classification."""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class SpiderExample:
    db_id: str
    question: str
    gold_sql: str
    schema_ddl: str
    difficulty: str


def _classify_difficulty(sql: str) -> str:
    """Classify SQL difficulty based on structural complexity.

    Heuristic based on the Spider benchmark's difficulty criteria:
    - easy: single table, no JOIN/subquery/GROUP BY/ORDER BY/set ops
    - medium: one JOIN or one aggregation clause
    - hard: multiple JOINs, subqueries, or complex clauses
    """
    sql_lower = sql.lower()
    joins = len(re.findall(r'\bjoin\b', sql_lower))
    subqueries = sql_lower.count('select') - 1
    has_group = bool(re.search(r'\bgroup\s+by\b', sql_lower))
    has_order = bool(re.search(r'\border\s+by\b', sql_lower))
    has_having = bool(re.search(r'\bhaving\b', sql_lower))
    has_set_op = bool(re.search(r'\b(intersect|union|except)\b', sql_lower))
    has_nested = subqueries > 0

    complexity = joins + subqueries + int(has_group) + int(has_having) + int(has_set_op)

    if complexity == 0 and not has_order:
        return "easy"
    elif complexity <= 1:
        return "medium"
    else:
        return "hard"


def _build_schema_placeholder(db_id: str) -> str:
    """Return a minimal schema placeholder when real DDL is unavailable."""
    return f"-- database: {db_id}"


def load_spider(
    split: str = "validation",
    max_samples: int | None = None,
    schema_source: str | None = None,
) -> list[SpiderExample]:
    """Load Spider benchmark examples.

    Args:
        split: Dataset split ('train' or 'validation').
        max_samples: Limit number of examples. None = all.
        schema_source: Optional path/name of a dataset with schema DDL.
            If None, builds a schema placeholder from db_id.
    """
    ds = load_dataset("xlangai/spider", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Try to build a db_id -> schema_ddl mapping from sql-create-context
    schema_map: dict[str, str] = {}
    if schema_source:
        try:
            ctx_ds = load_dataset(schema_source, split="train")
            for ex in ctx_ds:
                # sql-create-context has 'context' field with DDL
                ddl = ex.get("context", "")
                if ddl:
                    schema_map[ddl] = ddl
        except Exception:
            pass

    examples: list[SpiderExample] = []
    for row in ds:
        db_id = row["db_id"]
        gold_sql = row["query"]
        question = row["question"]

        schema_ddl = _build_schema_placeholder(db_id)
        difficulty = _classify_difficulty(gold_sql)

        examples.append(SpiderExample(
            db_id=db_id,
            question=question,
            gold_sql=gold_sql,
            schema_ddl=schema_ddl,
            difficulty=difficulty,
        ))

    return examples
