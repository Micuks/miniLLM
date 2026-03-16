from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any


_WHITESPACE = re.compile(r"\s+")


@dataclass
class EvalResult:
    exact_match: bool
    execution_match: bool | None
    pred_sql: str
    gold_sql: str
    error: str | None = None


def normalize_sql(sql: str) -> str:
    """A lightweight SQL normalizer for stable string matching.

    We intentionally avoid heavyweight parser dependencies so this works in minimal
    environments used for interview demos.
    """
    s = sql.strip().rstrip(";")
    s = _WHITESPACE.sub(" ", s).strip()
    return s.lower()


def _run_query(conn: sqlite3.Connection, query: str) -> list[tuple[Any, ...]]:
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    # stabilize ordering when SQL has no ORDER BY
    return sorted(rows)


def execution_match(schema_ddl: str, pred_sql: str, gold_sql: str) -> tuple[bool | None, str | None]:
    """Return execution match on in-memory sqlite.

    Returns (None, error) when execution cannot be performed.
    """
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(schema_ddl)
            pred_rows = _run_query(conn, pred_sql)
            gold_rows = _run_query(conn, gold_sql)
        return pred_rows == gold_rows, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def score_text2sql(schema_ddl: str, pred_sql: str, gold_sql: str, *, with_execution: bool) -> EvalResult:
    exact = normalize_sql(pred_sql) == normalize_sql(gold_sql)
    exec_match = None
    err = None
    if with_execution:
        exec_match, err = execution_match(schema_ddl, pred_sql, gold_sql)
    return EvalResult(
        exact_match=exact,
        execution_match=exec_match,
        pred_sql=pred_sql.strip(),
        gold_sql=gold_sql.strip(),
        error=err,
    )
