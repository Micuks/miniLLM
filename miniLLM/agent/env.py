"""SQL execution environment for Agent training and evaluation.

Provides a sandboxed SQLite environment where the agent can execute SQL
queries and receive structured feedback (results or errors).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..sql_eval import extract_sql


@dataclass
class ExecutionResult:
    """Structured result from executing a SQL query."""
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: str | None = None

    def format_observation(self, max_rows: int = 20) -> str:
        """Format as a human-readable observation string for the ReAct trajectory."""
        if not self.success:
            return f"Error: {self.error}"
        if not self.rows:
            return "Query executed successfully. No rows returned."
        header = " | ".join(self.columns) if self.columns else ""
        lines = [header, "-" * len(header)] if header else []
        for row in self.rows[:max_rows]:
            lines.append(" | ".join(str(v) for v in row))
        if len(self.rows) > max_rows:
            lines.append(f"... ({len(self.rows) - max_rows} more rows)")
        return "\n".join(lines)


class SQLExecutionEnv:
    """Sandboxed SQL execution environment backed by in-memory SQLite.

    Usage:
        env = SQLExecutionEnv(schema_ddl)
        result = env.execute("SELECT * FROM users LIMIT 5")
        print(result.format_observation())
    """

    def __init__(self, schema_ddl: str) -> None:
        self.schema_ddl = schema_ddl
        # Pre-validate DDL
        self._conn = sqlite3.connect(":memory:")
        try:
            self._conn.executescript(schema_ddl)
        except sqlite3.Error as e:
            self._conn.close()
            raise ValueError(f"Invalid schema DDL: {e}") from e

    def execute(self, sql: str) -> ExecutionResult:
        """Execute a SQL query and return structured results."""
        sql = extract_sql(sql).strip().rstrip(";")
        if not sql:
            return ExecutionResult(success=False, error="Empty SQL query")
        try:
            cur = self._conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return ExecutionResult(success=True, columns=columns, rows=sorted(rows))
        except sqlite3.Error as e:
            return ExecutionResult(success=False, error=str(e))

    def check_correctness(self, pred_sql: str, gold_sql: str) -> tuple[bool, str | None]:
        """Check if pred_sql produces same results as gold_sql."""
        pred_result = self.execute(pred_sql)
        gold_result = self.execute(gold_sql)
        if not pred_result.success:
            return False, f"Prediction error: {pred_result.error}"
        if not gold_result.success:
            return False, f"Gold error: {gold_result.error}"
        return pred_result.rows == gold_result.rows, None

    def reset(self) -> None:
        """Reset the database to initial schema state."""
        self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(self.schema_ddl)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SQLExecutionEnvFromDB(SQLExecutionEnv):
    """Execution environment backed by an existing SQLite database file.

    Unlike SQLExecutionEnv which uses in-memory DB from DDL,
    this connects to a real database with actual data (e.g., Spider DBs).
    """

    def __init__(self, db_path: str) -> None:
        self.schema_ddl = ""
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)

    def reset(self) -> None:
        self._conn.close()
        self._conn = sqlite3.connect(self._db_path)
