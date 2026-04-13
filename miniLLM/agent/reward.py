"""Reward functions for GRPO training.

Supports both in-memory DDL execution and real database files.
The default profile is intentionally denser than the original 0/1 reward so
GRPO can distinguish "well-formed but not yet correct" trajectories from
completely collapsed outputs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .env import SQLExecutionEnv, SQLExecutionEnvFromDB
from .react import extract_final_sql, parse_trajectory
from ..sql_eval import normalize_sql


_SQL_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]*|!=|<=|>=|=|<|>|[(),.*]", re.IGNORECASE)
_WHOLE_QUERY_QUOTED_RE = re.compile(r'^\s*["\'`]+\s*(select|with)\b', re.IGNORECASE)
_CLAUSE_PATTERNS = {
    "select": re.compile(r"\bselect\b", re.IGNORECASE),
    "from": re.compile(r"\bfrom\b", re.IGNORECASE),
    "where": re.compile(r"\bwhere\b", re.IGNORECASE),
    "join": re.compile(r"\bjoin\b", re.IGNORECASE),
    "group_by": re.compile(r"\bgroup\s+by\b", re.IGNORECASE),
    "order_by": re.compile(r"\border\s+by\b", re.IGNORECASE),
    "having": re.compile(r"\bhaving\b", re.IGNORECASE),
    "limit": re.compile(r"\blimit\b", re.IGNORECASE),
    "distinct": re.compile(r"\bdistinct\b", re.IGNORECASE),
    "union": re.compile(r"\bunion\b", re.IGNORECASE),
    "intersect": re.compile(r"\bintersect\b", re.IGNORECASE),
    "except": re.compile(r"\bexcept\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class RewardWeights:
    format: float
    validity: float
    structure: float
    execution: float
    correctness: float


@dataclass(frozen=True)
class RewardBreakdown:
    format_score: float
    validity_score: float
    structure_score: float
    execution_score: float
    correctness_score: float
    error_penalty: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def resolve_reward_weights(profile: str = "dense", progress: float | None = None) -> RewardWeights:
    """Resolve reward weights for a profile at the given training progress."""
    if profile == "legacy":
        return RewardWeights(
            format=0.10,
            validity=0.00,
            structure=0.00,
            execution=0.20,
            correctness=0.70,
        )

    # Default: early training prioritizes format/validity, then shifts weight
    # toward execution correctness once the policy stops collapsing.
    t = min(max(progress or 0.0, 0.0), 1.0)
    start = RewardWeights(
        format=0.35,
        validity=0.20,
        structure=0.25,
        execution=0.10,
        correctness=0.10,
    )
    end = RewardWeights(
        format=0.24,
        validity=0.16,
        structure=0.14,
        execution=0.16,
        correctness=0.30,
    )
    return RewardWeights(
        format=(1 - t) * start.format + t * end.format,
        validity=(1 - t) * start.validity + t * end.validity,
        structure=(1 - t) * start.structure + t * end.structure,
        execution=(1 - t) * start.execution + t * end.execution,
        correctness=(1 - t) * start.correctness + t * end.correctness,
    )


def format_reward(completion: str) -> float:
    """Reward for following the ReAct format correctly."""
    text = completion.strip()
    parsed = parse_trajectory(completion)
    has_thought = bool(re.search(r"(^|\n)Thought:", completion))
    has_action = any(step.action_sql for step in parsed.steps)
    has_answer = parsed.final_answer is not None
    answer_count = len(re.findall(r"(^|\n)Answer:", completion))

    score = 0.0
    if text.startswith("Thought:"):
        score += 0.15
    if has_thought:
        score += 0.10
    if has_action:
        score += 0.20
    if has_answer:
        score += 0.35
    if has_action and has_answer and completion.rfind("Answer:") > completion.rfind("Action:"):
        score += 0.10
    if has_answer and text.endswith(parsed.final_answer or ""):
        score += 0.10

    # Explicit penalties are important because format collapse was the dominant
    # failure mode in the previous GRPO run.
    if not has_answer:
        score -= 0.35
    if not text.startswith("Thought:"):
        score -= 0.10
    if answer_count > 1:
        score -= 0.10 * (answer_count - 1)

    return max(min(score, 1.0), -0.5)


def _make_env(schema_ddl: str, db_path: str | None = None):
    """Create the appropriate execution environment."""
    if db_path:
        return SQLExecutionEnvFromDB(db_path)
    return SQLExecutionEnv(schema_ddl)


def _sql_token_set(sql: str) -> set[str]:
    return {tok.lower() for tok in _SQL_TOKEN_RE.findall(normalize_sql(sql))}


def _clause_vector(sql: str) -> dict[str, bool]:
    normalized = normalize_sql(sql)
    return {name: bool(pattern.search(normalized)) for name, pattern in _CLAUSE_PATTERNS.items()}


def _f1(overlap: int, pred_count: int, gold_count: int) -> float:
    if pred_count == 0 or gold_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_count
    recall = overlap / gold_count
    return 2 * precision * recall / max(precision + recall, 1e-8)


def sql_validity_reward(
    completion: str,
    schema_ddl: str,
    db_path: str | None = None,
    *,
    executed_successfully: bool | None = None,
) -> float:
    """Dense reward for producing something that at least looks like valid SQL."""
    sql = extract_final_sql(completion)
    if not sql:
        return 0.0

    normalized = normalize_sql(sql)
    score = 0.0
    if normalized.startswith(("select ", "with ")):
        score += 0.45
    if " from " in f" {normalized} ":
        score += 0.20
    if len(normalized.split()) >= 4:
        score += 0.10

    if executed_successfully is None:
        try:
            with _make_env(schema_ddl, db_path) as env:
                executed_successfully = env.execute(sql).success
        except Exception:  # noqa: BLE001
            executed_successfully = False

    if executed_successfully:
        return 1.0
    return min(score, 0.75)


def execution_error_penalty(sql: str | None, error: str | None) -> float:
    """Extra negative reward for common EX=? failure modes."""
    if not error:
        return 0.0

    lowered = error.lower()
    penalty = 0.0
    if "empty sql" in lowered:
        penalty -= 0.20
    if "syntax error" in lowered:
        penalty -= 0.35
    if "no such column" in lowered:
        penalty -= 0.35
    if "no such table" in lowered:
        penalty -= 0.30
    if "ambiguous column name" in lowered:
        penalty -= 0.20
    if sql and _WHOLE_QUERY_QUOTED_RE.match(sql):
        penalty -= 0.10
    return max(penalty, -0.60)


def sql_structure_reward(completion: str, gold_sql: str) -> float:
    """Token/clause overlap with the gold SQL for denser supervision."""
    sql = extract_final_sql(completion)
    if not sql:
        return 0.0

    pred_norm = normalize_sql(sql)
    gold_norm = normalize_sql(gold_sql)
    if pred_norm == gold_norm:
        return 1.0

    pred_tokens = _sql_token_set(pred_norm)
    gold_tokens = _sql_token_set(gold_norm)
    token_overlap = len(pred_tokens & gold_tokens)
    token_f1 = _f1(token_overlap, len(pred_tokens), len(gold_tokens))

    pred_clauses = _clause_vector(pred_norm)
    gold_clauses = _clause_vector(gold_norm)
    clause_matches = sum(
        1 for name in _CLAUSE_PATTERNS if pred_clauses[name] == gold_clauses[name]
    )
    clause_score = clause_matches / max(len(_CLAUSE_PATTERNS), 1)

    return min(1.0, 0.75 * token_f1 + 0.25 * clause_score)


def _row_set(rows: list[tuple[object, ...]]) -> set[tuple[object, ...]]:
    return {tuple(row) for row in rows}


def execution_partial_reward(
    completion: str,
    gold_sql: str,
    schema_ddl: str,
    db_path: str | None = None,
) -> float:
    """Reward partial execution agreement even when exact correctness is false."""
    sql = extract_final_sql(completion)
    if not sql:
        return 0.0

    try:
        with _make_env(schema_ddl, db_path) as env:
            pred_result = env.execute(sql)
            gold_result = env.execute(gold_sql)
            return _execution_partial_reward_from_results(pred_result, gold_result)
    except Exception:  # noqa: BLE001
        return 0.0


def _execution_partial_reward_from_results(pred_result, gold_result) -> float:
    if not pred_result.success or not gold_result.success:
        return 0.0
    if pred_result.rows == gold_result.rows:
        return 1.0

    pred_rows = _row_set(pred_result.rows)
    gold_rows = _row_set(gold_result.rows)
    row_overlap = len(pred_rows & gold_rows)
    row_score = _f1(row_overlap, len(pred_rows), len(gold_rows))

    pred_cols = {c.lower() for c in pred_result.columns}
    gold_cols = {c.lower() for c in gold_result.columns}
    col_overlap = len(pred_cols & gold_cols)
    col_score = _f1(col_overlap, len(pred_cols), len(gold_cols))

    pred_n = len(pred_result.rows)
    gold_n = len(gold_result.rows)
    size_score = min(pred_n, gold_n) / max(pred_n, gold_n, 1)

    return min(1.0, 0.70 * row_score + 0.20 * col_score + 0.10 * size_score)


def execution_reward(completion: str, schema_ddl: str, db_path: str | None = None) -> float:
    """Reward for producing executable SQL."""
    sql = extract_final_sql(completion)
    if not sql:
        return 0.0
    try:
        with _make_env(schema_ddl, db_path) as env:
            result = env.execute(sql)
            return 1.0 if result.success else 0.0
    except (ValueError, Exception):
        return 0.0


def correctness_reward(
    completion: str, gold_sql: str, schema_ddl: str, db_path: str | None = None
) -> float:
    """Reward for producing correct SQL (execution match with gold)."""
    sql = extract_final_sql(completion)
    if not sql:
        return 0.0
    try:
        with _make_env(schema_ddl, db_path) as env:
            correct, _ = env.check_correctness(sql, gold_sql)
            return 1.0 if correct else 0.0
    except (ValueError, Exception):
        return 0.0


def combined_reward(
    completion: str,
    gold_sql: str,
    schema_ddl: str,
    db_path: str | None = None,
    *,
    profile: str = "dense",
    progress: float | None = None,
    weights: RewardWeights | None = None,
) -> float:
    """Combined reward with denser intermediate signals."""
    return reward_breakdown(
        completion,
        gold_sql,
        schema_ddl,
        db_path=db_path,
        profile=profile,
        progress=progress,
        weights=weights,
    ).total


def reward_breakdown(
    completion: str,
    gold_sql: str,
    schema_ddl: str,
    db_path: str | None = None,
    *,
    profile: str = "dense",
    progress: float | None = None,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """Return detailed reward components for logging and ablation."""
    weights = weights or resolve_reward_weights(profile, progress)

    r_fmt = format_reward(completion)
    r_struct = sql_structure_reward(completion, gold_sql)

    sql = extract_final_sql(completion)
    pred_result = None
    gold_result = None
    executed_successfully = False
    pred_error = None
    if sql:
        try:
            with _make_env(schema_ddl, db_path) as env:
                pred_result = env.execute(sql)
                gold_result = env.execute(gold_sql)
            executed_successfully = bool(pred_result.success)
            pred_error = pred_result.error
        except Exception:  # noqa: BLE001
            pred_result = None
            gold_result = None

    r_valid = sql_validity_reward(
        completion,
        schema_ddl,
        db_path=db_path,
        executed_successfully=executed_successfully,
    )
    if pred_result is not None and gold_result is not None:
        r_exec = _execution_partial_reward_from_results(pred_result, gold_result)
        r_correct = (
            1.0
            if pred_result.success and gold_result.success and pred_result.rows == gold_result.rows
            else 0.0
        )
    else:
        r_exec = 0.0
        r_correct = 0.0
    r_err = execution_error_penalty(sql, pred_error)

    total = (
        weights.format * r_fmt
        + weights.validity * r_valid
        + weights.structure * r_struct
        + weights.execution * r_exec
        + weights.correctness * r_correct
        + r_err
    )
    return RewardBreakdown(
        format_score=r_fmt,
        validity_score=r_valid,
        structure_score=r_struct,
        execution_score=r_exec,
        correctness_score=r_correct,
        error_penalty=r_err,
        total=total,
    )
