# miniLLM Agent

Agent-first Text-to-SQL project built around ReAct trajectories, real SQLite execution feedback, DeepSpeed training, and practical deployment on single-GPU machines.

## Current Status

The repository started as a Text-to-SQL SFT baseline and has now pivoted to an agent workflow:

- `Direct SQL` for fast first-pass generation
- `Interactive ReAct` for tool-using correction when direct generation fails
- `ReAct SFT` as the current best training recipe
- `GRPO` as an experimental RL branch for reward-driven improvement

On Spider validation (`200` samples), the current best deployed strategy is:

- `ReAct SFT v2 + Direct + Interactive Fallback`: `78.6%` EX
- `ReAct SFT v2` direct-only: `71.7%` EX
- Current `GRPO` branch is useful as an experiment, but does **not** beat ReAct SFT yet

See [docs/REPORT.md](docs/REPORT.md) for the full write-up and [docs/AGENT.md](docs/AGENT.md) for the code-level walkthrough.

## What This Repo Contains

- ReAct trajectory generation with teacher models and real DB observations
- ReAct SFT training with native DeepSpeed ZeRO-2
- GRPO RL training with execution-based rewards
- Agent evaluation in `single-pass` and `interactive` modes
- Baseline SQL SFT, serving, quantization, and benchmarking utilities

## Training Stack

### 1. ReAct data generation

Teacher trajectories are generated for Spider train examples, then every `Action` SQL is executed against the real SQLite database and rewritten with real `Observation` text.

Output record format:

```json
{
  "schema": "...",
  "question": "...",
  "gold_sql": "...",
  "trajectory": "Thought: ...\nAction: execute_sql[\"...\"]\nObservation: ...\nAnswer: ...",
  "error_recovery": true,
  "db_id": "concert_singer",
  "difficulty": "medium"
}
```

### 2. ReAct SFT

The model is trained to emit full ReAct trajectories:

```text
Thought -> Action -> Observation -> ... -> Answer
```

This stage is supervised. The target is the full trajectory, not just the final SQL.

### 3. GRPO

The RL stage does **not** consume gold trajectories. It samples multiple completions per prompt, extracts the final SQL, executes it, and optimizes a reward composed of:

- format adherence
- SQL validity
- structural overlap with gold SQL
- partial execution agreement
- final correctness

This branch is still experimental. Current results show that GRPO improves some local reward behavior but does not yet outperform the ReAct SFT policy used with fallback.

## Inference Modes

### Direct

Generate one SQL query directly and execute it.

### Interactive ReAct

Generate step by step. After each `Action`, execute the SQL, inject the real `Observation`, and continue generation.

### Direct + Fallback

This is the current best deployment strategy:

1. Try direct SQL generation first
2. Execute on the real Spider DB
3. If execution fails, switch to interactive ReAct correction

This preserves the speed of direct decoding on easy questions and the robustness of agent-style correction on harder ones.

## Repository Map

```text
miniLLM/
├── miniLLM/agent/
│   ├── env.py              # SQLite execution environment
│   ├── react.py            # ReAct prompt format and trajectory parsing
│   ├── reward.py           # GRPO rewards
│   └── data_gen.py         # Teacher-generated ReAct trajectories
├── miniLLM/train.py        # Baseline SQL SFT
├── miniLLM/train_react_sft.py
├── miniLLM/train_grpo.py
├── miniLLM/eval.py         # Baseline SQL eval
├── miniLLM/eval_agent.py   # Agent eval: single-pass / interactive
├── miniLLM/eval_spider.py
├── scripts/gen_react_data.sh
├── scripts/train_react_sft.sh
├── scripts/train_grpo.sh
├── scripts/eval_agent.sh
└── docs/
    ├── REPORT.md
    └── AGENT.md
```

## Quickstart

### Environment

```bash
uv sync
```

If you want the full agent stack:

```bash
uv pip install deepspeed openai
```

### Baseline SQL SFT

```bash
bash scripts/train.sh
bash scripts/eval.sh
```

### Generate ReAct trajectories

```bash
API_KEY=... bash scripts/gen_react_data.sh
```

### Train ReAct SFT

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct \
DATA_PATH=outputs/react_trajectories.jsonl \
MAX_STEPS=500 \
bash scripts/train_react_sft.sh
```

### Evaluate the agent

Single-pass:

```bash
ADAPTER=outputs/react-sft bash scripts/eval_agent.sh
```

Interactive:

```bash
ADAPTER=outputs/react-sft INTERACTIVE=1 bash scripts/eval_agent.sh
```

### Train GRPO

```bash
SFT_ADAPTER=outputs/react-sft \
MAX_STEPS=300 \
bash scripts/train_grpo.sh
```

Use this as an experimental branch, not as the default production recipe.

## Recommended Reading Order

- [docs/REPORT.md](docs/REPORT.md): experimental results and failure analysis
- [docs/AGENT.md](docs/AGENT.md): architecture, file map, and training workflow
- [docs/LEARNING.md](docs/LEARNING.md): implementation notes and follow-ups

## Practical Notes

- The repo is optimized for single-GPU workstations with `24GB` VRAM class hardware
- DeepSpeed ZeRO-2 is used instead of Trainer wrappers so the training loop is explicit
- Evaluation on Spider should be interpreted carefully:
  attempted-only EX can overstate performance if many samples end in execution errors
- Current GRPO work is valuable mainly as a negative result and debugging surface, not as the strongest shipped model path
