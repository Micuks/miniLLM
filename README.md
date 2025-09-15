miniLLM — Text-to-SQL Fine-tune + Infra Kickstart

Overview

This repository kickstarts a practical "Algorithm + Infra" project:
- Fine-tune a 7B–8B instruct model for Text-to-SQL with QLoRA
- Provide a baseline FastAPI inference service
- Prepare for later optimization with vLLM

Hardware & Software

- GPU: 24GB VRAM recommended
- OS: Linux
- Python: 3.10+
- CUDA: 12.1+ (for GPU inference/training)

Quickstart

1) Manage env and deps with uv (preferred)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

2) Train with QLoRA (SFT)

```bash
# Default uses Qwen/Qwen2.5-7B-Instruct and b-mc2/sql-create-context
bash scripts/train.sh
```

3) Evaluate on 20 samples

```bash
bash scripts/eval.sh
```

4) Serve a baseline API (FastAPI + Transformers)

```bash
bash scripts/serve.sh
# POST http://localhost:8000/generate_sql with JSON:
# {"schema": "CREATE TABLE ...", "question": "..."}
```

Dataset (recommended start)

- b-mc2/sql-create-context — schema-in-context Text-to-SQL dataset

Notes

- Default base model is Qwen/Qwen2.5-7B-Instruct to avoid gated access; you can switch to Meta Llama 3 8B Instruct by passing --model-name-or-path.
- If running GPU training, ensure the correct CUDA-enabled torch is installed. See: https://download.pytorch.org/whl/
- You can still use pip: bash scripts/install.sh


# miniLLM