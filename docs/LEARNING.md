Learning path with this repo

Goals

- Understand QLoRA fine-tuning flow on a Text-to-SQL dataset
- Learn to serve an LLM via FastAPI and compare outputs before/after SFT

Hands-on steps

1) Inspect the dataset and prompts
- Read `miniLLM/prompts.py` and understand the format for supervised fine-tuning and inference
- Explore `b-mc2/sql-create-context` with a quick snippet or the eval script

2) Run a tiny SFT
- Use `bash scripts/train.sh` to run ~100 steps SFT (LoRA on 4-bit base)
- Skim `miniLLM/train.py` to see TRL `SFTTrainer`, LoRA, and 4-bit config

3) Evaluate baseline vs tuned
- Run `bash scripts/eval.sh` and observe predicted SQL vs gold SQL for 20 samples
- Enable `WITH_EXECUTION=1` for optional sqlite execution match
- Compare pre-tuned base model vs after applying LoRA adapters via `ADAPTER_PATH=...`

4) Serve and iterate
- `bash scripts/serve.sh` to start a simple API for SQL generation
- POST schema + question to see results quickly

What to read next (short list)

- TRL `SFTTrainer` docs and examples
- PEFT LoRA design and target modules for decoder-only models
- b-mc2/sql-create-context dataset card for schema context design

Ideas for extension

- Replace base model with Meta Llama 3 8B Instruct and re-run
- Add LoRA merge/export to produce a single deployable model folder
- Add robust SQL canonicalization (AST-based) for stricter EM scoring
- Swap baseline inference with vLLM and benchmark latency/throughput



