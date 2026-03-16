"""vLLM-powered Text-to-SQL serving with AsyncLLMEngine, multi-LoRA, and streaming."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest

from ..prompts import build_inference_prompt, SYSTEM_PROMPT
from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOKENS_GENERATED,
    ACTIVE_REQUESTS,
    track_gpu_memory,
)
from .adapter_registry import AdapterRegistry

logger = logging.getLogger(__name__)

# ── Global state ──
engine: AsyncLLMEngine | None = None
tokenizer = None
adapter_registry = AdapterRegistry()


def _build_engine_args() -> AsyncEngineArgs:
    model = os.getenv("MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-7B-Instruct")
    return AsyncEngineArgs(
        model=model,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "2048")),
        enable_lora=True,
        max_loras=int(os.getenv("MAX_LORAS", "4")),
        max_lora_rank=int(os.getenv("MAX_LORA_RANK", "64")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tokenizer
    args = _build_engine_args()
    engine = AsyncLLMEngine.from_engine_args(args)
    # Retrieve tokenizer from the engine
    engine_model_config = await engine.get_model_config()
    from transformers import AutoTokenizer as _AT
    tokenizer = _AT.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("vLLM engine ready: %s", args.model)
    yield
    logger.info("Shutting down vLLM engine")


app = FastAPI(title="miniLLM vLLM Text-to-SQL API", lifespan=lifespan)

# ── Request / response schemas ──

class GenerateRequest(BaseModel):
    schema_ddl: str
    question: str
    adapter_name: str | None = None
    max_tokens: int = 256
    temperature: float = 0.0


class GenerateResponse(BaseModel):
    sql: str
    tokens_generated: int
    latency_ms: float


# ── Health endpoints ──

_startup_complete = False


@app.get("/health/live")
async def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    return {"status": "ready"}


@app.get("/health/startup")
async def health_startup():
    if engine is None:
        raise HTTPException(503, "Engine starting")
    return {"status": "started"}


# ── Adapter management ──

class AdapterLoadRequest(BaseModel):
    name: str
    path: str


@app.post("/adapters")
async def load_adapter(req: AdapterLoadRequest):
    adapter_registry.register(req.name, req.path)
    return {"status": "loaded", "name": req.name}


@app.delete("/adapters/{name}")
async def unload_adapter(name: str):
    adapter_registry.unregister(name)
    return {"status": "unloaded", "name": name}


@app.get("/adapters")
async def list_adapters():
    return {"adapters": adapter_registry.list_adapters()}


# ── Core generation ──

async def _generate(request_id: str, prompt: str, req: GenerateRequest) -> tuple[str, int]:
    """Run vLLM generation, return (text, token_count)."""
    sampling = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    lora_request = None
    if req.adapter_name:
        adapter_info = adapter_registry.get(req.adapter_name)
        lora_request = LoRARequest(
            lora_name=adapter_info["name"],
            lora_int_id=adapter_info["id"],
            lora_local_path=adapter_info["path"],
        )

    results_generator = engine.generate(prompt, sampling, request_id, lora_request=lora_request)
    final_output = None
    async for output in results_generator:
        final_output = output

    text = final_output.outputs[0].text
    n_tokens = len(final_output.outputs[0].token_ids)
    return text, n_tokens


@app.post("/generate_sql", response_model=GenerateResponse)
async def generate_sql(req: GenerateRequest):
    request_id = f"req-{time.monotonic_ns()}"
    prompt = build_inference_prompt(req.schema_ddl, req.question, tokenizer)

    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()
    t0 = time.perf_counter()
    try:
        text, n_tokens = await _generate(request_id, prompt, req)
    finally:
        ACTIVE_REQUESTS.dec()
    latency = (time.perf_counter() - t0) * 1000
    REQUEST_LATENCY.observe(latency / 1000)
    TOKENS_GENERATED.inc(n_tokens)
    track_gpu_memory()

    return GenerateResponse(sql=text.strip(), tokens_generated=n_tokens, latency_ms=latency)


@app.post("/generate_sql/stream")
async def generate_sql_stream(req: GenerateRequest):
    """Streaming SSE endpoint for SQL generation."""
    request_id = f"req-{time.monotonic_ns()}"
    prompt = build_inference_prompt(req.schema_ddl, req.question, tokenizer)

    sampling = SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens)

    lora_request = None
    if req.adapter_name:
        adapter_info = adapter_registry.get(req.adapter_name)
        lora_request = LoRARequest(
            lora_name=adapter_info["name"],
            lora_int_id=adapter_info["id"],
            lora_local_path=adapter_info["path"],
        )

    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()

    async def event_generator():
        prev_text = ""
        try:
            async for output in engine.generate(prompt, sampling, request_id, lora_request=lora_request):
                current_text = output.outputs[0].text
                delta = current_text[len(prev_text):]
                if delta:
                    yield f"data: {delta}\n\n"
                prev_text = current_text
            yield "data: [DONE]\n\n"
        finally:
            ACTIVE_REQUESTS.dec()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Prometheus metrics endpoint ──

@app.get("/metrics")
async def metrics():
    from .metrics import generate_metrics_text
    return StreamingResponse(
        iter([generate_metrics_text()]),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
