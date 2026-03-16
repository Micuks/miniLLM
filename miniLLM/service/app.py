"""HuggingFace Transformers-based Text-to-SQL API with streaming, metrics, and health checks."""
from __future__ import annotations

import threading
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ..prompts import build_inference_prompt
from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOKENS_GENERATED,
    ACTIVE_REQUESTS,
    track_gpu_memory,
    generate_metrics_text,
)
from .adapter_registry import AdapterRegistry


app = FastAPI(title="miniLLM Text-to-SQL API")
adapter_registry = AdapterRegistry()


class GenerateRequest(BaseModel):
    schema: str
    question: str
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: str | None = None


class GenerateResponse(BaseModel):
    sql: str


# ── Model loading ──

@lru_cache(maxsize=4)
def _load_model_bundle(model_name_or_path: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    return tokenizer, model


# ── Health endpoints ──

@app.get("/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    return {"status": "ready"}


@app.get("/health/startup")
def health_startup():
    return {"status": "started"}


# ── Adapter management ──

class AdapterLoadRequest(BaseModel):
    name: str
    path: str


@app.post("/adapters")
def load_adapter(req: AdapterLoadRequest):
    adapter_registry.register(req.name, req.path)
    return {"status": "loaded", "name": req.name}


@app.delete("/adapters/{name}")
def unload_adapter(name: str):
    adapter_registry.unregister(name)
    return {"status": "unloaded", "name": name}


@app.get("/adapters")
def list_adapters():
    return {"adapters": adapter_registry.list_adapters()}


# ── Generation ──

@app.post("/generate_sql", response_model=GenerateResponse)
def generate_sql(req: GenerateRequest) -> GenerateResponse:
    tokenizer, model = _load_model_bundle(req.model_name_or_path, req.adapter_path)

    prompt = build_inference_prompt(req.schema, req.question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()
    t0 = time.perf_counter()
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        ACTIVE_REQUESTS.dec()

    latency = time.perf_counter() - t0
    REQUEST_LATENCY.observe(latency)
    n_new = output_ids.shape[1] - inputs["input_ids"].shape[1]
    TOKENS_GENERATED.inc(n_new)
    track_gpu_memory()

    text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return GenerateResponse(sql=text.strip())


@app.post("/generate_sql/stream")
def generate_sql_stream(req: GenerateRequest):
    """Streaming endpoint using TextIteratorStreamer."""
    tokenizer, model = _load_model_bundle(req.model_name_or_path, req.adapter_path)

    prompt = build_inference_prompt(req.schema, req.question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()

    gen_kwargs = {
        **inputs,
        "max_new_tokens": 256,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    def event_generator():
        try:
            for text in streamer:
                if text:
                    yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            ACTIVE_REQUESTS.dec()
            thread.join()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Metrics ──

@app.get("/metrics")
def metrics():
    return StreamingResponse(
        iter([generate_metrics_text()]),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
