from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..prompts import build_inference_prompt


app = FastAPI(title="miniLLM Text-to-SQL API")


class GenerateRequest(BaseModel):
    schema: str
    question: str
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: str | None = None


class GenerateResponse(BaseModel):
    sql: str


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


@app.post("/generate_sql", response_model=GenerateResponse)
def generate_sql(req: GenerateRequest) -> GenerateResponse:
    tokenizer, model = _load_model_bundle(req.model_name_or_path, req.adapter_path)

    prompt = build_inference_prompt(req.schema, req.question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return GenerateResponse(sql=text.strip())
