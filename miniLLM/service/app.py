from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..prompts import build_inference_prompt


app = FastAPI(title="miniLLM Text-to-SQL API")


class GenerateRequest(BaseModel):
    schema: str
    question: str
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"


class GenerateResponse(BaseModel):
    sql: str


@app.post("/generate_sql", response_model=GenerateResponse)
def generate_sql(req: GenerateRequest) -> GenerateResponse:
    tokenizer = AutoTokenizer.from_pretrained(req.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        req.model_name_or_path, device_map="auto", trust_remote_code=True
    )

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
    text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return GenerateResponse(sql=text.strip())



