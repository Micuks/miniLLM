"""Throughput benchmark: compare HF vs vLLM backends at various concurrency levels."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
from datasets import load_dataset


@dataclass
class LatencyStats:
    ttft_ms: float
    e2e_latency_ms: float
    throughput_req_per_s: float


@dataclass
class BenchResult:
    backend: str
    concurrency: int
    n_requests: int
    ttft_mean_ms: float
    e2e_mean_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    throughput_req_per_s: float


async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    streaming: bool = False,
) -> tuple[float, float]:
    """Send a single request, return (ttft_ms, e2e_latency_ms)."""
    t0 = time.perf_counter()
    ttft = None

    if streaming:
        async with session.post(url, json=payload) as resp:
            async for line in resp.content:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                decoded = line.decode().strip()
                if decoded == "data: [DONE]":
                    break
    else:
        async with session.post(url, json=payload) as resp:
            await resp.json()
            ttft = (time.perf_counter() - t0) * 1000

    e2e = (time.perf_counter() - t0) * 1000
    return ttft or e2e, e2e


async def bench_backend(
    base_url: str,
    endpoint: str,
    payloads: list[dict],
    concurrency: int,
    streaming: bool = False,
) -> BenchResult:
    """Benchmark a backend at a given concurrency level."""
    url = f"{base_url}{endpoint}"
    sem = asyncio.Semaphore(concurrency)
    ttfts: list[float] = []
    e2es: list[float] = []

    async with aiohttp.ClientSession() as session:

        async def bounded_request(payload):
            async with sem:
                ttft, e2e = await _send_request(session, url, payload, streaming)
                ttfts.append(ttft)
                e2es.append(e2e)

        t0 = time.perf_counter()
        await asyncio.gather(*[bounded_request(p) for p in payloads])
        total_time = time.perf_counter() - t0

    e2es.sort()
    n = len(e2es)

    return BenchResult(
        backend=base_url,
        concurrency=concurrency,
        n_requests=n,
        ttft_mean_ms=sum(ttfts) / n,
        e2e_mean_ms=sum(e2es) / n,
        e2e_p50_ms=e2es[n // 2],
        e2e_p95_ms=e2es[int(n * 0.95)],
        throughput_req_per_s=n / total_time,
    )


def load_bench_payloads(
    dataset_name: str = "b-mc2/sql-create-context",
    n_samples: int = 50,
    use_vllm_schema: bool = False,
) -> list[dict]:
    """Load payloads for benchmarking."""
    ds = load_dataset(dataset_name, split="train").select(range(n_samples))
    payloads = []
    for sample in ds:
        if use_vllm_schema:
            payloads.append({
                "schema_ddl": sample.get("context", ""),
                "question": sample.get("question", ""),
            })
        else:
            payloads.append({
                "schema": sample.get("context", ""),
                "question": sample.get("question", ""),
            })
    return payloads


async def run_full_bench(
    hf_url: str,
    vllm_url: str,
    n_samples: int = 50,
    concurrency_levels: list[int] | None = None,
) -> list[BenchResult]:
    """Run full benchmark comparing HF and vLLM at multiple concurrency levels."""
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16, 32]

    hf_payloads = load_bench_payloads(n_samples=n_samples, use_vllm_schema=False)
    vllm_payloads = load_bench_payloads(n_samples=n_samples, use_vllm_schema=True)

    results = []
    for c in concurrency_levels:
        print(f"Benchmarking HF at concurrency={c}...")
        r = await bench_backend(hf_url, "/generate_sql", hf_payloads, c)
        results.append(r)

        print(f"Benchmarking vLLM at concurrency={c}...")
        r = await bench_backend(vllm_url, "/generate_sql", vllm_payloads, c)
        results.append(r)

        print(f"Benchmarking vLLM streaming at concurrency={c}...")
        r = await bench_backend(vllm_url, "/generate_sql/stream", vllm_payloads, c, streaming=True)
        results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(description="HF vs vLLM throughput benchmark")
    parser.add_argument("--hf-url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8001")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument(
        "--concurrency", type=str, default="1,4,8,16,32",
        help="Comma-separated concurrency levels"
    )
    parser.add_argument("--output", type=str, default="outputs/throughput_bench.json")
    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]

    results = asyncio.run(
        run_full_bench(args.hf_url, args.vllm_url, args.n_samples, concurrency_levels)
    )

    output = [asdict(r) for r in results]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    # Print summary table
    print(f"\n{'Backend':<40} {'Conc':>5} {'TTFT(ms)':>10} {'E2E(ms)':>10} "
          f"{'P50(ms)':>10} {'P95(ms)':>10} {'RPS':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r.backend:<40} {r.concurrency:>5} {r.ttft_mean_ms:>10.1f} "
              f"{r.e2e_mean_ms:>10.1f} {r.e2e_p50_ms:>10.1f} {r.e2e_p95_ms:>10.1f} "
              f"{r.throughput_req_per_s:>8.1f}")

    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
