"""Generate a combined quantization benchmark report from individual results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_report(result_files: list[str], output_path: str) -> dict:
    """Merge individual benchmark JSONs into a combined report with markdown table."""
    results = []
    for f in result_files:
        with open(f) as fh:
            results.append(json.load(fh))

    # Build markdown summary table
    header = (
        "| Variant | Size (MB) | GPU Mem (MB) | Latency Mean (ms) | P50 (ms) | P95 (ms) "
        "| P99 (ms) | Throughput (tok/s) | EM | EX |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    rows = [header, sep]
    for r in results:
        ex = f"{r['execution_match']:.3f}" if r.get("execution_match") is not None else "N/A"
        rows.append(
            f"| {r['variant']} | {r['model_size_mb']:.0f} | {r['gpu_memory_mb']:.0f} "
            f"| {r['latency_mean_ms']:.1f} | {r['latency_p50_ms']:.1f} "
            f"| {r['latency_p95_ms']:.1f} | {r['latency_p99_ms']:.1f} "
            f"| {r['throughput_tok_per_s']:.1f} | {r['exact_match']:.3f} | {ex} |"
        )
    markdown_table = "\n".join(rows)

    report = {
        "variants": results,
        "markdown_summary": markdown_table,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(markdown_table)
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate quantization benchmark report")
    parser.add_argument("result_files", nargs="+", help="Paths to individual benchmark JSON files")
    parser.add_argument("--output", type=str, default="outputs/quant_report.json")
    args = parser.parse_args()
    generate_report(args.result_files, args.output)


if __name__ == "__main__":
    main()
