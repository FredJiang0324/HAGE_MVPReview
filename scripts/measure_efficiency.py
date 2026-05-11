#!/usr/bin/env python3
"""
Efficiency Metrics: Tokens/Query and Latency by Category.

Extracts tokens per query (k) and latency (s) from existing result files.
Can also re-run sample 0 with different top_k values to measure the effect.

Usage:
    # Extract from existing results (no re-run needed)
    python scripts/measure_efficiency.py --results-dir results_gpt_4o_mini --sample 0

    # Compare across models
    python scripts/measure_efficiency.py --results-dir results_gpt_4o_mini --sample 0
    python scripts/measure_efficiency.py --results-dir results_qwen2_5:3b --sample 0

    # Re-run sample 0 with specific top_k values to measure efficiency at different retrieval depths
    python scripts/measure_efficiency.py --rerun --top-k-values 5,10,15,20,30 --model gpt-4o-mini --sample 0
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CAT_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}


def extract_efficiency_from_results(results_dir, sample_id):
    """Extract tokens/query and latency from an existing result file."""
    result_file = Path(results_dir) / f"fixed_results_sample{sample_id}.json"
    if not result_file.exists():
        logger.error(f"Result file not found: {result_file}")
        return None

    with open(result_file) as f:
        data = json.load(f)

    results = data["results"]

    # Per-category stats
    cat_stats = defaultdict(lambda: {
        "count": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "answer_latencies": [],
        "retrieval_latencies": [],
        "context_nodes_list": [],
    })

    for r in results:
        cat = r["category"]
        s = cat_stats[cat]
        s["count"] += 1

        tu = r.get("token_usage", {})
        s["total_prompt_tokens"] += tu.get("prompt_tokens", 0)
        s["total_completion_tokens"] += tu.get("completion_tokens", 0)
        s["total_tokens"] += tu.get("total_tokens", 0)

        s["answer_latencies"].append(r.get("answer_latency_seconds", 0))
        s["retrieval_latencies"].append(r.get("retrieval_latency_seconds", 0))
        s["context_nodes_list"].append(r.get("context_nodes", 0))

    return cat_stats


def print_efficiency_table(cat_stats, title=""):
    """Print formatted efficiency table."""
    print(f"\n{'='*80}")
    print(f"EFFICIENCY METRICS{': ' + title if title else ''}")
    print(f"{'='*80}")
    print(f"{'Cat':<14s} | {'#Q':>4s} | {'Nodes':>5s} | {'Tokens/Q (k)':>13s} | "
          f"{'Retrieval (s)':>13s} | {'Answer (s)':>11s} | {'Total (s)':>10s}")
    print("-" * 80)

    all_tokens = []
    all_retrieval = []
    all_answer = []
    all_count = 0

    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        n = s["count"]
        if n == 0:
            continue

        avg_tokens_k = (s["total_tokens"] / n) / 1000
        avg_retrieval = np.mean(s["retrieval_latencies"])
        avg_answer = np.mean(s["answer_latencies"])
        avg_total = avg_retrieval + avg_answer
        avg_nodes = np.mean(s["context_nodes_list"])

        cat_name = CAT_NAMES.get(cat, f"Cat {cat}")
        print(f"{cat_name:<14s} | {n:4d} | {avg_nodes:5.0f} | {avg_tokens_k:13.1f} | "
              f"{avg_retrieval:13.2f} | {avg_answer:11.2f} | {avg_total:10.2f}")

        all_tokens.extend([s["total_tokens"] / n] * n)
        all_retrieval.extend(s["retrieval_latencies"])
        all_answer.extend(s["answer_latencies"])
        all_count += n

    if all_count > 0:
        print("-" * 80)
        avg_tok = np.mean(all_tokens) / 1000
        avg_ret = np.mean(all_retrieval)
        avg_ans = np.mean(all_answer)
        print(f"{'Overall':<14s} | {all_count:4d} | {'':>5s} | {avg_tok:13.1f} | "
              f"{avg_ret:13.2f} | {avg_ans:11.2f} | {avg_ret + avg_ans:10.2f}")

    return {
        "total_questions": all_count,
        "avg_tokens_per_query_k": float(np.mean(all_tokens) / 1000) if all_tokens else 0,
        "avg_retrieval_latency_s": float(np.mean(all_retrieval)) if all_retrieval else 0,
        "avg_answer_latency_s": float(np.mean(all_answer)) if all_answer else 0,
    }


def rerun_with_top_k(top_k_values, args):
    """Re-run sample 0 with different max_top_k values to measure efficiency."""
    results = []

    for top_k in top_k_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running with max_top_k={top_k}")
        logger.info(f"{'='*60}")

        # Use a temp output dir to avoid overwriting main results
        temp_results_dir = Path(args.output_dir) / f"topk_{top_k}"
        temp_results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "test_fixed_memory.py",
            "--sample", str(args.sample),
            "--max-top-k", str(top_k),
            "--model", args.model,
        ]

        if args.backend:
            cmd.extend(["--backend", args.backend])

        logger.info(f"Running: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

        if proc.returncode != 0:
            logger.error(f"Failed for top_k={top_k}: {proc.stderr[-1000:]}")
            continue

        # Copy result file to temp dir
        model_name = args.model.replace(".", "_").replace("-", "_").replace(":", "_")
        src = Path(f"results_{model_name}") / f"fixed_results_sample{args.sample}.json"
        if src.exists():
            import shutil
            dst = temp_results_dir / f"fixed_results_sample{args.sample}.json"
            shutil.copy2(str(src), str(dst))

            # Extract metrics
            cat_stats = extract_efficiency_from_results(str(temp_results_dir), args.sample)
            if cat_stats:
                summary = print_efficiency_table(cat_stats, title=f"top_k={top_k}")
                summary["top_k"] = top_k
                results.append(summary)

    # Print comparison table
    if results:
        print(f"\n{'='*60}")
        print("TOP-K EFFICIENCY COMPARISON")
        print(f"{'='*60}")
        print(f"{'top_k':>6s} | {'Tokens/Q (k)':>13s} | {'Retrieval (s)':>13s} | {'Answer (s)':>11s} | {'Total (s)':>10s}")
        print("-" * 60)
        for r in results:
            total = r["avg_retrieval_latency_s"] + r["avg_answer_latency_s"]
            print(f"{r['top_k']:6d} | {r['avg_tokens_per_query_k']:13.1f} | "
                  f"{r['avg_retrieval_latency_s']:13.2f} | {r['avg_answer_latency_s']:11.2f} | "
                  f"{total:10.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure efficiency metrics")
    parser.add_argument("--results-dir", type=str, default="results_gpt_4o_mini",
                        help="Directory with existing result files")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample ID to analyze")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run with different top_k values")
    parser.add_argument("--top-k-values", type=str, default="5,10,15,20,30",
                        help="top_k values to test (comma-separated, with --rerun)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model for re-run")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["openai", "ollama"])
    parser.add_argument("--output-dir", type=str, default="results_efficiency",
                        help="Output directory for re-run results")

    args = parser.parse_args()

    if args.rerun:
        top_k_values = [int(x) for x in args.top_k_values.split(",")]
        all_results = rerun_with_top_k(top_k_values, args)

        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "efficiency_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {output_dir / 'efficiency_results.json'}")
    else:
        # Extract from existing results
        cat_stats = extract_efficiency_from_results(args.results_dir, args.sample)
        if cat_stats:
            summary = print_efficiency_table(cat_stats, title=f"{args.results_dir} sample {args.sample}")

            # Also output JSON
            output = {"per_category": {}, "overall": summary}
            for cat in sorted(cat_stats.keys()):
                s = cat_stats[cat]
                n = s["count"]
                if n > 0:
                    output["per_category"][str(cat)] = {
                        "name": CAT_NAMES.get(cat, f"Cat {cat}"),
                        "count": n,
                        "avg_tokens_k": round(s["total_tokens"] / n / 1000, 1),
                        "avg_retrieval_s": round(np.mean(s["retrieval_latencies"]), 2),
                        "avg_answer_s": round(np.mean(s["answer_latencies"]), 2),
                        "avg_nodes": round(np.mean(s["context_nodes_list"]), 1),
                    }
            print(f"\nJSON:\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    sys.exit(main())
