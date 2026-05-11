#!/usr/bin/env python3
"""
Phase 2 Benchmark: Compare all scorers

Compares:
  1. Random baseline — uniform random neighbor selection
  2. Cosine-only baseline — pick highest cosine-similarity neighbor (no edge weights)
  3. RL Inductive — trained QueryRouter (cross-graph)
  4. RL Transductive — trained per-edge embeddings (sample 0)

Metrics:
  - Success Rate (target node reached within max_hops)
  - Average Return
  - Average Steps
  - Inference Speed (ms/query)

Usage:
    python scripts/evaluate_phase2.py
    python scripts/evaluate_phase2.py --max-hops 5 --device cpu
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.pytorch_graph import load_all_graphs, QADataset
from memory.model import CoEvoMem, build_transductive_model, build_inductive_model
from memory.env import GraphTraversalEnv, run_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Baseline models
# ======================================================================

class RandomModel:
    """Uniform random neighbor selection."""
    mode = "random"

    def get_action(self, graph, current_node, query_emb, greedy=False):
        neighbors = graph.neighbors(current_node)
        if not neighbors:
            return -1, torch.tensor(0.0), torch.zeros(0)
        n = len(neighbors)
        probs = torch.ones(n) / n
        idx = torch.randint(0, n, (1,)).item()
        log_prob = torch.tensor(-np.log(n))
        return idx, log_prob, probs


class CosineOnlyModel:
    """Pick neighbor with highest cosine similarity to query (no edge weights)."""
    mode = "cosine_only"

    def get_action(self, graph, current_node, query_emb, greedy=False):
        neighbors = graph.neighbors(current_node)
        if not neighbors:
            return -1, torch.tensor(0.0), torch.zeros(0)
        nbr_indices = torch.tensor([n for n, _ in neighbors], dtype=torch.long, device=graph.device)
        nbr_embs = graph.node_embeddings[nbr_indices]
        sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), nbr_embs, dim=-1)
        probs = torch.softmax(sims, dim=0)
        if greedy:
            idx = probs.argmax().item()
        else:
            idx = torch.distributions.Categorical(probs).sample().item()
        log_prob = probs[idx].log()
        return idx, log_prob, probs


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_model(model, dataset, env, device, greedy=True):
    """Evaluate a model on a dataset. Returns metrics dict."""
    successes = 0
    total_reward = 0.0
    total_steps = 0
    count = 0
    times = []

    if hasattr(model, 'eval'):
        model.eval()

    with torch.no_grad():
        for item in dataset.items:
            if item["query_embedding"] is None:
                continue

            graph = item["graph"]
            query_emb = item["query_embedding"].to(device)
            targets = item["target_node_indices"]

            t0 = time.perf_counter()
            rewards, _, success, steps = run_episode(
                model, env, graph, query_emb, targets, greedy=greedy
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if success:
                successes += 1
            total_reward += sum(rewards)
            total_steps += steps
            count += 1
            times.append(elapsed_ms)

    if count == 0:
        return {"success_rate": 0, "avg_reward": 0, "avg_steps": 0, "count": 0, "ms_per_query": 0}

    return {
        "success_rate": successes / count,
        "avg_reward": total_reward / count,
        "avg_steps": total_steps / count,
        "count": count,
        "ms_per_query": np.mean(times),
        "total_successes": successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Benchmark")
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inductive-ckpt", type=str, default="checkpoints/inductive/model_final.pt")
    parser.add_argument("--transductive-ckpt", type=str, default="checkpoints/transductive_s0/model_final.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load data ----
    logger.info("Loading graphs and QA data...")
    graphs = load_all_graphs(device=args.device)
    locomo = json.load(open("data/locomo10.json"))
    for i, s in enumerate(locomo):
        s["sample_id"] = i

    ds = QADataset(locomo, graphs)

    from memory.vector_db import VectorEncoder
    encoder = VectorEncoder(model_name="all-MiniLM-L6-v2", use_openai=False)
    ds.encoder = encoder
    ds.encode_queries()

    train_ids = list(range(8))
    val_ids = [8, 9]
    train_ds, val_ds = ds.split_by_sample(train_ids, val_ids)

    # Also get sample-0-only dataset for transductive eval
    s0_ds, _ = ds.split_by_sample([0], [])

    env = GraphTraversalEnv(max_hops=args.max_hops)

    # ---- Build models ----
    models = {}

    # 1. Random baseline
    models["Random"] = RandomModel()

    # 2. Cosine-only baseline
    models["Cosine-Only"] = CosineOnlyModel()

    # 3. RL Inductive
    if os.path.exists(args.inductive_ckpt):
        ind_model = build_inductive_model(device=args.device)
        ckpt = torch.load(args.inductive_ckpt, map_location=args.device, weights_only=False)
        ind_model.load_state_dict(ckpt["model_state_dict"])
        models["RL Inductive"] = ind_model
        logger.info(f"Loaded inductive model from {args.inductive_ckpt}")
    else:
        logger.warning(f"Inductive checkpoint not found: {args.inductive_ckpt}")

    # 4. RL Transductive (sample 0 only)
    if os.path.exists(args.transductive_ckpt):
        g0 = next(g for g in graphs if g.sample_id == 0)
        trans_model = build_transductive_model(g0, device=args.device)
        ckpt = torch.load(args.transductive_ckpt, map_location=args.device, weights_only=False)
        trans_model.load_state_dict(ckpt["model_state_dict"])
        models["RL Transductive (S0)"] = trans_model
        logger.info(f"Loaded transductive model from {args.transductive_ckpt}")
    else:
        logger.warning(f"Transductive checkpoint not found: {args.transductive_ckpt}")

    # ---- Evaluate ----
    results = {}

    # Evaluate on train set (samples 0-7)
    print("\n" + "=" * 80)
    print("EVALUATION ON TRAIN SAMPLES (0-7)")
    print("=" * 80)
    for name, model in models.items():
        if name == "RL Transductive (S0)":
            continue  # Only valid on sample 0
        metrics = evaluate_model(model, train_ds, env, args.device)
        results[f"{name} (train)"] = metrics
        print(f"  {name:25s} | SR: {metrics['success_rate']:.3f} | "
              f"Reward: {metrics['avg_reward']:+.2f} | "
              f"Steps: {metrics['avg_steps']:.1f} | "
              f"Speed: {metrics['ms_per_query']:.2f} ms/q | "
              f"N={metrics['count']}")

    # Evaluate on val set (samples 8-9) — zero-shot
    print("\n" + "=" * 80)
    print("EVALUATION ON VAL SAMPLES (8-9) — ZERO-SHOT")
    print("=" * 80)
    for name, model in models.items():
        if name == "RL Transductive (S0)":
            continue  # Only valid on sample 0
        metrics = evaluate_model(model, val_ds, env, args.device)
        results[f"{name} (val)"] = metrics
        print(f"  {name:25s} | SR: {metrics['success_rate']:.3f} | "
              f"Reward: {metrics['avg_reward']:+.2f} | "
              f"Steps: {metrics['avg_steps']:.1f} | "
              f"Speed: {metrics['ms_per_query']:.2f} ms/q | "
              f"N={metrics['count']}")

    # Evaluate transductive on sample 0 only
    if "RL Transductive (S0)" in models:
        print("\n" + "=" * 80)
        print("EVALUATION ON SAMPLE 0 ONLY (Transductive vs others)")
        print("=" * 80)
        for name, model in models.items():
            metrics = evaluate_model(model, s0_ds, env, args.device)
            results[f"{name} (S0)"] = metrics
            print(f"  {name:25s} | SR: {metrics['success_rate']:.3f} | "
                  f"Reward: {metrics['avg_reward']:+.2f} | "
                  f"Steps: {metrics['avg_steps']:.1f} | "
                  f"Speed: {metrics['ms_per_query']:.2f} ms/q | "
                  f"N={metrics['count']}")

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Scorer':<30s} | {'Split':>8s} | {'SR':>6s} | {'Reward':>7s} | {'Steps':>5s} | {'ms/q':>6s}")
    print("  " + "-" * 75)
    for key, m in results.items():
        print(f"  {key:<30s} | {'':>8s} | {m['success_rate']:>5.1%} | {m['avg_reward']:>+6.2f} | {m['avg_steps']:>5.1f} | {m['ms_per_query']:>5.2f}")

    # ---- Save results ----
    output_path = "checkpoints/phase2_benchmark.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert for JSON serialization
    serializable = {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv) if isinstance(vv, (np.integer, int)) else vv for kk, vv in v.items()} for k, v in results.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
