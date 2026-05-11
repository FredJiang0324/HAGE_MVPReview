#!/usr/bin/env python3
"""
Co-Evolution Ablation Study: Router-Only vs Edge-Only vs Joint

Runs three inductive-style experiments on the same 8/2 train/val split:
  1. router_only  — freeze edge features, train QueryRouter (= current inductive)
  2. edge_only    — freeze router, train edge features via RL
  3. coevo        — jointly train both (co-evolution)

Plus two non-learned baselines:
  - random        — random neighbor selection
  - cosine_only   — pick highest cosine-similarity neighbor

Usage:
    python scripts/run_coevo_ablation.py --epochs 100
    python scripts/run_coevo_ablation.py --epochs 50 --device cuda
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.pytorch_graph import load_all_graphs, QADataset
from memory.model import CoEvoMem, build_inductive_model, build_coevo_model, QueryRouter
from memory.rl_trainer import RLTrainer
from memory.env import GraphTraversalEnv, run_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Baselines (no learning) ───────────────────────────────────────────

def evaluate_baseline(
    scorer_name: str,
    dataset: QADataset,
    max_hops: int = 5,
) -> Dict:
    """Evaluate a non-learned baseline scorer."""
    env = GraphTraversalEnv(max_hops=max_hops)
    successes = 0
    total_reward = 0.0
    total_steps = 0
    count = 0

    for item in dataset.items:
        if item["query_embedding"] is None:
            continue

        graph = item["graph"]
        query_emb = item["query_embedding"]
        targets = item["target_node_indices"]

        env.reset(graph, query_emb, targets)
        if env.done:
            if env.targets_found == env.target_set:
                successes += 1
            count += 1
            continue

        episode_reward = 0.0
        episode_success = False
        while not env.done:
            neighbors = graph.neighbors(env.current_node)
            if not neighbors:
                env.done = True
                episode_reward += env.rc.dead_end_penalty
                break

            if scorer_name == "random":
                action_idx = np.random.randint(len(neighbors))
            elif scorer_name == "cosine_only":
                nbr_indices = torch.tensor([n for n, _ in neighbors], device=graph.device)
                nbr_embs = graph.node_embeddings[nbr_indices]
                sims = torch.nn.functional.cosine_similarity(
                    query_emb.unsqueeze(0), nbr_embs, dim=-1
                )
                action_idx = sims.argmax().item()
            else:
                raise ValueError(f"Unknown baseline: {scorer_name}")

            result = env.step(action_idx)
            episode_reward += result.reward
            if result.success:
                episode_success = True

        if episode_success:
            successes += 1
        total_reward += episode_reward
        total_steps += env.steps
        count += 1

    if count == 0:
        return {"success_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0, "count": 0}

    return {
        "success_rate": successes / count,
        "avg_reward": total_reward / count,
        "avg_steps": total_steps / count,
        "count": count,
    }


# ── Edge-Only model: freeze router, train only edge features ─────────

def build_edge_only_model(graphs: list, device: str = "cpu") -> CoEvoMem:
    """
    Build a coevo model but freeze the router — only edge features train.
    """
    model = build_coevo_model(graphs, device=device)
    # Freeze router parameters
    for param in model.query_router.parameters():
        param.requires_grad = False
    return model


# ── Router-Only: standard inductive (edge features frozen) ───────────
# This is just build_inductive_model() — edge features stay on the graph.


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Co-Evolution Ablation Study")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--edge-lr", type=float, default=1e-4,
                        help="LR for edge features in coevo/edge_only modes")
    parser.add_argument("--anchor-lambda", type=float, default=1.0,
                        help="L2 anchor regularization strength for coevo edge features")
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_coevo_ablation")
    parser.add_argument("--train-samples", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--val-samples", type=str, default="8,9")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
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

    train_ids = [int(x) for x in args.train_samples.split(",")]
    val_ids = [int(x) for x in args.val_samples.split(",")]
    train_ds, val_ds = ds.split_by_sample(train_ids, val_ids)
    train_graphs = [g for g in graphs if g.sample_id in train_ids]

    all_results = {}

    # ── 1. Baselines (no training) ──
    for baseline in ["random", "cosine_only"]:
        logger.info(f"\n{'='*60}\nEvaluating baseline: {baseline}\n{'='*60}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_eval = evaluate_baseline(baseline, train_ds, max_hops=args.max_hops)
        val_eval = evaluate_baseline(baseline, val_ds, max_hops=args.max_hops)

        all_results[baseline] = {
            "train": train_eval,
            "val": val_eval,
            "params": 0,
        }
        logger.info(
            f"{baseline:15s} | Train SR: {train_eval['success_rate']:.3f} | "
            f"Val SR: {val_eval['success_rate']:.3f}"
        )

    # ── 2. Router-Only (= inductive, frozen edge features) ──
    logger.info(f"\n{'='*60}\nTraining: router_only (inductive)\n{'='*60}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_router = build_inductive_model(device=args.device)
    trainer_router = RLTrainer(
        model=model_router, train_dataset=train_ds, val_dataset=val_ds,
        lr=args.lr, max_hops=args.max_hops, device=args.device,
    )
    history_router = trainer_router.train(
        num_epochs=args.epochs, eval_every=args.eval_every,
        checkpoint_dir=str(output_dir / "ckpt_router_only"),
    )
    train_eval = trainer_router.evaluate(train_ds, greedy=True)
    val_eval = trainer_router.evaluate(val_ds, greedy=True)
    n_params = sum(p.numel() for p in model_router.parameters() if p.requires_grad)
    all_results["router_only"] = {
        "train": train_eval, "val": val_eval,
        "params": n_params, "history": history_router,
    }
    logger.info(
        f"{'router_only':15s} | Train SR: {train_eval['success_rate']:.3f} | "
        f"Val SR: {val_eval['success_rate']:.3f} | Params: {n_params:,}"
    )

    # ── 3. Edge-Only (freeze router, train edge features) ──
    logger.info(f"\n{'='*60}\nTraining: edge_only\n{'='*60}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_edge = build_edge_only_model(train_graphs, device=args.device)
    # For edge_only, only edge features are trainable — use edge_lr
    trainer_edge = RLTrainer(
        model=model_edge, train_dataset=train_ds, val_dataset=val_ds,
        lr=args.edge_lr, max_hops=args.max_hops, device=args.device,
    )
    history_edge = trainer_edge.train(
        num_epochs=args.epochs, eval_every=args.eval_every,
        checkpoint_dir=str(output_dir / "ckpt_edge_only"),
    )
    train_eval = trainer_edge.evaluate(train_ds, greedy=True)
    val_eval = trainer_edge.evaluate(val_ds, greedy=True)
    n_params = sum(p.numel() for p in model_edge.parameters() if p.requires_grad)
    all_results["edge_only"] = {
        "train": train_eval, "val": val_eval,
        "params": n_params, "history": history_edge,
    }
    logger.info(
        f"{'edge_only':15s} | Train SR: {train_eval['success_rate']:.3f} | "
        f"Val SR: {val_eval['success_rate']:.3f} | Params: {n_params:,}"
    )

    # ── 4. CoEvo (jointly train edge features + router) ──
    logger.info(f"\n{'='*60}\nTraining: coevo (joint)\n{'='*60}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_coevo = build_coevo_model(train_graphs, device=args.device)
    trainer_coevo = RLTrainer(
        model=model_coevo, train_dataset=train_ds, val_dataset=val_ds,
        lr=args.lr, edge_lr=args.edge_lr, anchor_lambda=args.anchor_lambda,
        max_hops=args.max_hops, device=args.device,
    )
    history_coevo = trainer_coevo.train(
        num_epochs=args.epochs, eval_every=args.eval_every,
        checkpoint_dir=str(output_dir / "ckpt_coevo"),
    )
    train_eval = trainer_coevo.evaluate(train_ds, greedy=True)
    val_eval = trainer_coevo.evaluate(val_ds, greedy=True)
    n_params_router = sum(
        p.numel() for p in model_coevo.query_router.parameters() if p.requires_grad
    )
    n_params_edge = sum(
        p.numel() for p in model_coevo.edge_features_dict.parameters() if p.requires_grad
    )
    n_params_total = n_params_router + n_params_edge
    all_results["coevo"] = {
        "train": train_eval, "val": val_eval,
        "params": n_params_total,
        "params_router": n_params_router,
        "params_edge": n_params_edge,
        "history": history_coevo,
    }
    logger.info(
        f"{'coevo':15s} | Train SR: {train_eval['success_rate']:.3f} | "
        f"Val SR: {val_eval['success_rate']:.3f} | "
        f"Params: {n_params_total:,} (router={n_params_router:,}, edge={n_params_edge:,})"
    )

    # ── Summary Table ──
    logger.info(f"\n{'='*60}")
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Method':15s} | {'Train SR':>10s} | {'Val SR':>10s} | {'Params':>10s}")
    logger.info("-" * 55)
    for name in ["random", "cosine_only", "router_only", "edge_only", "coevo"]:
        r = all_results[name]
        logger.info(
            f"{name:15s} | "
            f"{r['train']['success_rate']:10.3f} | "
            f"{r['val']['success_rate']:10.3f} | "
            f"{r['params']:>10,}"
        )

    # ── Save results ──
    # Strip non-serializable history for JSON
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {
            "train": r["train"],
            "val": r["val"],
            "params": r.get("params", 0),
        }
        if "params_router" in r:
            save_results[name]["params_router"] = r["params_router"]
            save_results[name]["params_edge"] = r["params_edge"]

    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
