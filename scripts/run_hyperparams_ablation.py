#!/usr/bin/env python3
"""
Hyperparameter Ablation for CoEvo Edge Learning.

Two ablation studies:

1. Anchor Regularization (λ_anchor):
   - λ_anchor ∈ {0.0, 0.1, 0.5, 1.0, 2.0}
   - All other hyperparams at default (lr=1e-3, edge_lr=1e-4, max_hops=5)

2. Max Hops:
   - max_hops ∈ {3, 5, 7}
   - All other hyperparams at default (anchor_lambda=1.0)

Both use 5-fold CV with the same fold definitions as run_5fold_cv.py.

Usage:
    # Run both ablations
    python scripts/run_hyperparams_ablation.py

    # Run only anchor lambda ablation
    python scripts/run_hyperparams_ablation.py --study anchor

    # Run only max hops ablation
    python scripts/run_hyperparams_ablation.py --study hops

    # Custom values
    python scripts/run_hyperparams_ablation.py --study anchor --anchor-lambdas 0.0,0.1,0.5,1.0,2.0,5.0
    python scripts/run_hyperparams_ablation.py --study hops --max-hops-list 2,3,5,7,10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.pytorch_graph import load_all_graphs, QADataset
from memory.model import build_coevo_model
from memory.rl_trainer import RLTrainer
from memory.env import GraphTraversalEnv, run_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Fold definitions (same as run_5fold_cv.py) ──
FOLDS = [
    {"fold": 1, "train": [2, 3, 4, 5, 6, 7, 8, 9], "val": [0, 1]},
    {"fold": 2, "train": [0, 1, 4, 5, 6, 7, 8, 9], "val": [2, 3]},
    {"fold": 3, "train": [0, 1, 2, 3, 6, 7, 8, 9], "val": [4, 5]},
    {"fold": 4, "train": [0, 1, 2, 3, 4, 5, 8, 9], "val": [6, 7]},
    {"fold": 5, "train": [0, 1, 2, 3, 4, 5, 6, 7], "val": [8, 9]},
]


def evaluate_cosine_baseline(dataset, max_hops):
    """Evaluate cosine-only baseline for a given max_hops."""
    env = GraphTraversalEnv(max_hops=max_hops)
    successes = 0
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

        while not env.done:
            neighbors = graph.neighbors(env.current_node)
            if not neighbors:
                env.done = True
                break

            # Cosine similarity scoring
            sims = torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0),
                torch.stack([graph.node_embeddings[n] for n, _ in neighbors]),
                dim=-1,
            )
            best_idx = sims.argmax().item()
            result = env.step(best_idx)

        if env.targets_found == env.target_set:
            successes += 1
        total_steps += env.steps
        count += 1

    if count == 0:
        return {"success_rate": 0.0, "avg_steps": 0.0, "count": 0}
    return {
        "success_rate": successes / count,
        "avg_steps": total_steps / count,
        "count": count,
    }


def train_fold(fold_def, args, graphs, ds, anchor_lambda, max_hops):
    """Train CoEvo for one fold with specific anchor_lambda and max_hops."""
    fold_id = fold_def["fold"]
    train_ids = fold_def["train"]
    val_ids = fold_def["val"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds, val_ds = ds.split_by_sample(train_ids, val_ids)
    train_graphs = [g for g in graphs if g.sample_id in train_ids]

    model = build_coevo_model(train_graphs, device=args.device)
    trainer = RLTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        lr=args.lr, edge_lr=args.edge_lr, anchor_lambda=anchor_lambda,
        max_hops=max_hops, device=args.device,
    )

    ckpt_dir = str(Path(args.output_dir) / args.current_variant / f"fold{fold_id}")
    t0 = time.time()
    history = trainer.train(
        num_epochs=args.epochs, eval_every=args.eval_every,
        checkpoint_dir=ckpt_dir,
    )
    elapsed = time.time() - t0

    # Find best val SR epoch
    val_entries = [h for h in history if "val_success_rate" in h]
    if val_entries:
        best_entry = max(val_entries, key=lambda h: h["val_success_rate"])
        best_val_sr = best_entry["val_success_rate"]
        best_epoch = best_entry["epoch"]
    else:
        best_val_sr = 0.0
        best_epoch = args.epochs

    # Final evaluation
    train_eval = trainer.evaluate(train_ds, greedy=True)
    val_eval = trainer.evaluate(val_ds, greedy=True)

    return {
        "fold": fold_id,
        "train_sr": train_eval["success_rate"],
        "val_sr": val_eval["success_rate"],
        "best_val_sr": best_val_sr,
        "best_epoch": best_epoch,
        "avg_steps": val_eval["avg_steps"],
        "time_s": elapsed,
    }


def run_variant(variant_name, anchor_lambda, max_hops, args, graphs, ds):
    """Run all 5 folds for a single variant."""
    logger.info(f"\n{'='*60}")
    logger.info(f"VARIANT: {variant_name} (anchor_lambda={anchor_lambda}, max_hops={max_hops})")
    logger.info(f"{'='*60}")

    args.current_variant = variant_name
    fold_results = []

    folds_to_run = [int(x) for x in args.folds.split(",")]
    selected_folds = [f for f in FOLDS if f["fold"] in folds_to_run]

    for fold_def in selected_folds:
        result = train_fold(fold_def, args, graphs, ds, anchor_lambda, max_hops)
        fold_results.append(result)
        logger.info(
            f"  Fold {result['fold']} | Val SR: {result['val_sr']:.3f} | "
            f"Best Val SR: {result['best_val_sr']:.3f} @ epoch {result['best_epoch']} | "
            f"Avg Steps: {result['avg_steps']:.2f} | {result['time_s']:.1f}s"
        )

    # Aggregate
    val_srs = [r["best_val_sr"] for r in fold_results]
    train_srs = [r["train_sr"] for r in fold_results]
    avg_steps = [r["avg_steps"] for r in fold_results]

    summary = {
        "variant": variant_name,
        "anchor_lambda": anchor_lambda,
        "max_hops": max_hops,
        "mean_val_sr": float(np.mean(val_srs)),
        "std_val_sr": float(np.std(val_srs)),
        "mean_train_sr": float(np.mean(train_srs)),
        "mean_avg_steps": float(np.mean(avg_steps)),
        "folds": fold_results,
    }

    logger.info(
        f"  -> {variant_name}: Val SR = {summary['mean_val_sr']:.3f} +/- {summary['std_val_sr']:.3f} | "
        f"Train SR = {summary['mean_train_sr']:.3f} | "
        f"Avg Steps = {summary['mean_avg_steps']:.2f}"
    )

    return summary


def run_anchor_ablation(args, graphs, ds):
    """Anchor regularization ablation."""
    logger.info("\n" + "=" * 60)
    logger.info("STUDY: ANCHOR REGULARIZATION (lambda_anchor)")
    logger.info("=" * 60)

    lambdas = [float(x) for x in args.anchor_lambdas.split(",")]
    results = []

    for lam in lambdas:
        name = f"anchor_{lam:.1f}".replace(".", "_")
        summary = run_variant(name, anchor_lambda=lam, max_hops=args.max_hops,
                              args=args, graphs=graphs, ds=ds)
        summary["lambda_anchor"] = lam
        results.append(summary)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ANCHOR REGULARIZATION — SUMMARY")
    logger.info(f"{'lambda':>8s} | {'Train SR':>10s} | {'Val SR':>10s} | {'+-Std':>7s} | {'Avg Steps':>10s}")
    logger.info("-" * 55)
    for r in results:
        logger.info(
            f"{r['lambda_anchor']:8.2f} | {r['mean_train_sr']:10.3f} | "
            f"{r['mean_val_sr']:10.3f} | {r['std_val_sr']:7.3f} | "
            f"{r['mean_avg_steps']:10.2f}"
        )

    return results


def run_hops_ablation(args, graphs, ds):
    """Max hops ablation."""
    logger.info("\n" + "=" * 60)
    logger.info("STUDY: MAX HOPS")
    logger.info("=" * 60)

    hops_list = [int(x) for x in args.max_hops_list.split(",")]
    results = []

    folds_to_run = [int(x) for x in args.folds.split(",")]
    selected_folds = [f for f in FOLDS if f["fold"] in folds_to_run]

    for hops in hops_list:
        name = f"hops_{hops}"

        # Also evaluate cosine baseline at this hop count for comparison
        cosine_srs = []
        for fold_def in selected_folds:
            train_ids = fold_def["train"]
            val_ids = fold_def["val"]
            _, val_ds = ds.split_by_sample(train_ids, val_ids)
            cosine_eval = evaluate_cosine_baseline(val_ds, max_hops=hops)
            cosine_srs.append(cosine_eval["success_rate"])

        summary = run_variant(name, anchor_lambda=args.anchor_lambda, max_hops=hops,
                              args=args, graphs=graphs, ds=ds)
        summary["cosine_baseline_val_sr"] = float(np.mean(cosine_srs))
        summary["cosine_baseline_std"] = float(np.std(cosine_srs))
        summary["delta_vs_cosine"] = summary["mean_val_sr"] - summary["cosine_baseline_val_sr"]
        results.append(summary)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MAX HOPS — SUMMARY")
    logger.info(f"{'Hops':>6s} | {'CoEvo SR':>10s} | {'+-Std':>7s} | {'Cosine SR':>10s} | {'Delta':>7s} | {'Avg Steps':>10s}")
    logger.info("-" * 65)
    for r in results:
        logger.info(
            f"{r['max_hops']:6d} | {r['mean_val_sr']:10.3f} | {r['std_val_sr']:7.3f} | "
            f"{r['cosine_baseline_val_sr']:10.3f} | {r['delta_vs_cosine']:+7.3f} | "
            f"{r['mean_avg_steps']:10.2f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Ablation for CoEvo")
    parser.add_argument("--study", type=str, default="both",
                        choices=["anchor", "hops", "both"],
                        help="Which ablation study to run")
    parser.add_argument("--anchor-lambdas", type=str, default="0.0,0.1,0.5,1.0,2.0",
                        help="Anchor lambda values (comma-separated)")
    parser.add_argument("--max-hops-list", type=str, default="3,5,7",
                        help="Max hops values (comma-separated)")

    # Training args (same defaults as run_5fold_cv.py)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--edge-lr", type=float, default=1e-4)
    parser.add_argument("--anchor-lambda", type=float, default=1.0,
                        help="Default anchor lambda (used by hops ablation)")
    parser.add_argument("--max-hops", type=int, default=5,
                        help="Default max hops (used by anchor ablation)")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_hyperparams_ablation")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5",
                        help="Which folds to run (comma-separated)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
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

    all_results = {}

    if args.study in ("anchor", "both"):
        anchor_results = run_anchor_ablation(args, graphs, ds)
        all_results["anchor_lambda"] = anchor_results

    if args.study in ("hops", "both"):
        hops_results = run_hops_ablation(args, graphs, ds)
        all_results["max_hops"] = hops_results

    # Save all results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
