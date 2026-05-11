#!/usr/bin/env python3
"""
Reward Design Ablation for CoEvo Edge Learning.

Two ablation studies:

1. Reward Design Ablation (Table 1):
   - hit_only:       +10 hit, no step/timeout penalties
   - hit_minus_step: +10 hit, -0.05 step, no timeout
   - full:           +10 hit, -0.05 step, -1.0 timeout (default)
   - normalized:     +1.0 hit, -0.05 step, -1.0 timeout

2. Step Penalty Sensitivity (Table 2):
   - λ_step ∈ {0.0, 0.01, 0.05, 0.1, 0.3}
   - All with +10 hit, -1.0 timeout

Both use 5-fold CV with the same fold definitions as run_5fold_cv.py.

Usage:
    # Run both ablations
    python scripts/run_reward_ablation.py

    # Run only reward design ablation
    python scripts/run_reward_ablation.py --study reward_design

    # Run only step penalty sensitivity
    python scripts/run_reward_ablation.py --study step_lambda

    # Custom step lambdas
    python scripts/run_reward_ablation.py --study step_lambda --lambdas 0.0,0.01,0.05,0.1,0.3,0.5,1.0
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
from memory.env import RewardConfig

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

# ── Reward design variants ──
REWARD_DESIGNS = {
    "hit_only": RewardConfig.hit_only,
    "hit_minus_step": RewardConfig.hit_minus_step,
    "full": RewardConfig.default,
    "normalized": RewardConfig.normalized_hit_minus_step,
}


def train_fold_with_reward(fold_def, reward_config, args, graphs, ds):
    """Train CoEvo for one fold with a specific reward config. Returns metrics dict."""
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
        lr=args.lr, edge_lr=args.edge_lr, anchor_lambda=args.anchor_lambda,
        max_hops=args.max_hops, device=args.device,
        reward_config=reward_config,
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


def run_variant(variant_name, reward_config, args, graphs, ds):
    """Run all 5 folds for a single reward variant."""
    logger.info(f"\n{'='*60}")
    logger.info(f"VARIANT: {variant_name}")
    logger.info(f"  hit_reward={reward_config.hit_reward}, "
                f"step_penalty={reward_config.step_penalty}, "
                f"timeout_penalty={reward_config.timeout_penalty}")
    logger.info(f"{'='*60}")

    args.current_variant = variant_name
    fold_results = []

    folds_to_run = [int(x) for x in args.folds.split(",")]
    selected_folds = [f for f in FOLDS if f["fold"] in folds_to_run]

    for fold_def in selected_folds:
        result = train_fold_with_reward(fold_def, reward_config, args, graphs, ds)
        fold_results.append(result)
        logger.info(
            f"  Fold {result['fold']} | Val SR: {result['val_sr']:.3f} | "
            f"Best Val SR: {result['best_val_sr']:.3f} @ epoch {result['best_epoch']} | "
            f"Avg Steps: {result['avg_steps']:.2f} | {result['time_s']:.1f}s"
        )

    # Aggregate
    val_srs = [r["best_val_sr"] for r in fold_results]
    avg_steps = [r["avg_steps"] for r in fold_results]

    summary = {
        "variant": variant_name,
        "reward_config": {
            "hit_reward": reward_config.hit_reward,
            "step_penalty": reward_config.step_penalty,
            "timeout_penalty": reward_config.timeout_penalty,
        },
        "mean_val_sr": float(np.mean(val_srs)),
        "std_val_sr": float(np.std(val_srs)),
        "mean_avg_steps": float(np.mean(avg_steps)),
        "folds": fold_results,
    }

    logger.info(
        f"  → {variant_name}: Val SR = {summary['mean_val_sr']:.3f} ± {summary['std_val_sr']:.3f} | "
        f"Avg Steps = {summary['mean_avg_steps']:.2f}"
    )

    return summary


def run_reward_design_ablation(args, graphs, ds):
    """Table 1: Reward Design Ablation."""
    logger.info("\n" + "=" * 60)
    logger.info("STUDY 1: REWARD DESIGN ABLATION")
    logger.info("=" * 60)

    results = []
    for name, factory in REWARD_DESIGNS.items():
        rc = factory()
        summary = run_variant(name, rc, args, graphs, ds)
        results.append(summary)

    # Print LaTeX-ready summary
    logger.info("\n" + "=" * 60)
    logger.info("REWARD DESIGN ABLATION — SUMMARY")
    logger.info(f"{'Variant':<20s} | {'Val SR':>10s} | {'± Std':>7s} | {'Avg Steps':>10s}")
    logger.info("-" * 55)
    for r in results:
        logger.info(
            f"{r['variant']:<20s} | {r['mean_val_sr']:10.3f} | {r['std_val_sr']:7.3f} | "
            f"{r['mean_avg_steps']:10.2f}"
        )

    return results


def run_step_lambda_sensitivity(args, graphs, ds):
    """Table 2: Step Penalty Sensitivity."""
    logger.info("\n" + "=" * 60)
    logger.info("STUDY 2: STEP PENALTY SENSITIVITY")
    logger.info("=" * 60)

    lambdas = [float(x) for x in args.lambdas.split(",")]
    results = []

    for lam in lambdas:
        name = f"lambda_{lam:.2f}".replace(".", "_")
        rc = RewardConfig.with_step_lambda(lam)
        summary = run_variant(name, rc, args, graphs, ds)
        summary["lambda_step"] = lam
        results.append(summary)

    # Print LaTeX-ready summary
    logger.info("\n" + "=" * 60)
    logger.info("STEP PENALTY SENSITIVITY — SUMMARY")
    logger.info(f"{'λ_step':>8s} | {'Val SR':>10s} | {'± Std':>7s} | {'Avg Steps':>10s}")
    logger.info("-" * 45)
    for r in results:
        logger.info(
            f"{r['lambda_step']:8.3f} | {r['mean_val_sr']:10.3f} | {r['std_val_sr']:7.3f} | "
            f"{r['mean_avg_steps']:10.2f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Reward Design Ablation for CoEvo")
    parser.add_argument("--study", type=str, default="both",
                        choices=["reward_design", "step_lambda", "both"],
                        help="Which ablation study to run")
    parser.add_argument("--lambdas", type=str, default="0.0,0.01,0.05,0.1,0.3",
                        help="Step penalty lambdas (comma-separated)")

    # Training args (same defaults as run_5fold_cv.py)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--edge-lr", type=float, default=1e-4)
    parser.add_argument("--anchor-lambda", type=float, default=1.0)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_reward_ablation")
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

    if args.study in ("reward_design", "both"):
        rd_results = run_reward_design_ablation(args, graphs, ds)
        all_results["reward_design"] = rd_results

    if args.study in ("step_lambda", "both"):
        sl_results = run_step_lambda_sensitivity(args, graphs, ds)
        all_results["step_lambda"] = sl_results

    # Save all results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
