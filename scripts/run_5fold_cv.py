#!/usr/bin/env python3
"""
5-Fold Cross-Validation for CoEvo Edge Learning.

Phase 1: Train CoEvo models for all 5 folds (routing-level).
Phase 2: Run end-to-end LLM evaluation on held-out samples using each fold's
         best checkpoint via test_fixed_memory.py.

Fold definitions (sample-level):
  Fold 1: train={2,3,4,5,6,7,8,9}, val={0,1}
  Fold 2: train={0,1,4,5,6,7,8,9}, val={2,3}
  Fold 3: train={0,1,2,3,6,7,8,9}, val={4,5}
  Fold 4: train={0,1,2,3,4,5,8,9}, val={6,7}
  Fold 5: train={0,1,2,3,4,5,6,7}, val={8,9}

Locked config: mode=coevo, lr=1e-3, edge_lr=1e-4, anchor_lambda=1.0,
               max_hops=5, epochs=200, early_stop=best_val_sr, seed=42

Usage:
    # Train all 5 folds (routing-level only)
    python scripts/run_5fold_cv.py --train-only

    # Train + end-to-end eval
    python scripts/run_5fold_cv.py

    # End-to-end eval only (requires trained checkpoints)
    python scripts/run_5fold_cv.py --eval-only --model gpt-4o-mini
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.pytorch_graph import load_all_graphs, QADataset
from memory.model import build_coevo_model
from memory.rl_trainer import RLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Fold definitions ──
FOLDS = [
    {"fold": 1, "train": [2, 3, 4, 5, 6, 7, 8, 9], "val": [0, 1]},
    {"fold": 2, "train": [0, 1, 4, 5, 6, 7, 8, 9], "val": [2, 3]},
    {"fold": 3, "train": [0, 1, 2, 3, 6, 7, 8, 9], "val": [4, 5]},
    {"fold": 4, "train": [0, 1, 2, 3, 4, 5, 8, 9], "val": [6, 7]},
    {"fold": 5, "train": [0, 1, 2, 3, 4, 5, 6, 7], "val": [8, 9]},
]


def train_fold(fold_def, args, graphs, ds, encoder):
    """Train CoEvo for a single fold. Returns history and best checkpoint path."""
    fold_id = fold_def["fold"]
    train_ids = fold_def["train"]
    val_ids = fold_def["val"]

    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_id}: train={train_ids}, val={val_ids}")
    logger.info(f"{'='*60}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds, val_ds = ds.split_by_sample(train_ids, val_ids)
    train_graphs = [g for g in graphs if g.sample_id in train_ids]

    model = build_coevo_model(train_graphs, device=args.device)
    trainer = RLTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        lr=args.lr, edge_lr=args.edge_lr, anchor_lambda=args.anchor_lambda,
        max_hops=args.max_hops, device=args.device,
    )

    ckpt_dir = str(Path(args.output_dir) / f"fold{fold_id}")
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

    # Also evaluate baselines on this fold's val set
    from scripts.run_coevo_ablation import evaluate_baseline
    cosine_eval = evaluate_baseline("cosine_only", val_ds, max_hops=args.max_hops)

    # Evaluate CoEvo at final state
    train_eval = trainer.evaluate(train_ds, greedy=True)
    val_eval = trainer.evaluate(val_ds, greedy=True)

    result = {
        "fold": fold_id,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_sr": train_eval["success_rate"],
        "val_sr": val_eval["success_rate"],
        "best_val_sr": best_val_sr,
        "best_epoch": best_epoch,
        "cosine_only_val_sr": cosine_eval["success_rate"],
        "time_s": elapsed,
    }

    # Best checkpoint path (for end-to-end eval)
    best_ckpt = Path(ckpt_dir) / f"model_epoch_{best_epoch}.pt"
    if not best_ckpt.exists():
        # Fallback to final if exact epoch checkpoint doesn't exist
        best_ckpt = Path(ckpt_dir) / "model_final.pt"
    result["best_checkpoint"] = str(best_ckpt)

    logger.info(
        f"Fold {fold_id} done | Train SR: {train_eval['success_rate']:.3f} | "
        f"Val SR: {val_eval['success_rate']:.3f} | "
        f"Best Val SR: {best_val_sr:.3f} @ epoch {best_epoch} | "
        f"Cosine-only: {cosine_eval['success_rate']:.3f} | "
        f"{elapsed:.1f}s"
    )

    return result


def run_end_to_end_eval(fold_def, checkpoint_path, args):
    """Run test_fixed_memory.py on held-out samples with CoEvo edge scorer."""
    fold_id = fold_def["fold"]
    val_ids = fold_def["val"]

    logger.info(f"\n{'='*60}")
    logger.info(f"End-to-end eval: Fold {fold_id}, samples={val_ids}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*60}")

    sample_args = " ".join(str(s) for s in val_ids)

    cmd = [
        sys.executable, "test_fixed_memory.py",
        "--sample", *[str(s) for s in val_ids],
        "--edge-scorer", "rl_coevo",
        "--rl-checkpoint", checkpoint_path,
        "--model", args.model,
        "--category-to-test", "1,2,3,4,5",
    ]

    if args.use_openai:
        cmd.append("--use-openai")

    if args.backend:
        cmd.extend(["--backend", args.backend])

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=7200,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    if result.returncode != 0:
        logger.error(f"End-to-end eval failed for fold {fold_id}:")
        logger.error(result.stderr[-2000:] if result.stderr else "No stderr")
        return None

    logger.info(result.stdout[-3000:] if result.stdout else "No stdout")
    return result.stdout


def collect_end_to_end_results(args):
    """Collect end-to-end results from result JSON files for all folds."""
    model_name = args.model.replace(".", "_").replace("-", "_")
    results_dir = Path(f"results_{model_name}")

    all_results = []
    for fold_def in FOLDS:
        for sample_id in fold_def["val"]:
            result_file = results_dir / f"fixed_results_sample{sample_id}.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                stats = data.get("stats", {})
                all_results.append({
                    "fold": fold_def["fold"],
                    "sample_id": sample_id,
                    "overall": stats.get("overall", {}),
                    "without_cat5": stats.get("without_category5", {}),
                    "category_breakdown": stats.get("category_breakdown", {}),
                })
            else:
                logger.warning(f"Result file not found: {result_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="5-Fold CV for CoEvo")
    # Training args
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--edge-lr", type=float, default=1e-4)
    parser.add_argument("--anchor-lambda", type=float, default=1.0)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_5fold_cv")

    # End-to-end eval args
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model for end-to-end evaluation")
    parser.add_argument("--use-openai", action="store_true",
                        help="Use OpenAI API instead of Azure")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["openai", "ollama"])

    # Mode flags
    parser.add_argument("--train-only", action="store_true",
                        help="Only train routing models, skip end-to-end eval")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run end-to-end eval (requires trained checkpoints)")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5",
                        help="Which folds to run (comma-separated)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_to_run = [int(x) for x in args.folds.split(",")]
    selected_folds = [f for f in FOLDS if f["fold"] in folds_to_run]

    # ── Phase 1: Train routing models ──
    if not args.eval_only:
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

        fold_results = []
        for fold_def in selected_folds:
            result = train_fold(fold_def, args, graphs, ds, encoder)
            fold_results.append(result)

        # ── Summary table ──
        logger.info(f"\n{'='*60}")
        logger.info("5-FOLD CV ROUTING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"{'Fold':>5s} | {'Val Samples':>12s} | {'CoEvo Val SR':>13s} | {'Best Epoch':>10s} | {'Cosine Val SR':>14s} | {'Delta':>7s}")
        logger.info("-" * 75)

        coevo_srs = []
        cosine_srs = []
        for r in fold_results:
            delta = r["best_val_sr"] - r["cosine_only_val_sr"]
            logger.info(
                f"{r['fold']:5d} | {str(r['val_ids']):>12s} | "
                f"{r['best_val_sr']:13.3f} | {r['best_epoch']:10d} | "
                f"{r['cosine_only_val_sr']:14.3f} | {delta:+7.3f}"
            )
            coevo_srs.append(r["best_val_sr"])
            cosine_srs.append(r["cosine_only_val_sr"])

        mean_coevo = np.mean(coevo_srs)
        std_coevo = np.std(coevo_srs)
        mean_cosine = np.mean(cosine_srs)
        std_cosine = np.std(cosine_srs)
        mean_delta = mean_coevo - mean_cosine

        logger.info("-" * 75)
        logger.info(
            f"{'Mean':>5s} | {'':>12s} | {mean_coevo:13.3f} | {'':>10s} | "
            f"{mean_cosine:14.3f} | {mean_delta:+7.3f}"
        )
        logger.info(f"{'Std':>5s} | {'':>12s} | {std_coevo:13.3f} | {'':>10s} | {std_cosine:14.3f}")

        # Wilcoxon signed-rank test
        try:
            from scipy.stats import wilcoxon
            stat, p_value = wilcoxon(coevo_srs, cosine_srs)
            logger.info(f"\nWilcoxon signed-rank test: stat={stat:.4f}, p={p_value:.4f}")
        except Exception as e:
            logger.info(f"\nWilcoxon test skipped: {e}")

        # Save routing results
        routing_results = {
            "config": {
                "mode": "coevo",
                "lr": args.lr,
                "edge_lr": args.edge_lr,
                "anchor_lambda": args.anchor_lambda,
                "max_hops": args.max_hops,
                "epochs": args.epochs,
                "seed": args.seed,
            },
            "folds": fold_results,
            "summary": {
                "coevo_mean_val_sr": mean_coevo,
                "coevo_std_val_sr": std_coevo,
                "cosine_mean_val_sr": mean_cosine,
                "cosine_std_val_sr": std_cosine,
                "mean_delta": mean_delta,
            },
        }
        with open(output_dir / "routing_results.json", "w") as f:
            json.dump(routing_results, f, indent=2)
        logger.info(f"\nRouting results saved to {output_dir / 'routing_results.json'}")

    # ── Phase 2: End-to-end LLM evaluation ──
    if not args.train_only:
        # Load routing results to get checkpoint paths
        routing_file = output_dir / "routing_results.json"
        if routing_file.exists():
            with open(routing_file) as f:
                routing_results = json.load(f)
            fold_results = routing_results["folds"]
        else:
            logger.error(f"No routing results found at {routing_file}. Run training first.")
            return 1

        logger.info(f"\n{'='*60}")
        logger.info("STARTING END-TO-END LLM EVALUATION")
        logger.info(f"Model: {args.model}")
        logger.info(f"{'='*60}")

        for fold_result in fold_results:
            fold_id = fold_result["fold"]
            if fold_id not in folds_to_run:
                continue
            fold_def = FOLDS[fold_id - 1]
            checkpoint_path = fold_result["best_checkpoint"]

            if not Path(checkpoint_path).exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                continue

            run_end_to_end_eval(fold_def, checkpoint_path, args)

        # Collect and summarize results
        e2e_results = collect_end_to_end_results(args)
        if e2e_results:
            logger.info(f"\n{'='*60}")
            logger.info("END-TO-END RESULTS SUMMARY")
            logger.info(f"{'='*60}")

            logger.info(f"{'Sample':>7s} | {'Fold':>5s} | {'Acc%':>7s} | {'F1%':>7s} | {'LLM%':>7s} | {'Acc% -C5':>9s}")
            logger.info("-" * 55)

            for r in e2e_results:
                overall = r["overall"]
                wo_c5 = r["without_cat5"]
                logger.info(
                    f"{r['sample_id']:7d} | {r['fold']:5d} | "
                    f"{overall.get('accuracy', 0):7.1f} | "
                    f"{overall.get('avg_f1', 0):7.1f} | "
                    f"{overall.get('avg_llm', 0):7.1f} | "
                    f"{wo_c5.get('accuracy', 0):9.1f}"
                )

            # Averages
            if len(e2e_results) > 0:
                avg_acc = np.mean([r["overall"].get("accuracy", 0) for r in e2e_results])
                avg_f1 = np.mean([r["overall"].get("avg_f1", 0) for r in e2e_results])
                avg_llm = np.mean([r["overall"].get("avg_llm", 0) for r in e2e_results])
                avg_acc_nc5 = np.mean([r["without_cat5"].get("accuracy", 0) for r in e2e_results])

                logger.info("-" * 55)
                logger.info(
                    f"{'Mean':>7s} | {'':>5s} | {avg_acc:7.1f} | "
                    f"{avg_f1:7.1f} | {avg_llm:7.1f} | {avg_acc_nc5:9.1f}"
                )

            # Save end-to-end results
            with open(output_dir / "e2e_results.json", "w") as f:
                json.dump(e2e_results, f, indent=2)
            logger.info(f"\nEnd-to-end results saved to {output_dir / 'e2e_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
