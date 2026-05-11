#!/usr/bin/env python3
"""
Sweep anchor_lambda for CoEvo mode + extended epochs.

Usage:
    python scripts/sweep_lambda.py
    python scripts/sweep_lambda.py --epochs 200 --device cuda
"""

import argparse
import csv
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Lambda sweep for CoEvo")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--edge-lr", type=float, default=1e-4)
    parser.add_argument("--lambdas", type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,"
                                "1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,"
                                "2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,"
                                "3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,"
                                "4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0")
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results_lambda_sweep")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data (once) ──
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
    train_graphs = [g for g in graphs if g.sample_id in train_ids]

    lambdas = [float(x) for x in args.lambdas.split(",")]
    all_results = {}

    for lam in lambdas:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training CoEvo with anchor_lambda={lam}")
        logger.info(f"{'='*60}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = build_coevo_model(train_graphs, device=args.device)
        trainer = RLTrainer(
            model=model, train_dataset=train_ds, val_dataset=val_ds,
            lr=args.lr, edge_lr=args.edge_lr, anchor_lambda=lam,
            max_hops=args.max_hops, device=args.device,
        )

        t0 = time.time()
        history = trainer.train(
            num_epochs=args.epochs, eval_every=args.eval_every,
            checkpoint_dir=str(output_dir / f"ckpt_lambda_{lam}"),
        )
        elapsed = time.time() - t0

        train_eval = trainer.evaluate(train_ds, greedy=True)
        val_eval = trainer.evaluate(val_ds, greedy=True)

        # Track best val SR across epochs
        val_history = [h.get("val_success_rate", 0) for h in history if "val_success_rate" in h]
        best_val_sr = max(val_history) if val_history else val_eval["success_rate"]
        best_val_epoch = (val_history.index(best_val_sr) + 1) * args.eval_every if val_history else args.epochs

        all_results[str(lam)] = {
            "train_sr": train_eval["success_rate"],
            "val_sr": val_eval["success_rate"],
            "best_val_sr": best_val_sr,
            "best_val_epoch": best_val_epoch,
            "train_reward": train_eval["avg_reward"],
            "val_reward": val_eval["avg_reward"],
            "time_s": elapsed,
            "history": history,
        }

        logger.info(
            f"lambda={lam:5.1f} | Train SR: {train_eval['success_rate']:.3f} | "
            f"Val SR: {val_eval['success_rate']:.3f} | "
            f"Best Val SR: {best_val_sr:.3f} @ epoch {best_val_epoch} | "
            f"{elapsed:.1f}s"
        )

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info("LAMBDA SWEEP RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"{'Lambda':>8s} | {'Train SR':>10s} | {'Val SR':>10s} | {'Best Val':>10s} | {'@ Epoch':>8s}")
    logger.info("-" * 60)
    for lam in lambdas:
        r = all_results[str(lam)]
        logger.info(
            f"{lam:8.1f} | {r['train_sr']:10.3f} | {r['val_sr']:10.3f} | "
            f"{r['best_val_sr']:10.3f} | {r['best_val_epoch']:>8d}"
        )

    # ── Save JSON (full) ──
    results_path = output_dir / "sweep_results.json"
    save_json = {}
    for k, v in all_results.items():
        save_json[k] = {key: val for key, val in v.items() if key != "history"}
    with open(results_path, "w") as f:
        json.dump(save_json, f, indent=2)
    logger.info(f"JSON saved to {results_path}")

    # ── CSV 1: Summary table ──
    summary_path = output_dir / "sweep_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "best_val_sr", "best_epoch", "final_train_sr", "final_val_sr"])
        for lam in lambdas:
            r = all_results[str(lam)]
            writer.writerow([lam, r["best_val_sr"], r["best_val_epoch"],
                             r["train_sr"], r["val_sr"]])
    logger.info(f"Summary CSV saved to {summary_path}")

    # ── CSV 2: Per-epoch learning curves ──
    curves_path = output_dir / "learning_curves.csv"
    with open(curves_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: epoch, lambda_0.1_train_sr, lambda_0.1_val_sr, lambda_0.5_train_sr, ...
        header = ["epoch"]
        for lam in lambdas:
            header.extend([f"lambda_{lam}_train_sr", f"lambda_{lam}_val_sr"])
        writer.writerow(header)

        # Find max epochs across all runs
        max_epoch = max(len(all_results[str(lam)]["history"]) for lam in lambdas)

        for epoch_idx in range(max_epoch):
            row = [epoch_idx + 1]
            for lam in lambdas:
                hist = all_results[str(lam)]["history"]
                if epoch_idx < len(hist):
                    h = hist[epoch_idx]
                    row.append(h.get("train_success_rate", ""))
                    row.append(h.get("val_success_rate", ""))
                else:
                    row.extend(["", ""])
            writer.writerow(row)
    logger.info(f"Learning curves CSV saved to {curves_path}")


if __name__ == "__main__":
    main()
