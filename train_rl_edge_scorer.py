#!/usr/bin/env python3
"""
Phase 2 Training Script — CoEvoMem RL Edge Scorer

Usage:
    # Transductive (per-graph, sample 0)
    python train_rl_edge_scorer.py --mode transductive --sample 0 --epochs 100

    # Inductive (cross-graph, 8/2 split)
    python train_rl_edge_scorer.py --mode inductive --epochs 100

    # Quick smoke test
    python train_rl_edge_scorer.py --mode inductive --epochs 5 --max-hops 3
"""

import argparse
import json
import logging
import sys
import time

import torch
import numpy as np

from memory.pytorch_graph import load_all_graphs, QADataset
from memory.model import build_transductive_model, build_inductive_model, build_coevo_model, build_hybrid_model
from memory.rl_trainer import RLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CoEvoMem RL Edge Scorer")
    parser.add_argument("--mode", choices=["transductive", "inductive", "coevo", "hybrid"], default="inductive")
    parser.add_argument("--sample", type=int, default=0, help="Sample ID for transductive mode")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--edge-lr", type=float, default=None,
                        help="Separate LR for edge features in coevo mode (default: lr/10)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-hops", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated sample IDs for training")
    parser.add_argument("--val-samples", type=str, default="8,9",
                        help="Comma-separated sample IDs for validation")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load data ----
    logger.info("Loading graphs...")
    graphs = load_all_graphs(device=args.device)

    logger.info("Loading QA data...")
    locomo = json.load(open("data/locomo10.json"))
    for i, s in enumerate(locomo):
        s["sample_id"] = i

    ds = QADataset(locomo, graphs)

    # ---- Encode queries ----
    logger.info("Encoding query embeddings...")
    # Re-use the encoder from the graph builder
    from memory.vector_db import VectorEncoder
    encoder = VectorEncoder(model_name="all-MiniLM-L6-v2", use_openai=False)
    ds.encoder = encoder
    ds.encode_queries()

    # ---- Build model ----
    train_ids = [int(x) for x in args.train_samples.split(",")]
    val_ids = [int(x) for x in args.val_samples.split(",")]

    if args.mode == "transductive":
        graph = next(g for g in graphs if g.sample_id == args.sample)
        model = build_transductive_model(graph, device=args.device)
        # Transductive: train and val must come from the SAME graph
        # (edge embeddings are graph-specific). 80/20 split within the sample.
        same_graph_ds, _ = ds.split_by_sample([args.sample], [])
        n = len(same_graph_ds)
        split = int(n * 0.8)
        np.random.seed(args.seed)
        perm = np.random.permutation(n).tolist()
        train_ds = QADataset.__new__(QADataset)
        train_ds.encoder = ds.encoder
        train_ds.items = [same_graph_ds.items[i] for i in perm[:split]]
        val_ds = QADataset.__new__(QADataset)
        val_ds.encoder = ds.encoder
        val_ds.items = [same_graph_ds.items[i] for i in perm[split:]]
        ckpt_dir = f"{args.checkpoint_dir}/transductive_s{args.sample}"
    elif args.mode == "coevo":
        # Co-evolution: jointly train edge features + router
        train_graphs = [g for g in graphs if g.sample_id in train_ids]
        model = build_coevo_model(train_graphs, device=args.device)
        train_ds, val_ds = ds.split_by_sample(train_ids, val_ids)
        ckpt_dir = f"{args.checkpoint_dir}/coevo"
        logger.info(
            f"CoEvo mode: jointly training edge features + router on samples {train_ids}, "
            f"zero-shot validation on {val_ids}"
        )
    elif args.mode == "hybrid":
        # Hybrid: per-graph edge embeddings + shared router
        # Train on train_ids, validate on val_ids (unseen graphs)
        train_graphs = [g for g in graphs if g.sample_id in train_ids]
        model = build_hybrid_model(train_graphs, device=args.device)
        train_ds, val_ds = ds.split_by_sample(train_ids, val_ids)
        ckpt_dir = f"{args.checkpoint_dir}/hybrid"
        logger.info(
            f"Hybrid mode: edge embeddings for samples {train_ids}, "
            f"router generalizes to val samples {val_ids}"
        )
    else:
        model = build_inductive_model(device=args.device)
        train_ds, val_ds = ds.split_by_sample(train_ids, val_ids)
        ckpt_dir = f"{args.checkpoint_dir}/inductive"

    logger.info(
        f"Mode: {args.mode} | "
        f"Model params: {sum(p.numel() for p in model.parameters()):,} | "
        f"Train: {len(train_ds)} items | Val: {len(val_ds)} items"
    )

    # ---- Train ----
    edge_lr = args.edge_lr if args.edge_lr is not None else args.lr / 10.0
    trainer = RLTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=args.lr,
        edge_lr=edge_lr if args.mode == "coevo" else None,
        gamma=args.gamma,
        max_hops=args.max_hops,
        device=args.device,
    )

    t0 = time.time()
    history = trainer.train(
        num_epochs=args.epochs,
        eval_every=args.eval_every,
        checkpoint_dir=ckpt_dir,
    )
    total_time = time.time() - t0

    # ---- Final evaluation ----
    logger.info("=" * 60)
    logger.info("Final evaluation (greedy):")

    train_eval = trainer.evaluate(train_ds, greedy=True)
    logger.info(f"  Train: SR={train_eval['success_rate']:.3f}, "
                f"Reward={train_eval['avg_reward']:.2f}, "
                f"Steps={train_eval['avg_steps']:.1f}")

    if val_ds and len(val_ds) > 0:
        val_eval = trainer.evaluate(val_ds, greedy=True)
        logger.info(f"  Val:   SR={val_eval['success_rate']:.3f}, "
                    f"Reward={val_eval['avg_reward']:.2f}, "
                    f"Steps={val_eval['avg_steps']:.1f}")

    logger.info(f"Total training time: {total_time:.1f}s")
    logger.info(f"Checkpoints saved to: {ckpt_dir}/")

    # ---- Save history ----
    history_path = f"{ckpt_dir}/history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
