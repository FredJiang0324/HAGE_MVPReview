"""
REINFORCE Training Pipeline (Phase 2)

Trains CoEvoMem in either Transductive or Inductive mode using
policy-gradient (REINFORCE with baseline subtraction).

Training loop:
  - 8/2 sample split (train / zero-shot validation)
  - Tracks success rate (ID-match) and average return
  - Validates on unseen samples every N epochs
  - Zero LLM calls
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .model import CoEvoMem, build_transductive_model, build_inductive_model, build_coevo_model
from .env import GraphTraversalEnv, RewardConfig, run_episode
from .pytorch_graph import PyTorchGraph, QADataset

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    REINFORCE trainer for CoEvoMem.

    Supports both Transductive (per-graph) and Inductive (cross-graph) modes.
    """

    def __init__(
        self,
        model: CoEvoMem,
        train_dataset: QADataset,
        val_dataset: Optional[QADataset] = None,
        lr: float = 1e-3,
        edge_lr: Optional[float] = None,
        anchor_lambda: float = 0.0,
        gamma: float = 0.99,
        max_hops: int = 5,
        device: str = "cpu",
        reward_config: RewardConfig = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.gamma = gamma
        self.max_hops = max_hops
        self.device = torch.device(device)
        self.anchor_lambda = anchor_lambda
        self.reward_config = reward_config or RewardConfig.default()

        # Separate LR groups for coevo mode: slower LR for edge features
        if model.mode == "coevo" and edge_lr is not None:
            router_params = list(model.query_router.parameters())
            edge_params = list(model.edge_features_dict.parameters())
            self.optimizer = torch.optim.Adam([
                {"params": router_params, "lr": lr},
                {"params": edge_params, "lr": edge_lr},
            ])
            logger.info(f"CoEvo optimizer: router_lr={lr}, edge_lr={edge_lr}")
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.env = GraphTraversalEnv(max_hops=max_hops, reward_config=self.reward_config)

        # Running baseline for variance reduction
        self.baseline = 0.0
        self.baseline_decay = 0.99

        # Track train/val sample splits for checkpoint metadata
        self.train_sample_ids = sorted(set(it["sample_id"] for it in train_dataset))
        self.val_sample_ids = sorted(set(it["sample_id"] for it in val_dataset)) if val_dataset else []

        # Metrics history
        self.history: List[Dict] = []

    def _discounted_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns G_t for each timestep."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def train_one_item(self, item: dict) -> Tuple[float, float, bool]:
        """
        Run one episode and update model.

        Returns:
            (total_reward, loss, success)
        """
        graph = item["graph"]
        query_emb = item["query_embedding"].to(self.device)
        targets = item["target_node_indices"]

        rewards, log_probs, success, steps = run_episode(
            self.model, self.env, graph, query_emb, targets, greedy=False
        )

        if not log_probs:
            return 0.0, 0.0, success

        # Discounted returns
        returns = self._discounted_returns(rewards)
        total_reward = sum(rewards)

        # Update baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * total_reward

        # Policy gradient with baseline subtraction
        loss = torch.tensor(0.0, device=self.device)
        for log_prob, G in zip(log_probs, returns):
            advantage = G - self.baseline
            loss = loss + (-log_prob * advantage)

        # Anchor regularization: keep edge features close to Phase 1 init
        if self.anchor_lambda > 0:
            loss = loss + self.anchor_lambda * self.model.edge_anchor_loss()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_reward, loss.item(), success

    def evaluate(self, dataset: QADataset, greedy: bool = True) -> Dict:
        """
        Evaluate model on a dataset (no gradient updates).

        Returns:
            dict with success_rate, avg_reward, avg_steps
        """
        self.model.eval()
        successes = 0
        total_reward = 0.0
        total_steps = 0
        count = 0

        with torch.no_grad():
            for item in dataset.items:
                if item["query_embedding"] is None:
                    continue

                graph = item["graph"]
                query_emb = item["query_embedding"].to(self.device)
                targets = item["target_node_indices"]

                rewards, _, success, steps = run_episode(
                    self.model, self.env, graph, query_emb, targets, greedy=greedy
                )

                if success:
                    successes += 1
                total_reward += sum(rewards)
                total_steps += steps
                count += 1

        self.model.train()

        if count == 0:
            return {"success_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0, "count": 0}

        return {
            "success_rate": successes / count,
            "avg_reward": total_reward / count,
            "avg_steps": total_steps / count,
            "count": count,
        }

    def train(
        self,
        num_epochs: int = 100,
        eval_every: int = 10,
        checkpoint_dir: Optional[str] = None,
        shuffle: bool = True,
    ) -> List[Dict]:
        """
        Main training loop.

        Args:
            num_epochs: Number of full passes over the training data.
            eval_every: Run validation every N epochs.
            checkpoint_dir: Save checkpoints here (optional).

        Returns:
            Training history (list of per-epoch metrics dicts).
        """
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        train_items = [it for it in self.train_dataset.items if it["query_embedding"] is not None]
        if not train_items:
            raise RuntimeError("No training items with query embeddings. Call dataset.encode_queries() first.")

        logger.info(
            f"Training: {len(train_items)} items, "
            f"{num_epochs} epochs, mode={self.model.mode}"
        )

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            t0 = time.time()

            epoch_rewards = []
            epoch_losses = []
            epoch_successes = 0

            # Shuffle training items
            indices = list(range(len(train_items)))
            if shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                item = train_items[idx]
                reward, loss, success = self.train_one_item(item)
                epoch_rewards.append(reward)
                epoch_losses.append(loss)
                if success:
                    epoch_successes += 1

            elapsed = time.time() - t0
            n = len(train_items)
            metrics = {
                "epoch": epoch,
                "train_success_rate": epoch_successes / n,
                "train_avg_reward": np.mean(epoch_rewards),
                "train_avg_loss": np.mean(epoch_losses),
                "train_time_s": elapsed,
            }

            # Validation
            if self.val_dataset and epoch % eval_every == 0:
                val_metrics = self.evaluate(self.val_dataset, greedy=True)
                metrics["val_success_rate"] = val_metrics["success_rate"]
                metrics["val_avg_reward"] = val_metrics["avg_reward"]
                metrics["val_avg_steps"] = val_metrics["avg_steps"]

                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train SR: {metrics['train_success_rate']:.3f} | "
                    f"Val SR: {val_metrics['success_rate']:.3f} | "
                    f"Reward: {metrics['train_avg_reward']:.2f} | "
                    f"Loss: {metrics['train_avg_loss']:.3f} | "
                    f"{elapsed:.1f}s"
                )
            else:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train SR: {metrics['train_success_rate']:.3f} | "
                    f"Reward: {metrics['train_avg_reward']:.2f} | "
                    f"Loss: {metrics['train_avg_loss']:.3f} | "
                    f"{elapsed:.1f}s"
                )

            self.history.append(metrics)

            # Checkpoint
            if checkpoint_dir and epoch % eval_every == 0:
                ckpt_path = Path(checkpoint_dir) / f"model_epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "baseline": self.baseline,
                    "metrics": metrics,
                    "mode": self.model.mode,
                    "train_sample_ids": self.train_sample_ids,
                    "val_sample_ids": self.val_sample_ids,
                }, str(ckpt_path))

        # Save final model
        if checkpoint_dir:
            final_path = Path(checkpoint_dir) / "model_final.pt"
            save_dict = {
                "epoch": num_epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "baseline": self.baseline,
                "history": self.history,
                "mode": self.model.mode,
                "train_sample_ids": self.train_sample_ids,
                "val_sample_ids": self.val_sample_ids,
            }
            # Save graph_edge_counts for hybrid model reconstruction
            if self.model.mode == "hybrid" and hasattr(self.model, 'edge_embeddings_dict'):
                save_dict["graph_edge_counts"] = {
                    int(sid): emb.num_embeddings
                    for sid, emb in self.model.edge_embeddings_dict.items()
                }
            torch.save(save_dict, str(final_path))
            logger.info(f"Saved final model to {final_path}")

        return self.history
