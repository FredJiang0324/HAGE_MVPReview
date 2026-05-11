"""
RL Edge Scorer Adapter

Wraps a trained CoEvoMem model behind the EdgeScorerInterface so it can
be used as a drop-in replacement inside QueryEngine / test_fixed_memory.py.

Usage:
    from memory.rl_edge_adapter import RLEdgeScorerAdapter

    adapter = RLEdgeScorerAdapter.from_checkpoint(
        checkpoint_path="checkpoints/transductive_s0/model_final.pt",
        graph_cache_dir="locomo_trg_cache_gpt_4o_mini",
        sample_id=0,
        mode="transductive",
    )
    # Pass to QueryEngine
    query_engine = QueryEngine(trg, node_index, edge_scorer=adapter)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch

from .edge_scorer import EdgeScorerInterface, StaticEdgeScorer
from .graph_db import EventNode, LinkType
from .pytorch_graph import (
    PyTorchGraph,
    build_pytorch_graph,
    load_graph_json,
    EDGE_TYPE_TO_IDX,
)
from .model import CoEvoMem, build_transductive_model, build_inductive_model, build_hybrid_model, build_coevo_model

logger = logging.getLogger(__name__)


class RLEdgeScorerAdapter(EdgeScorerInterface):
    """
    Adapts a trained CoEvoMem model to the EdgeScorerInterface.

    Translates EventNode UUIDs to PyTorch graph indices, runs the model's
    forward pass, and returns a [0, 1] edge importance score.
    """

    def __init__(
        self,
        model: CoEvoMem,
        pytorch_graph: PyTorchGraph,
        fallback_scorer: Optional[EdgeScorerInterface] = None,
        device: str = "cpu",
        alpha: float = 1.0,
    ):
        """
        Args:
            alpha: Blending weight for RL vs static scores.
                   1.0 = pure RL, 0.0 = pure static,
                   0.5 = equal blend of RL + static.
        """
        self.model = model
        self.model.eval()
        self.graph = pytorch_graph
        self.device = torch.device(device)
        self.fallback = fallback_scorer or StaticEdgeScorer()
        self.alpha = alpha

        # Build reverse lookup: uuid -> node index
        self.uuid_to_idx: Dict[str, int] = pytorch_graph.node_id_to_idx

        # Build edge lookup: (src_idx, dst_idx) -> list of edge indices
        self._edge_lookup: Dict[Tuple[int, int], List[int]] = {}
        src = pytorch_graph.edge_index[0].cpu().tolist()
        dst = pytorch_graph.edge_index[1].cpu().tolist()
        for eidx, (s, d) in enumerate(zip(src, dst)):
            self._edge_lookup.setdefault((s, d), []).append(eidx)

        # Stats
        self.rl_hits = 0
        self.fallback_hits = 0

        logger.info(
            f"RLEdgeScorerAdapter ready | mode={model.mode} | "
            f"nodes={pytorch_graph.num_nodes} | edges={pytorch_graph.num_edges}"
        )

    def score_edge(
        self,
        source_node: EventNode,
        target_node: EventNode,
        edge_type: LinkType,
        query_text: str,
        query_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Score an edge using the RL model.

        Falls back to StaticEdgeScorer if either node is not in the
        PyTorch graph (e.g. SESSION nodes were excluded).
        """
        src_idx = self.uuid_to_idx.get(source_node.node_id)
        dst_idx = self.uuid_to_idx.get(target_node.node_id)

        # Fall back if nodes aren't in the RL graph
        if src_idx is None or dst_idx is None:
            self.fallback_hits += 1
            return self.fallback.score_edge(
                source_node, target_node, edge_type,
                query_text, query_embedding, context,
            )

        # Check if this edge exists in the PyTorch graph
        edge_indices = self._edge_lookup.get((src_idx, dst_idx))
        if not edge_indices:
            # Edge not in graph — still compute a score using node similarity
            # weighted by the model's general tendency
            self.fallback_hits += 1
            return self.fallback.score_edge(
                source_node, target_node, edge_type,
                query_text, query_embedding, context,
            )

        self.rl_hits += 1

        # Convert query embedding to tensor
        if isinstance(query_embedding, np.ndarray):
            q_emb = torch.tensor(query_embedding, dtype=torch.float32, device=self.device)
        else:
            q_emb = query_embedding.to(self.device)

        # Ensure 1D
        if q_emb.dim() == 2:
            q_emb = q_emb[0]

        # Run the model to get raw logits (pre-softmax) for better score spread
        with torch.no_grad():
            logits = self.model.forward_logits(self.graph, src_idx, q_emb)

        if logits.numel() == 0:
            self.fallback_hits += 1
            return self.fallback.score_edge(
                source_node, target_node, edge_type,
                query_text, query_embedding, context,
            )

        # Find which neighbor index corresponds to dst_idx
        neighbors = self.graph.neighbors(src_idx)  # [(dst, edge_idx), ...]
        rl_score = None
        for i, (nbr_idx, _) in enumerate(neighbors):
            if nbr_idx == dst_idx:
                # Normalize logits to [0, 1] using min-max over neighbors
                min_l = logits.min().item()
                max_l = logits.max().item()
                if max_l - min_l > 1e-8:
                    rl_score = (logits[i].item() - min_l) / (max_l - min_l)
                else:
                    rl_score = 0.5
                break

        if rl_score is None:
            # dst_idx is not a direct neighbor of src_idx in the RL graph
            self.fallback_hits += 1
            return self.fallback.score_edge(
                source_node, target_node, edge_type,
                query_text, query_embedding, context,
            )

        # Blend RL score with static score
        if self.alpha < 1.0:
            static_score = self.fallback.score_edge(
                source_node, target_node, edge_type,
                query_text, query_embedding, context,
            )
            blended = self.alpha * rl_score + (1.0 - self.alpha) * static_score
            return float(np.clip(blended, 0.0, 1.0))

        return float(np.clip(rl_score, 0.0, 1.0))

    def get_stats(self) -> Dict[str, int]:
        """Return adapter usage statistics."""
        total = self.rl_hits + self.fallback_hits
        return {
            "rl_hits": self.rl_hits,
            "fallback_hits": self.fallback_hits,
            "total": total,
            "rl_hit_rate": self.rl_hits / total if total > 0 else 0.0,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        graph_cache_dir: str = "locomo_trg_cache_gpt_4o_mini",
        sample_id: int = 0,
        mode: str = "transductive",
        llm_score_path: Optional[str] = "edge_score_cache/query_weights_v2.json",
        device: str = "cpu",
        alpha: float = 1.0,
        train_sample_ids: Optional[list] = None,
    ) -> "RLEdgeScorerAdapter":
        """
        Load a trained model from checkpoint and build the adapter.

        Args:
            checkpoint_path: Path to model_final.pt or model_epoch_N.pt
            graph_cache_dir: Directory with sample{N}/graph.json files
            sample_id: Which sample's graph to load for evaluation
            mode: "transductive", "inductive", "coevo", "edge_only",
                  "router_only", or "hybrid"
            llm_score_path: Phase 1 LLM score cache
            device: torch device
            train_sample_ids: List of sample IDs used during training
                (needed for coevo/edge_only to reconstruct ParameterDict).
                Defaults to [0..7] if not specified.
        """
        from .pytorch_graph import load_all_graphs

        # Load graph for the target sample
        graph_path = Path(graph_cache_dir) / f"sample{sample_id}" / "graph.json"
        graph_json = load_graph_json(str(graph_path))

        llm_scores = None
        if llm_score_path and Path(llm_score_path).exists():
            with open(llm_score_path) as f:
                llm_scores = json.load(f)

        pytorch_graph = build_pytorch_graph(
            graph_json, sample_id=sample_id,
            llm_score_cache=llm_scores, device=device,
        )

        # Normalize mode aliases
        # router_only is inductive (edge features frozen on graph)
        # edge_only is coevo with router frozen
        actual_mode = mode
        if mode == "router_only":
            actual_mode = "inductive"
        elif mode == "edge_only":
            actual_mode = "coevo"  # same architecture, router was frozen during training

        # Load checkpoint early so we can read metadata for model reconstruction
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_train_ids = ckpt.get("train_sample_ids")
        ckpt_val_ids = ckpt.get("val_sample_ids")
        ckpt_mode = ckpt.get("mode", "unknown")
        ckpt_epoch = ckpt.get("epoch", "?")

        # For coevo/edge_only: prefer checkpoint's train IDs over caller's
        if actual_mode == "coevo" and train_sample_ids is None:
            if ckpt_train_ids is not None:
                train_sample_ids = ckpt_train_ids
            else:
                train_sample_ids = list(range(8))
                logger.warning(
                    "Checkpoint has no train_sample_ids metadata; "
                    "defaulting to [0..7]. Pass train_sample_ids to override."
                )

        # Build model matching the checkpoint's architecture
        if actual_mode == "transductive":
            model = build_transductive_model(pytorch_graph, device=device)
        elif actual_mode == "coevo":
            # coevo and edge_only both use build_coevo_model
            # Need all training graphs to reconstruct nn.ParameterDict
            all_graphs = load_all_graphs(device=device)
            train_graphs = [g for g in all_graphs if g.sample_id in train_sample_ids]
            model = build_coevo_model(train_graphs, device=device)
            if mode == "edge_only":
                for param in model.query_router.parameters():
                    param.requires_grad = False
        elif actual_mode == "hybrid":
            model = CoEvoMem(
                mode="hybrid",
                graph_edge_counts=ckpt.get("graph_edge_counts", {}),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model = model.to(device)
            logger.info(
                f"Loaded hybrid model from {checkpoint_path} "
                f"(epoch={ckpt_epoch})"
            )
            return cls(model=model, pytorch_graph=pytorch_graph, device=device, alpha=alpha)
        else:
            # inductive / router_only
            model = build_inductive_model(device=device)

        # Load weights into reconstructed model
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Log split metadata and warn about train/val contamination
        logger.info(
            f"Loaded {mode} model from {checkpoint_path} "
            f"(mode={ckpt_mode}, epoch={ckpt_epoch})"
        )
        if ckpt_train_ids is not None:
            logger.info(f"  Checkpoint train samples: {ckpt_train_ids}")
            logger.info(f"  Checkpoint val samples:   {ckpt_val_ids}")
            if sample_id in ckpt_train_ids:
                logger.warning(
                    f"  *** Sample {sample_id} was in the TRAINING set for this checkpoint. "
                    f"Results are NOT held-out. ***"
                )
            elif ckpt_val_ids and sample_id in ckpt_val_ids:
                logger.info(f"  Sample {sample_id} is held-out (val set).")
            else:
                logger.warning(
                    f"  Sample {sample_id} was in neither train nor val of this checkpoint."
                )
        else:
            logger.warning(
                f"  Checkpoint has no train/val split metadata. "
                f"Cannot verify whether sample {sample_id} is held-out."
            )

        return cls(model=model, pytorch_graph=pytorch_graph, device=device, alpha=alpha)
