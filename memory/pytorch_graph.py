"""
PyTorch Graph Representation for RL Training (Phase 2)

Converts HAGE's JSON graph dumps into vectorized PyTorch tensors suitable
for fast, GPU-accelerated RL training with zero LLM calls.

Each sample produces a PyTorchGraph containing:
  - node_embeddings: (num_nodes, 384) frozen tensor
  - edge_index: (2, num_edges) COO-format edge list
  - edge_type_ids: (num_edges,) integer edge-type labels
  - edge_features: (num_edges, 4) static features from Phase 1 LLM scores
  - dia_id_to_node: mapping from evidence dia_ids to node indices
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Canonical ordering — do not change; downstream code relies on this.
EDGE_TYPE_TO_IDX = {
    "TEMPORAL": 0,
    "SEMANTIC": 1,
    "CAUSAL": 2,
    "ENTITY": 3,
}
NUM_EDGE_TYPES = len(EDGE_TYPE_TO_IDX)


class PyTorchGraph:
    """
    Immutable, vectorised representation of a single HAGE sample graph.

    All tensors live on *device* and are created once at construction time.
    """

    def __init__(
        self,
        node_embeddings: torch.Tensor,       # (N, 384) float32
        edge_index: torch.Tensor,            # (2, E) long
        edge_type_ids: torch.Tensor,         # (E,) long
        edge_features: torch.Tensor,         # (E, 4) float32
        node_id_to_idx: Dict[str, int],      # uuid -> index
        dia_id_to_node_idx: Dict[str, int],  # "D1:3" -> index
        sample_id: int,
        device: torch.device,
    ):
        self.node_embeddings = node_embeddings.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_type_ids = edge_type_ids.to(device)
        self.edge_features = edge_features.to(device)

        self.node_id_to_idx = node_id_to_idx
        self.dia_id_to_node_idx = dia_id_to_node_idx
        self.sample_id = sample_id
        self.device = device

        self.num_nodes = node_embeddings.size(0)
        self.num_edges = edge_index.size(1)

        # Pre-compute adjacency list (CPU dict for quick neighbor lookup)
        self._adj: Dict[int, List[Tuple[int, int]]] = {}  # src -> [(dst, edge_idx)]
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        for eidx, (s, d) in enumerate(zip(src, dst)):
            self._adj.setdefault(s, []).append((d, eidx))

    # ------------------------------------------------------------------
    # Convenience helpers used by the RL env
    # ------------------------------------------------------------------

    def neighbors(self, node_idx: int) -> List[Tuple[int, int]]:
        """Return [(neighbor_idx, edge_idx), …] for *node_idx*."""
        return self._adj.get(node_idx, [])

    def neighbor_indices(self, node_idx: int) -> torch.Tensor:
        """Return a 1-D long tensor of neighbor node indices."""
        nbrs = self._adj.get(node_idx, [])
        if not nbrs:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.tensor([n for n, _ in nbrs], dtype=torch.long, device=self.device)

    def edge_indices_from(self, node_idx: int) -> torch.Tensor:
        """Return a 1-D long tensor of edge indices leaving *node_idx*."""
        nbrs = self._adj.get(node_idx, [])
        if not nbrs:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.tensor([e for _, e in nbrs], dtype=torch.long, device=self.device)

    def resolve_evidence(self, evidence_list: List[str]) -> List[int]:
        """Map QA evidence strings (e.g. ['D1:3', 'D2:8']) to node indices."""
        indices = []
        for ev in evidence_list:
            idx = self.dia_id_to_node_idx.get(ev)
            if idx is not None:
                indices.append(idx)
        return indices

    def __repr__(self) -> str:
        return (
            f"PyTorchGraph(sample={self.sample_id}, "
            f"nodes={self.num_nodes}, edges={self.num_edges}, "
            f"device={self.device})"
        )


# ======================================================================
# Builder: JSON graph ➜ PyTorchGraph
# ======================================================================

def load_graph_json(graph_path: str) -> dict:
    """Load a HAGE graph.json file."""
    with open(graph_path, "r") as f:
        return json.load(f)


def _build_node_tensors(
    nodes: List[dict],
    embedding_dim: int = 384,
) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, int]]:
    """
    Extract EVENT nodes with valid embeddings.

    Returns:
        node_embeddings (N, 384), node_id_to_idx, dia_id_to_node_idx
    """
    embeddings = []
    node_id_to_idx: Dict[str, int] = {}
    dia_id_to_node_idx: Dict[str, int] = {}

    for node in nodes:
        # Only include EVENT nodes with correct-dimension embeddings
        if node.get("node_type") != "EVENT":
            continue
        emb = node.get("embedding_vector")
        if emb is None or len(emb) != embedding_dim:
            continue

        idx = len(embeddings)
        embeddings.append(emb)
        node_id_to_idx[node["node_id"]] = idx

        dia_id = node.get("attributes", {}).get("dia_id")
        if dia_id:
            dia_id_to_node_idx[dia_id] = idx

    if not embeddings:
        raise ValueError("No valid EVENT nodes found in graph")

    node_embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return node_embeddings, node_id_to_idx, dia_id_to_node_idx


def _build_edge_tensors(
    links: List[dict],
    node_id_to_idx: Dict[str, int],
    llm_score_cache: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edge tensors, keeping only edges whose endpoints are in *node_id_to_idx*.

    Returns:
        edge_index (2, E), edge_type_ids (E,), edge_features (E, 4)
    """
    src_list: List[int] = []
    dst_list: List[int] = []
    type_list: List[int] = []
    feat_list: List[List[float]] = []

    # Default feature vector (uniform) when no LLM scores available
    default_feat = [0.25] * NUM_EDGE_TYPES

    for link in links:
        s_id = link.get("source_node_id", "")
        d_id = link.get("target_node_id", "")
        if s_id not in node_id_to_idx or d_id not in node_id_to_idx:
            continue

        lt = link.get("link_type", "")
        # Fix mislabeled entity links: SAME_ENTITY sub_type stored as SEMANTIC
        if lt == "SEMANTIC" and link.get("properties", {}).get("sub_type") == "SAME_ENTITY":
            lt = "ENTITY"
        if lt not in EDGE_TYPE_TO_IDX:
            continue

        s_idx = node_id_to_idx[s_id]
        d_idx = node_id_to_idx[d_id]
        type_idx = EDGE_TYPE_TO_IDX[lt]

        # Edge feature: try Phase 1 LLM cache first, else use edge-type one-hot
        feat = None
        if llm_score_cache:
            link_id = link.get("link_id", "")
            cached = llm_score_cache.get(link_id)
            if cached:
                feat = [
                    cached.get("SEMANTIC", 0.25),
                    cached.get("TEMPORAL", 0.25),
                    cached.get("CAUSAL", 0.25),
                    cached.get("ENTITY", 0.25),
                ]
        if feat is None:
            # One-hot from edge type — gives the router real signal
            feat = [0.0] * NUM_EDGE_TYPES
            feat[type_idx] = 1.0

        src_list.append(s_idx)
        dst_list.append(d_idx)
        type_list.append(type_idx)
        feat_list.append(feat)

    if not src_list:
        raise ValueError("No valid edges found in graph")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type_ids = torch.tensor(type_list, dtype=torch.long)
    edge_features = torch.tensor(feat_list, dtype=torch.float32)

    return edge_index, edge_type_ids, edge_features


def build_pytorch_graph(
    graph_json: dict,
    sample_id: int,
    llm_score_cache: Optional[Dict[str, Dict[str, float]]] = None,
    device: str = "cpu",
    embedding_dim: int = 384,
) -> PyTorchGraph:
    """
    Convert a single HAGE graph.json dict into a PyTorchGraph.

    Args:
        graph_json: Loaded JSON dict with 'nodes' and 'links' keys.
        sample_id: Integer sample identifier.
        llm_score_cache: Optional Phase 1 LLM scores (query_weights_v2.json).
        device: Target torch device.
        embedding_dim: Expected embedding dimension (384 for MiniLM).

    Returns:
        PyTorchGraph ready for RL training.
    """
    dev = torch.device(device)

    node_embeddings, node_id_to_idx, dia_id_to_node_idx = _build_node_tensors(
        graph_json["nodes"], embedding_dim
    )
    edge_index, edge_type_ids, edge_features = _build_edge_tensors(
        graph_json["links"], node_id_to_idx, llm_score_cache
    )

    # Freeze node embeddings (they are never trained)
    node_embeddings.requires_grad_(False)

    graph = PyTorchGraph(
        node_embeddings=node_embeddings,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_features=edge_features,
        node_id_to_idx=node_id_to_idx,
        dia_id_to_node_idx=dia_id_to_node_idx,
        sample_id=sample_id,
        device=dev,
    )

    logger.info(
        f"Built {graph} | dia_id mappings: {len(dia_id_to_node_idx)}"
    )
    return graph


# ======================================================================
# Multi-graph loader
# ======================================================================

def load_all_graphs(
    cache_dir: str = "locomo_trg_cache_gpt_4o_mini",
    num_samples: int = 10,
    llm_score_path: Optional[str] = "edge_score_cache/query_weights_v2.json",
    device: str = "cpu",
) -> List[PyTorchGraph]:
    """
    Load all 10 HAGE sample graphs as PyTorchGraph objects.

    Args:
        cache_dir: Root directory containing sample0/graph.json, etc.
        num_samples: Number of samples to load (default 10).
        llm_score_path: Path to Phase 1 LLM score cache JSON.
        device: Target torch device.

    Returns:
        List of PyTorchGraph, one per sample.
    """
    cache_root = Path(cache_dir)

    # Load LLM score cache (optional)
    llm_scores = None
    if llm_score_path and Path(llm_score_path).exists():
        with open(llm_score_path, "r") as f:
            llm_scores = json.load(f)
        logger.info(f"Loaded {len(llm_scores)} LLM score entries")

    graphs: List[PyTorchGraph] = []
    for sid in range(num_samples):
        graph_path = cache_root / f"sample{sid}" / "graph.json"
        if not graph_path.exists():
            logger.warning(f"Graph not found: {graph_path}, skipping sample {sid}")
            continue

        graph_json = load_graph_json(str(graph_path))
        pg = build_pytorch_graph(
            graph_json, sample_id=sid, llm_score_cache=llm_scores, device=device
        )
        graphs.append(pg)

    logger.info(f"Loaded {len(graphs)} graphs")
    return graphs


# ======================================================================
# QA Dataset: maps questions → (start_node, target_nodes)
# ======================================================================

class QADataset:
    """
    Pairs each QA question with its graph and target node(s).

    For RL training, each item is:
      (graph, query_embedding, target_node_indices, qa_dict)
    """

    def __init__(
        self,
        locomo_data: List[dict],
        graphs: List[PyTorchGraph],
        encoder=None,
    ):
        """
        Args:
            locomo_data: Raw LoCoMo JSON list (10 samples).
            graphs: Corresponding PyTorchGraph list.
            encoder: Sentence-transformer encoder (VectorEncoder) for queries.
                     If None, query embeddings must be provided externally.
        """
        self.items: List[dict] = []
        self.encoder = encoder

        graph_by_sid = {g.sample_id: g for g in graphs}

        for sample in locomo_data:
            sid = sample.get("sample_id")
            if sid is None:
                continue
            graph = graph_by_sid.get(sid)
            if graph is None:
                continue

            for qa in sample.get("qa", []):
                # Some QAs only have adversarial_answer, skip those
                answer = qa.get("answer")
                if not answer:
                    continue

                evidence = qa.get("evidence", [])
                target_indices = graph.resolve_evidence(evidence)
                if not target_indices:
                    continue  # skip QAs we can't ground

                item = {
                    "sample_id": sid,
                    "graph": graph,
                    "question": qa["question"],
                    "answer": answer,
                    "category": qa.get("category"),
                    "evidence": evidence,
                    "target_node_indices": target_indices,
                    "query_embedding": None,  # lazily computed
                }
                self.items.append(item)

        logger.info(
            f"QADataset: {len(self.items)} grounded QA items "
            f"from {len(graph_by_sid)} graphs"
        )

    def encode_queries(self):
        """Pre-compute query embeddings for all items (requires encoder)."""
        if self.encoder is None:
            raise RuntimeError("No encoder provided — pass one to the constructor")

        questions = [it["question"] for it in self.items]
        # Batch encode
        embeddings = self.encoder.encode(questions)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        for i, item in enumerate(self.items):
            emb = embeddings[i] if isinstance(embeddings, np.ndarray) else embeddings[i].numpy()
            item["query_embedding"] = torch.tensor(emb, dtype=torch.float32)

    def split_by_sample(
        self, train_ids: List[int], val_ids: List[int]
    ) -> Tuple["QADataset", "QADataset"]:
        """Split items by sample_id into train and val sets."""
        train_ds = QADataset.__new__(QADataset)
        train_ds.items = [it for it in self.items if it["sample_id"] in train_ids]
        train_ds.encoder = self.encoder

        val_ds = QADataset.__new__(QADataset)
        val_ds.items = [it for it in self.items if it["sample_id"] in val_ids]
        val_ds.encoder = self.encoder

        logger.info(
            f"Split: train={len(train_ds)} items ({train_ids}), "
            f"val={len(val_ds)} items ({val_ids})"
        )
        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]
