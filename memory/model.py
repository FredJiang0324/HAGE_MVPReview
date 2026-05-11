"""
CoEvoMem: Dual-Mode RL Edge Scorer (Phase 2)

Unified model with two modes:
  - Transductive (Mode 2.1): Learnable per-edge embeddings for a specific graph.
    Parameter: nn.Embedding(num_edges, 4)
  - Inductive (Mode 2.2): Universal QueryRouter MLP that generalises to unseen graphs.
    Parameter: QueryRouter MLP; edge features frozen from Phase 1.

Forward logic:
  1. sim_scores = cosine(query_emb, neighbor_embeddings)
  2. w_ij = edge weight from active mode
  3. transition_probs = Softmax(λ·sim_scores + (1-λ)·w_ij)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .pytorch_graph import PyTorchGraph, NUM_EDGE_TYPES


class QueryRouter(nn.Module):
    """
    Inductive MLP that predicts edge importance from
    (query_embedding, edge_features) — generalises across graphs.

    Input:  query_emb (B, 384) + edge_feat (B, 4) → (B, 1)
    """

    def __init__(self, query_dim: int = 384, edge_feat_dim: int = NUM_EDGE_TYPES + 2):
        super().__init__()
        in_dim = query_dim + edge_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, query_emb: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: (B, 384)
            edge_feat: (B, 4)
        Returns:
            weights: (B,) positive edge weights
        """
        x = torch.cat([query_emb, edge_feat], dim=-1)
        return F.softplus(self.net(x).squeeze(-1))  # ensure positive


class CoEvoMem(nn.Module):
    """
    Multi-mode RL edge scorer.

    Args:
        mode: "transductive", "inductive", or "hybrid"
        num_edges: Total edges in the target graph (transductive only).
        edge_embed_dim: Dimension of learnable edge embeddings (transductive).
        query_dim: Dimension of query/node embeddings (384 for MiniLM).
        init_edge_features: (num_edges, 4) tensor to warm-start transductive
                            embeddings from Phase 1 LLM scores.
        graph_edge_counts: Dict[int, int] mapping sample_id -> num_edges (hybrid only).
        graph_init_features: Dict[int, Tensor] mapping sample_id -> (E, 4) init features (hybrid only).
    """

    def __init__(
        self,
        mode: str = "transductive",
        num_edges: int = 0,
        edge_embed_dim: int = NUM_EDGE_TYPES,
        query_dim: int = 384,
        init_edge_features: Optional[torch.Tensor] = None,
        graph_edge_counts: Optional[dict] = None,
        graph_init_features: Optional[dict] = None,
        lam: float = 0.5,
    ):
        super().__init__()
        assert mode in ("transductive", "inductive", "coevo", "hybrid")
        self.mode = mode
        self.query_dim = query_dim
        self.edge_embed_dim = edge_embed_dim
        self.lam = lam  # λ: balance between semantic similarity and structural weight

        if mode == "transductive":
            assert num_edges > 0, "Transductive mode requires num_edges > 0"
            self.edge_embeddings = nn.Embedding(num_edges, edge_embed_dim)
            # Warm-start from Phase 1 LLM scores if available
            if init_edge_features is not None:
                with torch.no_grad():
                    self.edge_embeddings.weight.copy_(init_edge_features)
        elif mode == "inductive":
            # Router input: edge_type one-hot (4) + sim_source (1) + sim_target (1) = 6
            self.query_router = QueryRouter(query_dim=query_dim, edge_feat_dim=edge_embed_dim + 2)
        elif mode == "coevo":
            # Co-evolution: jointly train edge features + router
            # Edge features are nn.Parameter (not frozen), router reads them
            # Router input: learnable edge feat (4) + runtime sims (2) = 6
            self.query_router = QueryRouter(query_dim=query_dim, edge_feat_dim=edge_embed_dim + 2)
            # Per-graph learnable edge features (warm-started from Phase 1)
            self.edge_features_dict = nn.ParameterDict()
            # Frozen copies of init values for anchor regularization
            self._edge_features_init = {}
            if graph_init_features:
                for sid, feat_tensor in graph_init_features.items():
                    self.edge_features_dict[str(sid)] = nn.Parameter(
                        feat_tensor.clone(), requires_grad=True
                    )
                    self.register_buffer(
                        f"_edge_init_{sid}",
                        feat_tensor.clone(),
                    )
                    self._edge_features_init[str(sid)] = f"_edge_init_{sid}"
        elif mode == "hybrid":
            # Shared router (generalizes across graphs)
            self.query_router = QueryRouter(query_dim=query_dim, edge_feat_dim=edge_embed_dim + 2)
            # Per-graph edge embeddings (graph-specific fine-tuning)
            self.edge_embeddings_dict = nn.ModuleDict()
            if graph_edge_counts:
                for sid, n_edges in graph_edge_counts.items():
                    emb = nn.Embedding(n_edges, edge_embed_dim)
                    if graph_init_features and sid in graph_init_features:
                        with torch.no_grad():
                            emb.weight.copy_(graph_init_features[sid])
                    self.edge_embeddings_dict[str(sid)] = emb
            # Learnable gate: how much to trust edge embeddings vs router
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        graph: PyTorchGraph,
        current_node: int,
        query_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute transition probabilities over neighbours of *current_node*.

        Args:
            graph: The PyTorchGraph being traversed.
            current_node: Index of the current node.
            query_emb: (384,) query embedding.

        Returns:
            probs: (num_neighbors,) probability distribution over neighbours.
                   Empty tensor if node is a dead end.
        """
        neighbor_data = graph.neighbors(current_node)
        if not neighbor_data:
            return torch.zeros(0, device=query_emb.device)

        nbr_indices = torch.tensor(
            [n for n, _ in neighbor_data], dtype=torch.long, device=graph.device
        )
        edge_indices = torch.tensor(
            [e for _, e in neighbor_data], dtype=torch.long, device=graph.device
        )

        # 1. Cosine similarity between query and each neighbour
        nbr_embs = graph.node_embeddings[nbr_indices]          # (K, 384)
        q = query_emb.unsqueeze(0)                              # (1, 384)
        sim_scores = F.cosine_similarity(q, nbr_embs, dim=-1)  # (K,)

        # 2. Edge weights w_ij (mode-dependent)
        w_ij = self._compute_edge_weights(
            graph, current_node, nbr_indices, edge_indices, query_emb, len(neighbor_data)
        )

        # 3. Transition scores: additive combination (Eq. 8 in paper)
        # S(n_j | n_i, q) = λ·cos(v_j, q) + (1-λ)·w_ij(q)
        logits = self.lam * sim_scores + (1 - self.lam) * w_ij
        probs = F.softmax(logits, dim=0)

        return probs

    def _enrich_edge_features(self, graph, current_node, nbr_indices, edge_feat, query_emb):
        """
        Append runtime per-edge similarity features to static edge type features.

        Adds:
          - sim_source: cosine(query, current_node)  — broadcast to all edges
          - sim_target: cosine(query, neighbor_node)  — unique per edge

        Input edge_feat: (K, 4), Output: (K, 6)
        """
        q = query_emb.unsqueeze(0)  # (1, 384)
        # Source similarity (same for all edges from this node)
        src_emb = graph.node_embeddings[current_node].unsqueeze(0)  # (1, 384)
        sim_src = F.cosine_similarity(q, src_emb, dim=-1)           # (1,)
        sim_src = sim_src.expand(edge_feat.size(0), 1)              # (K, 1)
        # Target similarity (unique per neighbor)
        tgt_embs = graph.node_embeddings[nbr_indices]               # (K, 384)
        sim_tgt = F.cosine_similarity(q, tgt_embs, dim=-1)         # (K,)
        sim_tgt = sim_tgt.unsqueeze(-1)                             # (K, 1)
        return torch.cat([edge_feat, sim_src, sim_tgt], dim=-1)     # (K, 6)

    def _compute_edge_weights(self, graph, current_node, nbr_indices, edge_indices, query_emb, num_neighbors):
        """Compute edge weights based on mode."""
        if self.mode == "transductive":
            edge_emb = self.edge_embeddings(edge_indices)       # (K, 4)
            w_ij = edge_emb.norm(dim=-1)                        # (K,)
            w_ij = F.softplus(w_ij)                             # positive
        elif self.mode == "inductive":
            edge_feat = graph.edge_features[edge_indices]       # (K, 4)
            edge_feat = self._enrich_edge_features(graph, current_node, nbr_indices, edge_feat, query_emb)
            q_expand = query_emb.unsqueeze(0).expand(num_neighbors, -1)
            w_ij = self.query_router(q_expand, edge_feat)       # (K,)
        elif self.mode == "coevo":
            # Co-evolution: edge features are learnable parameters
            sid = str(graph.sample_id)
            if sid in self.edge_features_dict:
                edge_feat = self.edge_features_dict[sid][edge_indices]  # (K, 4) — grad flows
            else:
                edge_feat = graph.edge_features[edge_indices]  # unseen graph: static
            edge_feat = self._enrich_edge_features(graph, current_node, nbr_indices, edge_feat, query_emb)
            q_expand = query_emb.unsqueeze(0).expand(num_neighbors, -1)
            w_ij = self.query_router(q_expand, edge_feat)      # (K,)
        elif self.mode == "hybrid":
            # Router component (always active, generalizes to unseen graphs)
            edge_feat = graph.edge_features[edge_indices]
            edge_feat = self._enrich_edge_features(graph, current_node, nbr_indices, edge_feat, query_emb)
            q_expand = query_emb.unsqueeze(0).expand(num_neighbors, -1)
            w_router = self.query_router(q_expand, edge_feat)

            # Edge embedding component (only for seen/training graphs)
            sid = str(graph.sample_id)
            if sid in self.edge_embeddings_dict:
                edge_emb = self.edge_embeddings_dict[sid](edge_indices)
                w_edge = F.softplus(edge_emb.norm(dim=-1))
                # Learned gate blends the two: gate * edge + (1-gate) * router
                g = torch.sigmoid(self.gate)
                w_ij = g * w_edge + (1.0 - g) * w_router
            else:
                # Unseen graph at inference: router only
                w_ij = w_router
        return w_ij

    def forward_logits(
        self,
        graph: PyTorchGraph,
        current_node: int,
        query_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return raw logits (λ·sim + (1-λ)·w_ij) before softmax.

        Used by the adapter to produce per-edge scores in [0, 1]
        without the peaky softmax distribution.
        """
        neighbor_data = graph.neighbors(current_node)
        if not neighbor_data:
            return torch.zeros(0, device=query_emb.device)

        nbr_indices = torch.tensor(
            [n for n, _ in neighbor_data], dtype=torch.long, device=graph.device
        )
        edge_indices = torch.tensor(
            [e for _, e in neighbor_data], dtype=torch.long, device=graph.device
        )

        nbr_embs = graph.node_embeddings[nbr_indices]
        q = query_emb.unsqueeze(0)
        sim_scores = F.cosine_similarity(q, nbr_embs, dim=-1)

        w_ij = self._compute_edge_weights(
            graph, current_node, nbr_indices, edge_indices, query_emb, len(neighbor_data)
        )

        return self.lam * sim_scores + (1 - self.lam) * w_ij

    def get_action(
        self,
        graph: PyTorchGraph,
        current_node: int,
        query_emb: torch.Tensor,
        greedy: bool = False,
    ) -> tuple:
        """
        Select next node to visit.

        Returns:
            (action_idx, log_prob, probs)
            action_idx: index into the neighbor list (not the global node id)
            log_prob: log probability of the selected action
            probs: full probability vector
        """
        probs = self.forward(graph, current_node, query_emb)
        if probs.numel() == 0:
            return -1, torch.tensor(0.0, device=query_emb.device), probs

        if greedy:
            action_idx = probs.argmax().item()
            log_prob = probs[action_idx].log()
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx, device=probs.device))

        return action_idx, log_prob, probs

    def edge_anchor_loss(self) -> torch.Tensor:
        """
        L2 regularization: penalize edge features for drifting from Phase 1 init.

        Keeps the learned edge feature distribution close to the static features
        that unseen (val) graphs will have, preventing distribution shift.

        Returns 0 for non-coevo modes.
        """
        if self.mode != "coevo" or not self._edge_features_init:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for sid, buf_name in self._edge_features_init.items():
            current = self.edge_features_dict[sid]
            init = getattr(self, buf_name)
            loss = loss + F.mse_loss(current, init)
        return loss


def build_transductive_model(graph: PyTorchGraph, device: str = "cpu") -> CoEvoMem:
    """
    Factory: create a Transductive CoEvoMem warm-started from a graph's
    Phase 1 edge features.
    """
    model = CoEvoMem(
        mode="transductive",
        num_edges=graph.num_edges,
        init_edge_features=graph.edge_features,
    )
    return model.to(device)


def build_inductive_model(device: str = "cpu") -> CoEvoMem:
    """Factory: create an Inductive CoEvoMem (graph-agnostic)."""
    model = CoEvoMem(mode="inductive")
    return model.to(device)


def build_coevo_model(graphs: list, device: str = "cpu") -> CoEvoMem:
    """
    Factory: create a CoEvo model — jointly trains edge features + router.

    Args:
        graphs: List of PyTorchGraph for training samples.
                Edge features are warm-started from Phase 1 and made learnable.
    """
    graph_init_features = {g.sample_id: g.edge_features for g in graphs}
    model = CoEvoMem(
        mode="coevo",
        graph_init_features=graph_init_features,
    )
    return model.to(device)


def build_hybrid_model(graphs: list, device: str = "cpu") -> CoEvoMem:
    """
    Factory: create a Hybrid CoEvoMem with per-graph edge embeddings + shared router.

    Args:
        graphs: List of PyTorchGraph for training samples.
                Each graph's edge embeddings are warm-started from Phase 1 features.
    """
    graph_edge_counts = {g.sample_id: g.num_edges for g in graphs}
    graph_init_features = {g.sample_id: g.edge_features for g in graphs}
    model = CoEvoMem(
        mode="hybrid",
        graph_edge_counts=graph_edge_counts,
        graph_init_features=graph_init_features,
    )
    return model.to(device)
