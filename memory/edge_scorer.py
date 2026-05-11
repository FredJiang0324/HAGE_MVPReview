"""
Edge Scorer Module

Provides abstraction for computing edge importance scores in graph traversal.
Supports multiple scoring strategies:
- Static: Rule-based scoring matching current HAGE behavior
- LLM: Query-aware scoring using language models
- Hybrid: Combination of multiple scorers

Usage:
    scorer = StaticEdgeScorer()
    score = scorer.score_edge(source_node, target_node, edge_type, query, query_emb)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging
from pathlib import Path
import json
import hashlib

from .graph_db import EventNode, LinkType

logger = logging.getLogger(__name__)


@dataclass
class EdgeScorerConfig:
    """Configuration for edge scorers."""
    scorer_type: str = "static"  # "static", "llm", "hybrid"
    cache_dir: str = "./edge_score_cache"
    prompt_version: str = "v1"
    temperature: float = 0.0
    use_cache: bool = True
    fallback_to_static: bool = True

    # Hybrid config
    hybrid_llm_weight: float = 0.7
    hybrid_static_weight: float = 0.3

    # Static scorer config
    edge_type_weights: Optional[Dict[str, float]] = None


class EdgeScorerInterface(ABC):
    """
    Abstract interface for computing edge importance scores.

    All edge scorers must implement this interface to be compatible
    with the QueryEngine's graph traversal system.
    """

    @abstractmethod
    def score_edge(
        self,
        source_node: EventNode,
        target_node: EventNode,
        edge_type: LinkType,
        query_text: str,
        query_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute importance score for an edge given query context.

        Args:
            source_node: Starting node of edge (current position in traversal)
            target_node: Ending node of edge (potential next node)
            edge_type: Type of link (TEMPORAL, SEMANTIC, CAUSAL, ENTITY)
            query_text: Natural language query
            query_embedding: Vector embedding of query (normalized)
            context: Additional context (e.g., traversal history, depth)

        Returns:
            float: Edge importance score in [0.0, 1.0] range
                  - 1.0 = highly relevant, should definitely traverse
                  - 0.5 = moderately relevant, traverse if no better options
                  - 0.0 = irrelevant, skip this edge

        Example:
            >>> scorer = StaticEdgeScorer()
            >>> score = scorer.score_edge(
            ...     source_node=current_node,
            ...     target_node=neighbor_node,
            ...     edge_type=LinkType.SEMANTIC,
            ...     query_text="What did Alice buy?",
            ...     query_embedding=query_emb
            ... )
            >>> print(f"Edge score: {score:.3f}")
            Edge score: 0.847
        """
        pass

    def batch_score_edges(
        self,
        edges: List[Tuple[EventNode, EventNode, LinkType]],
        query_text: str,
        query_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Batch version for efficiency (optional to override).

        Default implementation calls score_edge sequentially.
        Subclasses can override for batch optimization (e.g., LLM batch API).

        Args:
            edges: List of (source_node, target_node, edge_type) tuples
            query_text: Natural language query
            query_embedding: Vector embedding of query
            context: Additional context

        Returns:
            List of float scores matching the input edge list order
        """
        scores = []
        for source, target, edge_type in edges:
            score = self.score_edge(
                source, target, edge_type,
                query_text, query_embedding, context
            )
            scores.append(score)
        return scores


class StaticEdgeScorer(EdgeScorerInterface):
    """
    Baseline edge scorer matching current HAGE behavior.

    Scoring logic:
        score = semantic_similarity(query, target_node) × edge_type_weight

    Where:
        - semantic_similarity: Cosine similarity between embeddings
        - edge_type_weight: Predefined importance weight for each link type

    This provides a strong baseline and fallback for LLM-based scorers.

    Attributes:
        edge_type_weights: Mapping of LinkType to importance multiplier
            Default weights are based on empirical performance:
            - SEMANTIC: 1.0 (highest priority - direct concept match)
            - CAUSAL: 0.9 (high priority - explains relationships)
            - TEMPORAL: 0.8 (medium-high - time-based connections)
            - ENTITY: 0.7 (medium - shared entity mentions)

    Example:
        >>> scorer = StaticEdgeScorer()
        >>> # Customize edge type weights
        >>> scorer = StaticEdgeScorer(edge_type_weights={
        ...     LinkType.SEMANTIC: 1.0,
        ...     LinkType.TEMPORAL: 0.9,  # Prioritize temporal for "when" questions
        ... })
    """

    def __init__(self, edge_type_weights: Optional[Dict[LinkType, float]] = None):
        """
        Initialize static edge scorer.

        Args:
            edge_type_weights: Optional custom edge type weights.
                             If None, uses default weights.
        """
        if edge_type_weights is None:
            # Default weights based on HAGE's empirical performance
            # These can be tuned based on evaluation results
            self.edge_type_weights = {
                LinkType.SEMANTIC: 1.0,   # Direct semantic match
                LinkType.CAUSAL: 0.9,     # Causal reasoning
                LinkType.TEMPORAL: 0.8,   # Temporal relationships
                LinkType.ENTITY: 0.7      # Entity co-occurrence
            }
        else:
            self.edge_type_weights = edge_type_weights

        logger.info(f"StaticEdgeScorer initialized with weights: {self.edge_type_weights}")

    def _ensure_numpy_array(self, vec):
        """Convert various embedding formats to numpy array."""
        if vec is None:
            return None

        # Already a numpy array
        if isinstance(vec, np.ndarray):
            return vec

        # Handle list
        if isinstance(vec, list):
            return np.array(vec)

        # Handle string representation of numpy array (common issue)
        if isinstance(vec, (str, np.str_)):
            try:
                # Remove 'np.str_' wrapper if present
                vec_str = str(vec)
                # Parse the array string
                if vec_str.startswith('[') and vec_str.endswith(']'):
                    # Simple parsing: split by whitespace and convert to floats
                    values = vec_str.strip('[]').replace('\n', ' ').split()
                    return np.array([float(v) for v in values if v])
                else:
                    logger.warning(f"Cannot parse embedding string: {vec_str[:50]}...")
                    return None
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to convert embedding string to array: {e}")
                return None

        # Try direct numpy array conversion
        try:
            return np.array(vec)
        except Exception as e:
            logger.warning(f"Cannot convert {type(vec)} to numpy array: {e}")
            return None

    def score_edge(
        self,
        source_node: EventNode,
        target_node: EventNode,
        edge_type: LinkType,
        query_text: str,
        query_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute static edge score based on semantic similarity and edge type.

        Returns:
            float: Score in [0.0, 1.0] range
        """
        # Get target node embedding
        if target_node.embedding_vector is None:
            logger.debug(f"Target node {target_node.node_id} has no embedding, returning 0.0")
            return 0.0

        target_embedding = self._ensure_numpy_array(target_node.embedding_vector)
        query_emb_arr = self._ensure_numpy_array(query_embedding)

        if target_embedding is None or query_emb_arr is None:
            logger.debug(f"Failed to convert embeddings to arrays, returning 0.0")
            return 0.0

        # Compute cosine similarity
        # Both embeddings should be normalized, but ensure numerical stability
        query_norm = np.linalg.norm(query_emb_arr)
        target_norm = np.linalg.norm(target_embedding)

        if query_norm < 1e-8 or target_norm < 1e-8:
            logger.debug(f"Zero norm detected, returning 0.0")
            return 0.0

        # Cosine similarity: dot product of normalized vectors
        similarity = np.dot(query_emb_arr, target_embedding) / (query_norm * target_norm + 1e-8)

        # Clamp to [0, 1] range (cosine can be negative if vectors are opposite)
        similarity = max(0.0, min(1.0, similarity))

        # Apply edge type weight
        edge_weight = self.edge_type_weights.get(edge_type, 0.5)

        # Final score
        score = similarity * edge_weight

        # Log for debugging (only for high-scoring edges to reduce noise)
        if score > 0.7:
            logger.debug(
                f"High score edge: {score:.3f} | "
                f"Type: {edge_type.value} | "
                f"Sim: {similarity:.3f} | "
                f"Weight: {edge_weight:.2f} | "
                f"Target: {(getattr(target_node, 'content_narrative', None) or getattr(target_node, 'summary', str(target_node)))[:50]}..."
            )

        return float(score)

    def get_edge_type_weights(self) -> Dict[LinkType, float]:
        """Get current edge type weights (useful for analysis)."""
        return self.edge_type_weights.copy()

    def set_edge_type_weights(self, weights: Dict[LinkType, float]):
        """Update edge type weights (useful for tuning)."""
        self.edge_type_weights = weights
        logger.info(f"Edge type weights updated: {weights}")


class HybridEdgeScorer(EdgeScorerInterface):
    """
    Hybrid scorer combining LLM and static scorers.

    Strategies:
    - weighted_average: Combine scores with configurable weights
    - llm_primary: Use LLM when available, fall back to static
    - selective: Use LLM only for ambiguous cases (0.3 < static_score < 0.7)

    Example:
        >>> hybrid = HybridEdgeScorer(
        ...     llm_scorer=llm_scorer,
        ...     static_scorer=static_scorer,
        ...     combination_method="weighted_average",
        ...     llm_weight=0.7
        ... )
    """

    def __init__(
        self,
        llm_scorer: EdgeScorerInterface,
        static_scorer: StaticEdgeScorer,
        combination_method: str = "weighted_average",
        llm_weight: float = 0.7
    ):
        """
        Initialize hybrid scorer.

        Args:
            llm_scorer: LLM-based scorer instance
            static_scorer: Static scorer instance
            combination_method: How to combine scores
                - "weighted_average": llm_weight * llm + (1 - llm_weight) * static
                - "llm_primary": Use LLM if available, else static
                - "selective": Use LLM only for ambiguous static scores
            llm_weight: Weight for LLM score (only for weighted_average)
        """
        self.llm_scorer = llm_scorer
        self.static_scorer = static_scorer
        self.combination_method = combination_method
        self.llm_weight = llm_weight
        self.static_weight = 1.0 - llm_weight

        logger.info(
            f"HybridEdgeScorer initialized | "
            f"Method: {combination_method} | "
            f"LLM weight: {llm_weight:.2f}"
        )

    def score_edge(
        self,
        source_node: EventNode,
        target_node: EventNode,
        edge_type: LinkType,
        query_text: str,
        query_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute hybrid edge score."""

        # Always compute static score (fast)
        static_score = self.static_scorer.score_edge(
            source_node, target_node, edge_type,
            query_text, query_embedding, context
        )

        if self.combination_method == "llm_primary":
            # Try LLM first, fall back to static
            try:
                return self.llm_scorer.score_edge(
                    source_node, target_node, edge_type,
                    query_text, query_embedding, context
                )
            except Exception:
                return static_score

        elif self.combination_method == "selective":
            # Use LLM only for ambiguous cases
            if 0.3 <= static_score <= 0.7:
                # Ambiguous - query LLM
                try:
                    return self.llm_scorer.score_edge(
                        source_node, target_node, edge_type,
                        query_text, query_embedding, context
                    )
                except Exception:
                    return static_score
            else:
                # Clear case - use static
                return static_score

        else:  # "weighted_average"
            # Compute both scores and combine
            try:
                llm_score = self.llm_scorer.score_edge(
                    source_node, target_node, edge_type,
                    query_text, query_embedding, context
                )
                combined = (self.llm_weight * llm_score +
                           self.static_weight * static_score)
                return float(combined)
            except Exception:
                return static_score
