"""
Edge Scorer V2: O(1) LLM-based Query-Level Weight Scorer

Pure architecture:
- 1 LLM call per unique query to get edge type weights
- Fast vectorized scoring: similarity × query_weights[edge_type]
- No per-edge LLM calls, no hybrid logic

"""

import hashlib
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from memory.graph_db import EventNode, LinkType

logger = logging.getLogger(__name__)


# Re-export base interfaces for compatibility
from memory.edge_scorer import EdgeScorerInterface, StaticEdgeScorer


class LLMEdgeScorerV2(EdgeScorerInterface):
    """
    O(1) LLM-based edge scorer using query-level weights.

    Architecture:
    1. Call LLM once per unique query → {SEMANTIC: 0.9, TEMPORAL: 0.5, CAUSAL: 0.2, ENTITY: 0.7}
    2. Score all edges fast: similarity × query_weights[edge_type]

    Benefits:
    - ~100x faster than per-edge LLM calls
    - ~50x cheaper API costs
    - Mathematically pure: global weights + fast linear algebra
    """

    # Prompt template for query-level weight extraction
    WEIGHT_PROMPT = """You are analyzing a question to determine which types of relationships are most important for finding the answer.

**Question:** {query_text}

Rate the importance of each edge type (0.0 to 1.0) for answering this question:

- **SEMANTIC**: Content/topic similarity between memory chunks
- **TEMPORAL**: Time sequence and ordering of events
- **CAUSAL**: Cause-effect relationships between events
- **ENTITY**: Connections through shared entities/objects

Think about what the question is asking:
- "What" questions → SEMANTIC edges (content)
- "When" questions → TEMPORAL edges (time)
- "Why" questions → CAUSAL edges (reasons)
- "Who/Where" questions → ENTITY edges (objects/places)

Return ONLY a valid JSON object with exactly these 4 keys:
{{"SEMANTIC": <float>, "TEMPORAL": <float>, "CAUSAL": <float>, "ENTITY": <float>}}

Example outputs:
- "What did Alice buy?" → {{"SEMANTIC": 0.95, "TEMPORAL": 0.4, "CAUSAL": 0.2, "ENTITY": 0.7}}
- "Why did Bob leave?" → {{"SEMANTIC": 0.6, "TEMPORAL": 0.3, "CAUSAL": 0.95, "ENTITY": 0.4}}
- "When did the meeting happen?" → {{"SEMANTIC": 0.4, "TEMPORAL": 0.95, "CAUSAL": 0.2, "ENTITY": 0.5}}

JSON:"""

    def __init__(
        self,
        llm_controller,
        cache_dir: str = "./edge_score_cache",
        use_cache: bool = True,
        temperature: float = 0.0,
        fallback_to_static: bool = True,
        **kwargs  # Ignore extra args for compatibility
    ):
        """
        Initialize O(1) LLM edge scorer.

        Args:
            llm_controller: LLM controller instance
            cache_dir: Directory to cache query → weights mapping
            use_cache: Enable caching (highly recommended)
            temperature: LLM temperature (0.0 = deterministic)
            fallback_to_static: Use static scorer on LLM error
        """
        self.llm_controller = llm_controller
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.temperature = temperature
        self.fallback_scorer = StaticEdgeScorer() if fallback_to_static else None

        # Initialize cache (query_text → weights)
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "query_weights_v2.json"
            self.cache = self._load_cache()
        else:
            self.cache = {}

        # Track current query to avoid redundant calls
        self._current_query = None
        self._current_weights = None

        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_calls = 0  # Track total LLM calls

        logger.info(f"LLMEdgeScorerV2 initialized (O(1) mode) | Cache: {use_cache}")

    def __del__(self):
        """Save cache on cleanup."""
        if hasattr(self, 'use_cache') and self.use_cache:
            self._save_cache()
            logger.info(f"Cache saved: {self.llm_calls} total LLM calls, {len(self.cache)} queries cached")

    def get_query_weights(self, query_text: str) -> Dict[LinkType, float]:
        """
        Get edge type weights for a query (O(1) LLM call, cached).

        Args:
            query_text: The question being asked

        Returns:
            Dict mapping LinkType → weight (0.0-1.0)
        """
        # Check cache
        cache_key = self._create_cache_key(query_text)
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"Cache HIT for query: {query_text[:50]}...")
            return self.cache[cache_key]

        # Cache miss - call LLM
        self.cache_misses += 1
        self.llm_calls += 1
        logger.info(f"Cache MISS ({self.llm_calls}) - calling LLM for query: {query_text[:50]}...")

        try:
            # Build prompt
            prompt = self.WEIGHT_PROMPT.format(query_text=query_text)

            # Call LLM (single call per query)
            response = self.llm_controller.llm.get_completion(
                prompt=prompt,
                response_format={"type": "text"},
                temperature=self.temperature
            )

            # Parse JSON response
            weights_dict = self._parse_weights(response)

            # Convert to LinkType keys
            weights = {
                LinkType.SEMANTIC: weights_dict.get("SEMANTIC", 0.5),
                LinkType.TEMPORAL: weights_dict.get("TEMPORAL", 0.5),
                LinkType.CAUSAL: weights_dict.get("CAUSAL", 0.5),
                LinkType.ENTITY: weights_dict.get("ENTITY", 0.5)
            }

            # Validate weights
            for link_type, weight in weights.items():
                weights[link_type] = max(0.0, min(1.0, weight))  # Clamp to [0, 1]

            # Cache result
            if self.use_cache:
                self.cache[cache_key] = weights
                # Save periodically
                if self.llm_calls % 10 == 0:
                    self._save_cache()

            logger.debug(f"Weights: {weights}")
            return weights

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            if self.fallback_scorer:
                logger.info("Using fallback static weights")
                # Return default weights
                return {
                    LinkType.SEMANTIC: 1.0,
                    LinkType.TEMPORAL: 0.8,
                    LinkType.CAUSAL: 0.9,
                    LinkType.ENTITY: 0.7
                }
            else:
                raise

    def score_edge(
        self,
        source_node: EventNode,
        target_node: EventNode,
        edge_type: LinkType,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict] = None
    ) -> float:
        """
        Score an edge using query-level weights (FAST - no per-edge LLM calls).

        Formula: score = cosine_similarity(target, query) × query_weights[edge_type]

        Args:
            source_node: Source node
            target_node: Target node
            edge_type: Type of edge (SEMANTIC, TEMPORAL, etc.)
            query_text: The question being asked
            query_embedding: Query embedding vector
            context: Optional context (ignored)

        Returns:
            Score in [0.0, 1.0]
        """
        # Get query weights (cached after first call for this query)
        if self._current_query != query_text:
            self._current_query = query_text
            self._current_weights = self.get_query_weights(query_text)

        # Get edge type weight from query
        edge_weight = self._current_weights.get(edge_type, 0.5)

        # Compute semantic similarity
        if target_node.embedding_vector is not None and query_embedding is not None:
            # Convert embedding to numpy array (handles string representations)
            target_emb = self._ensure_numpy_array(target_node.embedding_vector)
            query_emb_arr = self._ensure_numpy_array(query_embedding)

            if target_emb is not None and query_emb_arr is not None:
                similarity = self._cosine_similarity(target_emb, query_emb_arr)
            else:
                similarity = 0.0
        else:
            similarity = 0.0

        # Combine: similarity × query_weight
        score = similarity * edge_weight

        return float(max(0.0, min(1.0, score)))  # Clamp to [0, 1]

    def _ensure_numpy_array(self, vec) -> Optional[np.ndarray]:
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

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _parse_weights(self, response: str) -> Dict[str, float]:
        """Parse JSON weights from LLM response."""
        import re

        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            json_str = json_match.group(0)
            try:
                weights = json.loads(json_str)
                # Ensure all values are floats
                return {k: float(v) for k, v in weights.items()}
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse JSON match: {e}")

        # Fallback: try to parse the entire response
        try:
            weights = json.loads(response.strip())
            # Ensure all values are floats
            return {k: float(v) for k, v in weights.items()}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            logger.warning(f"Response was: {response[:200]}")
            # Return default weights
            return {"SEMANTIC": 0.5, "TEMPORAL": 0.5, "CAUSAL": 0.5, "ENTITY": 0.5}

    def _create_cache_key(self, query_text: str) -> str:
        """Create cache key from query text."""
        return hashlib.md5(query_text.encode()).hexdigest()

    def _load_cache(self) -> Dict[str, Dict]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    raw_cache = json.load(f)

                # Convert string keys back to LinkType
                cache = {}
                for query_hash, weights_dict in raw_cache.items():
                    try:
                        # Ensure weights_dict has all required keys and valid float values
                        converted_weights = {}
                        for k, v in weights_dict.items():
                            try:
                                link_type = LinkType(k)
                                weight_value = float(v)
                                converted_weights[link_type] = weight_value
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Skipping invalid weight entry {k}={v}: {e}")
                                continue

                        # Only add if we have valid weights
                        if converted_weights:
                            cache[query_hash] = converted_weights
                    except Exception as e:
                        logger.warning(f"Skipping invalid cache entry: {e}")
                        continue

                logger.info(f"Loaded {len(cache)} cached query weights")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            # Convert LinkType keys to strings for JSON
            json_cache = {}
            for query_hash, weights in self.cache.items():
                json_cache[query_hash] = {
                    k.value: v for k, v in weights.items()
                }

            cache_copy = dict(json_cache)  # Avoid iteration errors
            with open(self.cache_file, 'w') as f:
                json.dump(cache_copy, f, indent=2)

            logger.debug(f"Saved {len(cache_copy)} query weights to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_llm_calls': self.llm_calls,
            'unique_queries_cached': len(self.cache)
        }
