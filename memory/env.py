"""
Multi-Graph RL Environment for Graph Traversal (Phase 2)

High-speed sandbox that:
  - Swaps between graphs (multi-graph support)
  - Calculates rewards via target node ID matching
  - Runs entirely on tensors — no LLM calls

Default Reward Function (Sparse & Strict):
  +10.0  Reaching the Target_Node_ID (Success)
  -0.05  Step penalty (Efficiency)
  -1.0   Reaching max hops without success

Configurable via RewardConfig for ablation studies.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .pytorch_graph import PyTorchGraph

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configurable reward parameters for ablation studies."""
    hit_reward: float = 10.0       # Reward for reaching target node
    step_penalty: float = -0.05    # Per-step cost
    timeout_penalty: float = -1.0  # Penalty for exceeding max_hops
    dead_end_penalty: float = -1.0 # Penalty for dead end
    normalize_hit: bool = False    # If True, hit_reward is set to 1.0

    def __post_init__(self):
        if self.normalize_hit:
            self.hit_reward = 1.0

    @staticmethod
    def default():
        """Full reward: +10 hit, -0.05 step, -1.0 timeout."""
        return RewardConfig()

    @staticmethod
    def hit_only():
        """Evidence hit only: no step penalty, no timeout penalty."""
        return RewardConfig(hit_reward=10.0, step_penalty=0.0, timeout_penalty=0.0)

    @staticmethod
    def hit_minus_step():
        """Evidence hit minus step penalty, no timeout penalty."""
        return RewardConfig(hit_reward=10.0, step_penalty=-0.05, timeout_penalty=0.0)

    @staticmethod
    def normalized_hit_minus_step():
        """Normalized hit (1.0) minus step penalty."""
        return RewardConfig(normalize_hit=True, step_penalty=-0.05, timeout_penalty=-1.0)

    @staticmethod
    def with_step_lambda(lambda_step: float):
        """Full reward with configurable step penalty weight."""
        return RewardConfig(hit_reward=10.0, step_penalty=-lambda_step, timeout_penalty=-1.0)


@dataclass
class StepResult:
    """Result of a single environment step."""
    next_node: int
    reward: float
    done: bool
    success: bool
    steps_taken: int


class GraphTraversalEnv:
    """
    RL environment for learning graph routing.

    Each episode:
      1. reset() with a query item → places agent at the start node
      2. step(action_idx) → moves to a neighbour, returns reward
      3. Episode ends on success (target reached) or max_hops exhausted

    The start node is the node most similar to the query embedding
    (cosine similarity).
    """

    def __init__(self, max_hops: int = 5, reward_config: RewardConfig = None):
        self.max_hops = max_hops
        self.rc = reward_config or RewardConfig.default()

        # Episode state
        self.graph: Optional[PyTorchGraph] = None
        self.query_emb: Optional[torch.Tensor] = None
        self.target_nodes: List[int] = []
        self.target_set: set = set()
        self.targets_found: set = set()
        self.current_node: int = -1
        self.steps: int = 0
        self.done: bool = True
        self.visited: set = set()

    def reset(
        self,
        graph: PyTorchGraph,
        query_emb: torch.Tensor,
        target_node_indices: List[int],
        start_node: Optional[int] = None,
    ) -> int:
        """
        Start a new episode.

        Args:
            graph: The graph to traverse.
            query_emb: (384,) query embedding tensor.
            target_node_indices: List of acceptable target node indices.
            start_node: Optional override for start position.
                        If None, picks the node closest to query_emb.

        Returns:
            current_node index.
        """
        self.graph = graph
        self.query_emb = query_emb.to(graph.device)
        self.target_nodes = target_node_indices
        self.target_set = set(target_node_indices)
        self.targets_found = set()
        self.steps = 0
        self.done = False
        self.visited = set()

        if start_node is not None:
            self.current_node = start_node
        else:
            self.current_node = self._find_start_node()

        self.visited.add(self.current_node)

        # Spawn-on-target: count it as found. Terminate only if it was the only target.
        if self.current_node in self.target_set:
            self.targets_found.add(self.current_node)
            if self.targets_found == self.target_set:
                self.done = True

        return self.current_node

    def step(self, action_idx: int) -> StepResult:
        """
        Take one step: move to the action_idx-th neighbour of current_node.

        Args:
            action_idx: Index into the neighbour list of current_node.

        Returns:
            StepResult with reward, done flag, etc.
        """
        assert not self.done, "Episode already finished — call reset()"

        neighbors = self.graph.neighbors(self.current_node)

        # Dead end — episode over with failure penalty
        if not neighbors or action_idx < 0 or action_idx >= len(neighbors):
            self.done = True
            return StepResult(
                next_node=self.current_node,
                reward=self.rc.dead_end_penalty,
                done=True,
                success=False,
                steps_taken=self.steps,
            )

        next_node, _ = neighbors[action_idx]
        self.steps += 1
        self.current_node = next_node
        self.visited.add(next_node)

        # Hit a new (unique) target. Already-found targets give no extra credit.
        new_target = (
            next_node in self.target_set
            and next_node not in self.targets_found
        )
        if new_target:
            self.targets_found.add(next_node)

        all_found = self.targets_found == self.target_set
        timed_out = self.steps >= self.max_hops

        # Reward = sum of components per Eq. 11: r_hit if a new target was reached
        # this step, plus the timeout penalty if the hop budget is now exhausted.
        # Step penalty applies only when no progress was made.
        if new_target:
            reward = self.rc.hit_reward
            if not all_found and timed_out:
                reward += self.rc.timeout_penalty
        elif timed_out:
            reward = self.rc.timeout_penalty
        else:
            reward = self.rc.step_penalty

        done = all_found or timed_out
        if done:
            self.done = True
        return StepResult(
            next_node=next_node,
            reward=reward,
            done=done,
            success=all_found,
            steps_taken=self.steps,
        )

    def _find_start_node(self) -> int:
        """Pick the node with highest cosine similarity to query_emb."""
        with torch.no_grad():
            sims = torch.nn.functional.cosine_similarity(
                self.query_emb.unsqueeze(0),   # (1, 384)
                self.graph.node_embeddings,     # (N, 384)
                dim=-1,
            )
            return sims.argmax().item()

    @property
    def num_neighbors(self) -> int:
        """Number of neighbours at the current node."""
        if self.graph is None:
            return 0
        return len(self.graph.neighbors(self.current_node))


def run_episode(
    model,
    env: GraphTraversalEnv,
    graph: PyTorchGraph,
    query_emb: torch.Tensor,
    target_node_indices: List[int],
    greedy: bool = False,
) -> Tuple[List[float], List[torch.Tensor], bool, int]:
    """
    Run a full episode and collect trajectory.

    Returns:
        rewards: list of per-step rewards
        log_probs: list of per-step log-probability tensors
        success: whether the target was reached
        steps: total steps taken
    """
    env.reset(graph, query_emb, target_node_indices)

    rewards: List[float] = []
    log_probs: List[torch.Tensor] = []

    # Edge case: spawned on a target and it was the only one needed.
    if env.done:
        return rewards, log_probs, env.targets_found == env.target_set, 0

    while not env.done:
        action_idx, log_prob, probs = model.get_action(
            graph, env.current_node, query_emb, greedy=greedy
        )

        if probs.numel() == 0:
            # Dead end: no neighbours to expand from current node.
            env.done = True
            rewards.append(env.rc.dead_end_penalty)
            log_probs.append(torch.tensor(0.0, device=query_emb.device))
            break

        result = env.step(action_idx)
        rewards.append(result.reward)
        log_probs.append(log_prob)

    success = env.targets_found == env.target_set
    return rewards, log_probs, success, env.steps
