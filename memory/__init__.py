"""
TRG Memory System Module

This module provides the Temporal Resonance Graph Memory implementation.
"""

from .graph_db import (
    GraphDBInterface,
    NetworkXGraphDB,
    EventNode,
    Link,
    NodeType,
    LinkType,
    LinkSubType,
    LinkStatus,
    TraversalConstraints
)

from .vector_db import (
    VectorDBInterface,
    FAISSVectorDB,
    NumpyVectorDB,
    VectorEncoder,
    VectorEntry,
    IndexType,
    create_vector_db
)

from .trg_memory import (
    TemporalResonanceGraphMemory,
    EventExtractionResult,
    QueryContext
)

from .episode_segmenter import (
    EpisodeSegmenter,
    Episode,
    MessageBuffer,
    BoundaryDetector
)

from .temporal_parser import TemporalParser
from .answer_formatter import AnswerFormatter
from .llm_judge import LLMJudge
from .memory_builder import MemoryBuilder
from .query_engine import QueryEngine
from .test_harness import TestHarness
from .evaluator import Evaluator

__all__ = [
    'GraphDBInterface',
    'NetworkXGraphDB',
    'EventNode',
    'Link',
    'NodeType',
    'LinkType',
    'LinkSubType',
    'LinkStatus',
    'TraversalConstraints',

    'VectorDBInterface',
    'FAISSVectorDB',
    'NumpyVectorDB',
    'VectorEncoder',
    'VectorEntry',
    'IndexType',
    'create_vector_db',

    'TemporalResonanceGraphMemory',
    'EventExtractionResult',
    'QueryContext',

    'EpisodeSegmenter',
    'Episode',
    'MessageBuffer',
    'BoundaryDetector',

    'TemporalParser',
    'AnswerFormatter',
    'LLMJudge',

    'MemoryBuilder',
    'QueryEngine',
    'TestHarness',
    'Evaluator'
]

__version__ = '1.0.0'