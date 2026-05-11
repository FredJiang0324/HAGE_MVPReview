"""
LLM Edge Scoring Prompts

This module contains prompt templates for LLM-based edge importance scoring.
Multiple prompt versions are provided for experimentation and optimization.

Prompt Design Principles:
- Clear task description
- Structured input format
- Explicit rating scale
- Consideration factors (semantic, temporal, causal, entity)
- Concise output format (single float)

Usage:
    from memory.llm_edge_prompts import EDGE_SCORING_PROMPT_V1
    prompt = EDGE_SCORING_PROMPT_V1.format(
        query_text="What did Alice buy?",
        source_content="Alice went to the store",
        ...
    )
"""

# Edge type descriptions for prompts
EDGE_TYPE_DESCRIPTIONS = {
    "TEMPORAL": "This edge connects events in chronological order (one happened before/after another).",
    "SEMANTIC": "This edge connects conceptually similar or related events.",
    "CAUSAL": "This edge indicates one event caused, influenced, or enabled another.",
    "ENTITY": "This edge connects events that share common entities (people, places, objects)."
}


# Version 1: Basic rating prompt (simple, direct)
EDGE_SCORING_PROMPT_V1 = """You are evaluating the importance of a graph edge for answering a question.

**Question:** {query_text}

**Source Node (Current Position):**
Content: {source_content}
Type: {source_type}
Timestamp: {source_timestamp}

**Target Node (Potential Next Step):**
Content: {target_content}
Type: {target_type}
Timestamp: {target_timestamp}

**Edge Type:** {edge_type}
{edge_type_description}

**Task:** Rate how useful it would be to traverse from Source to Target to answer the question.

**Rating Scale:**
- 1.0: Highly relevant, directly helps answer the question
- 0.7-0.9: Moderately relevant, provides useful context
- 0.4-0.6: Somewhat relevant, minor supporting information
- 0.1-0.3: Minimally relevant, tangentially related
- 0.0: Irrelevant, does not help answer the question

**Consider:**
- Does the target node contain information that answers or helps answer the question?
- Is the edge type appropriate for the question (e.g., temporal edges for "when" questions)?
- Do both nodes share entities mentioned in the question?
- Would following this edge lead you closer to the answer?

Return ONLY a single float number between 0.0 and 1.0.
"""


# Version 2: Chain-of-thought reasoning (more detailed analysis)
EDGE_SCORING_PROMPT_V2 = """You are evaluating whether to traverse a graph edge while answering a question.

**Question:** {query_text}

**Current Node (Source):**
{source_content}
[Type: {source_type}, Time: {source_timestamp}]

**Potential Next Node (Target):**
{target_content}
[Type: {target_type}, Time: {target_timestamp}]

**Edge Connection:** {edge_type}
{edge_type_description}

**Instructions:**
1. First, identify what information the question is asking for
2. Check if the target node contains or relates to this information
3. Consider if the edge type is appropriate for the question type
4. Evaluate the relevance on a 0.0-1.0 scale

**Reasoning Process:**
- Question seeks: [identify the core information needed]
- Target node provides: [what information the target offers]
- Edge relevance: [why this connection matters or doesn't]
- Final assessment: [high/medium/low relevance]

**Output Format:**
Score: [0.0-1.0]

Provide only the numeric score on the last line.
"""


# Version 3: Few-shot examples (learn from demonstrations)
EDGE_SCORING_PROMPT_V3 = """Evaluate edge relevance for answering questions. Here are examples:

**Example 1:**
Question: "What did Alice buy at the store?"
Source: "Alice went to the grocery store"
Target: "She purchased milk, eggs, and bread"
Edge Type: TEMPORAL
Analysis: Target directly answers the question (what Alice bought). Temporal edge is appropriate (sequence of events).
Score: 0.95

**Example 2:**
Question: "What did Alice buy at the store?"
Source: "Alice went to the grocery store"
Target: "Bob played tennis yesterday"
Edge Type: TEMPORAL
Analysis: Target is about a different person doing an unrelated activity. Not relevant to the question.
Score: 0.05

**Example 3:**
Question: "Why did Alice go to the store?"
Source: "Alice needed ingredients for dinner"
Target: "She went to the grocery store"
Edge Type: CAUSAL
Analysis: Source explains the reason (causal relationship). Target confirms the action. Highly relevant.
Score: 0.90

**Example 4:**
Question: "When did the party happen?"
Source: "Alice planned a party"
Target: "The celebration was on July 15th"
Edge Type: SEMANTIC
Analysis: Target provides the specific date answer. Semantic connection through "party" and "celebration".
Score: 0.92

**Now evaluate this edge:**
Question: {query_text}
Source: {source_content} [Type: {source_type}]
Target: {target_content} [Type: {target_type}]
Edge Type: {edge_type}

Return only the numeric score (0.0-1.0):
"""


# Version 4: Binary classification (simplified decision)
EDGE_SCORING_PROMPT_V4 = """Is this edge useful for answering the question?

Question: {query_text}

From: {source_content}
To: {target_content}

Connection: {edge_type} ({edge_type_description})

Answer with a score:
- 1.0 if the target node helps answer the question
- 0.5 if the target provides context but doesn't directly answer
- 0.0 if the target is irrelevant

Score:
"""


# Version 5: Query-type aware (adaptive to question type)
EDGE_SCORING_PROMPT_V5 = """Score edge relevance for this {query_type} question.

**Question:** {query_text}
**Question Type:** {query_type} (optimized for {query_type_hint})

**Current Node:**
{source_content}

**Next Node:**
{target_content}

**Edge:** {edge_type} - {edge_type_description}

**{query_type} Scoring Criteria:**
{scoring_criteria}

Rate 0.0 (irrelevant) to 1.0 (highly relevant):
"""

# Query-type specific hints
QUERY_TYPE_HINTS = {
    "WHAT": "factual information, entities, objects, actions",
    "WHY": "causal relationships, reasons, explanations",
    "WHEN": "temporal information, dates, sequences",
    "WHERE": "locations, places, spatial relationships",
    "WHO": "people, entities, actors",
    "HOW": "processes, methods, mechanisms"
}

QUERY_TYPE_CRITERIA = {
    "WHY": """
- Prioritize CAUSAL edges (cause-effect relationships)
- Look for explanations and reasons
- Check if target explains source or vice versa
    """,
    "WHEN": """
- Prioritize TEMPORAL edges (time-based relationships)
- Look for dates, times, sequences
- Check chronological ordering
    """,
    "WHAT": """
- Prioritize SEMANTIC edges (concept similarity)
- Look for direct factual information
- Check if target contains the specific information asked
    """,
    "WHO": """
- Prioritize ENTITY edges (shared entities)
- Look for people, actors, agents
- Check if entities match question entities
    """
}


# Version 6: Comparative scoring (rank multiple edges)
EDGE_SCORING_PROMPT_V6 = """You are comparing multiple potential paths. Score this edge:

Question: {query_text}

Edge Option:
From: {source_content}
To: {target_content}
Type: {edge_type}

How valuable is this edge for answering the question?
- Consider: relevance, directness, information gain
- Compare to: staying at source or exploring other edges

Score (0.0-1.0):
"""


# Prompt template mapping
PROMPT_TEMPLATES = {
    "v1": EDGE_SCORING_PROMPT_V1,
    "v2": EDGE_SCORING_PROMPT_V2,
    "v3": EDGE_SCORING_PROMPT_V3,
    "v4": EDGE_SCORING_PROMPT_V4,
    "v5": EDGE_SCORING_PROMPT_V5,
    "v6": EDGE_SCORING_PROMPT_V6,
}

# Default prompt (can be changed based on evaluation results)
DEFAULT_PROMPT = EDGE_SCORING_PROMPT_V1


def get_prompt_template(version: str = "v1") -> str:
    """
    Get prompt template by version.

    Args:
        version: Version identifier ("v1" through "v6")

    Returns:
        Prompt template string

    Example:
        >>> from memory.llm_edge_prompts import get_prompt_template
        >>> prompt = get_prompt_template("v3")  # Few-shot version
    """
    return PROMPT_TEMPLATES.get(version, DEFAULT_PROMPT)


def build_edge_prompt(
    version: str,
    query_text: str,
    source_content: str,
    source_type: str,
    source_timestamp: str,
    target_content: str,
    target_type: str,
    target_timestamp: str,
    edge_type: str,
    query_type: str = None
) -> str:
    """
    Build a complete edge scoring prompt.

    Args:
        version: Prompt version to use
        query_text: The question being asked
        source_content: Content of source node
        source_type: Type of source node
        source_timestamp: Timestamp of source node
        target_content: Content of target node
        target_type: Type of target node
        target_timestamp: Timestamp of target node
        edge_type: Type of edge connection
        query_type: Optional query type for v5 (WHAT, WHY, WHEN, etc.)

    Returns:
        Formatted prompt string

    Example:
        >>> prompt = build_edge_prompt(
        ...     version="v1",
        ...     query_text="What did Alice buy?",
        ...     source_content="Alice went shopping",
        ...     source_type="EVENT",
        ...     source_timestamp="2024-01-01",
        ...     target_content="She bought milk",
        ...     target_type="EVENT",
        ...     target_timestamp="2024-01-01",
        ...     edge_type="TEMPORAL"
        ... )
    """
    template = get_prompt_template(version)

    # Base formatting arguments
    format_args = {
        "query_text": query_text,
        "source_content": source_content[:500],  # Truncate if too long
        "source_type": source_type,
        "source_timestamp": source_timestamp,
        "target_content": target_content[:500],
        "target_type": target_type,
        "target_timestamp": target_timestamp,
        "edge_type": edge_type,
        "edge_type_description": EDGE_TYPE_DESCRIPTIONS.get(edge_type, "")
    }

    # Add query-type specific arguments for v5
    if version == "v5" and query_type:
        format_args.update({
            "query_type": query_type,
            "query_type_hint": QUERY_TYPE_HINTS.get(query_type, "information"),
            "scoring_criteria": QUERY_TYPE_CRITERIA.get(query_type, "Standard relevance criteria")
        })

    return template.format(**format_args)


def detect_query_type(query_text: str) -> str:
    """
    Detect question type from query text.

    Args:
        query_text: The question

    Returns:
        Query type: "WHAT", "WHY", "WHEN", "WHERE", "WHO", "HOW"

    Example:
        >>> detect_query_type("Why did Alice go to the store?")
        'WHY'
        >>> detect_query_type("What did Bob buy?")
        'WHAT'
    """
    query_lower = query_text.lower()

    if query_lower.startswith("why "):
        return "WHY"
    elif query_lower.startswith("when "):
        return "WHEN"
    elif query_lower.startswith("where "):
        return "WHERE"
    elif query_lower.startswith("who "):
        return "WHO"
    elif query_lower.startswith("how "):
        return "HOW"
    else:
        return "WHAT"  # Default for "what" and other questions


# Prompt evaluation helper
def compare_prompts(prompts_to_test=None):
    """
    Helper to compare different prompt versions.

    This is a placeholder for a more complete evaluation framework.
    In practice, you would:
    1. Run each prompt on a validation set
    2. Compare accuracy, cost, latency
    3. Select the best performing version

    Args:
        prompts_to_test: List of prompt versions to compare (default: all)

    Returns:
        Dictionary with comparison results
    """
    if prompts_to_test is None:
        prompts_to_test = list(PROMPT_TEMPLATES.keys())

    print("Prompt Comparison Framework")
    print("=" * 60)
    print("\nPrompts to evaluate:")
    for version in prompts_to_test:
        template = PROMPT_TEMPLATES[version]
        token_est = len(template.split()) * 1.3  # Rough token estimate
        print(f"  {version}: ~{token_est:.0f} tokens")

    return {
        "prompts": prompts_to_test,
    }
