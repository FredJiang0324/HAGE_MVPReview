"""
HotpotQA Dataset Loader

Loads HotpotQA dataset and converts to LoCoMo-compatible format.
This allows using existing HAGE memory building pipeline without modifications.
"""

import json
from typing import List, Dict, Union
from pathlib import Path
from datetime import datetime

# Import LoCoMo data structures (reuse existing)
from load_dataset import QA, Turn, Session, Conversation, EventSummary, Observation, LoCoMoSample


def load_hotpotqa_dataset(
    file_path: Union[str, Path],
    num_samples: int = None,
    start_idx: int = 0
) -> List[LoCoMoSample]:
    """
    Load HotpotQA dataset and convert to LoCoMo-compatible format.

    Each HotpotQA question becomes one "sample" where:
    - 10 documents → 10 "conversation turns" (treating each doc as a speaker turn)
    - 1 question → 1 QA pair

    Args:
        file_path: Path to hotpot_dev_distractor_v1.json
        num_samples: Number of samples to load (None = all)
        start_idx: Starting index for sampling

    Returns:
        List of LoCoMoSample objects (one per HotpotQA question)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Select subset if requested
    if num_samples is not None:
        end_idx = min(start_idx + num_samples, len(data))
        data = data[start_idx:end_idx]
        print(f"Loading HotpotQA samples {start_idx} to {end_idx-1} ({len(data)} samples)")
    else:
        print(f"Loading all {len(data)} HotpotQA samples")

    samples = []

    for idx, item in enumerate(data):
        sample = convert_hotpotqa_to_locomo(item, sample_idx=start_idx + idx)
        samples.append(sample)

    print(f"Converted {len(samples)} HotpotQA samples to LoCoMo format")
    return samples


def convert_hotpotqa_to_locomo(item: Dict, sample_idx: int) -> LoCoMoSample:
    """
    Convert a single HotpotQA item to LoCoMo format.

    Strategy:
    - Each document becomes a "turn" in the conversation
    - Document title is the "speaker"
    - Document sentences are concatenated as turn "text"
    - Single session containing all documents

    Args:
        item: HotpotQA sample dict with keys:
            - _id: question ID
            - question: question text
            - answer: answer text
            - context: list of [title, sentences] pairs (10 documents)
            - supporting_facts: list of [title, sent_idx] showing gold evidence
            - type: "bridge" or "comparison"
            - level: "easy", "medium", or "hard"

    Returns:
        LoCoMoSample object
    """
    sample_id = f"hotpotqa_{sample_idx}"

    # Convert documents to conversation turns
    turns = []
    for doc_idx, (doc_title, doc_sentences) in enumerate(item['context']):
        # Concatenate all sentences in the document
        doc_text = ' '.join(doc_sentences)

        # Create a turn with document as content
        turn = Turn(
            speaker=doc_title,  # Document title as "speaker"
            dia_id=f"doc_{doc_idx}",
            text=doc_text
        )
        turns.append(turn)

    # Create single session with all documents
    session = Session(
        session_id=1,
        date_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        turns=turns
    )

    # Create conversation with all documents
    conversation = Conversation(
        speaker_a="Document",
        speaker_b="Document",
        sessions={1: session}
    )

    # Create QA pair
    qa_pair = QA(
        question=item['question'],
        answer=item['answer'],
        evidence=item.get('supporting_facts', []),
        category=6,  # HotPotQA category - uses specialized multi-hop prompt with comparison/bridge detection
        adversarial_answer=None
    )

    # Create LoCoMo sample
    sample = LoCoMoSample(
        sample_id=sample_id,
        qa=[qa_pair],  # One question per sample
        conversation=conversation,
        event_summary=EventSummary(events={}),
        observation=Observation(observations={}),
        session_summary={}
    )

    return sample


def get_hotpotqa_stats(file_path: Union[str, Path]) -> Dict:
    """
    Get statistics about HotpotQA dataset.

    Returns:
        Dict with dataset statistics
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    stats = {
        'total_samples': len(data),
        'question_types': {},
        'difficulty_levels': {},
        'avg_context_docs': 0,
        'avg_supporting_facts': 0,
    }

    for item in data:
        # Count question types
        q_type = item.get('type', 'unknown')
        stats['question_types'][q_type] = stats['question_types'].get(q_type, 0) + 1

        # Count difficulty levels
        level = item.get('level', 'unknown')
        stats['difficulty_levels'][level] = stats['difficulty_levels'].get(level, 0) + 1

        # Count docs and supporting facts
        stats['avg_context_docs'] += len(item['context'])
        stats['avg_supporting_facts'] += len(item.get('supporting_facts', []))

    # Calculate averages
    n = len(data)
    stats['avg_context_docs'] /= n
    stats['avg_supporting_facts'] /= n

    return stats


if __name__ == '__main__':
    # Test loader
    import sys

    file_path = 'data/hotpot_dev_distractor_v1.json'

    print("="*70)
    print("HotpotQA Dataset Statistics")
    print("="*70)
    stats = get_hotpotqa_stats(file_path)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Avg documents per question: {stats['avg_context_docs']:.1f}")
    print(f"Avg supporting facts: {stats['avg_supporting_facts']:.1f}")
    print(f"\nQuestion types: {stats['question_types']}")
    print(f"Difficulty levels: {stats['difficulty_levels']}")

    print("\n" + "="*70)
    print("Loading first 5 samples...")
    print("="*70)
    samples = load_hotpotqa_dataset(file_path, num_samples=5)

    print(f"\nSample 0 structure:")
    sample = samples[0]
    print(f"  ID: {sample.sample_id}")
    print(f"  Question: {sample.qa[0].question}")
    print(f"  Answer: {sample.qa[0].answer}")
    print(f"  Documents: {len(sample.conversation.sessions[1].turns)}")
    print(f"  First doc title: {sample.conversation.sessions[1].turns[0].speaker}")
    print(f"  First doc preview: {sample.conversation.sessions[1].turns[0].text[:100]}...")
