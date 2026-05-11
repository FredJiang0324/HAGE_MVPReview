#!/usr/bin/env python3
"""
HotpotQA Evaluation Script

Evaluates HAGE on HotpotQA by building memory per question.
Uses existing LoCoMo pipeline without modifications.

Key difference from LoCoMo:
- LoCoMo: Build once per sample → Answer 200 questions
- HotpotQA: Build once per question → Answer 1 question (10 docs)
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable sentence-transformers progress bars
import logging as logging_module
logging_module.getLogger("sentence_transformers").setLevel(logging_module.WARNING)

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_hotpotqa import load_hotpotqa_dataset
from memory.memory_builder import MemoryBuilder
from memory.query_engine import QueryEngine
from memory.test_harness import TestHarness
from memory.evaluator import Evaluator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def reformat_context_for_qwen(context: str) -> str:
    """Strip LoCoMo-style metadata and present as clean documents for Qwen 3B."""
    import re
    lines = context.split('\n')
    documents = []
    doc_num = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('information ranked') or line.startswith('KEY FACTS'):
            continue
        m = re.match(r'^\d+\.\s+\*+[^*]*\*+\s*(.*)', line)
        if m:
            rest = m.group(1)
            speaker_m = re.search(r'\[speaker:\s*([^\]]+)\]', rest, re.IGNORECASE)
            speaker = speaker_m.group(1).strip().title() if speaker_m else ''
            text = re.sub(r'\[[^\]]*\]', '', rest).strip()
            text = re.sub(r'\*+', '', text).strip()
            if text:
                doc_num += 1
                label = f"Document {doc_num}" + (f" ({speaker})" if speaker else "")
                documents.append(f"{label}: {text}")
        elif re.match(r'^\d+\.', line):
            text = re.sub(r'^\d+\.\s*', '', line)
            text = re.sub(r'\[[^\]]*\]', '', text).strip()
            text = re.sub(r'\*+', '', text).strip()
            if text:
                doc_num += 1
                documents.append(f"Document {doc_num}: {text}")
    return '\n\n'.join(documents) if documents else context


def extract_hotpotqa_subqueries(question: str):
    """
    Extract sub-queries from a HotpotQA bridge question for a second retrieval pass.

    Bridge questions have embedded relative clauses like:
      "What position did [the woman who portrayed X] hold?"
    We extract both the outer question and the embedded clause as separate queries.

    Returns list of sub-query strings (may be empty for comparison questions).
    """
    import re
    q = question.strip()
    q_lower = q.lower()

    # Comparison questions (yes/no) don't need decomposition
    if q_lower.startswith(('are ', 'were ', 'is ', 'was ', 'do ', 'does ', 'did ')):
        return []

    subqueries = []

    # Pattern 1: "who/that/which <verb>" relative clauses
    # e.g. "the woman who portrayed Corliss Archer" → "who portrayed Corliss Archer"
    rel_clause = re.search(r'(who|that|which)\s+\w+.{5,60}?\b(in|on|at|by|from|of)\b[^?]{3,40}', q, re.IGNORECASE)
    if rel_clause:
        subqueries.append(rel_clause.group(0).strip(' .,'))

    # Pattern 2: quoted titles or proper noun phrases
    # e.g. "Big Stone Gap", "Kiss and Tell"
    quoted = re.findall(r'"([^"]+)"', q)
    subqueries.extend(quoted)

    # Pattern 3: "X is the <role> of Y" → extract Y
    role_match = re.search(r'(?:is|was)\s+the\s+\w+(?:\s+\w+)?\s+of\s+(.{5,50}?)(?:\s+(?:in|on|at|by|from|that|who)|[?]|$)', q, re.IGNORECASE)
    if role_match:
        subqueries.append(role_match.group(1).strip(' .,'))

    return subqueries


def setup_logging(log_dir, model_name):
    """Setup logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hotpotqa_run_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    # Console handler (WARNING and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler (INFO and above)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info(f"{'='*70}")
    logger.info(f"Starting HotpotQA evaluation for model: {model_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*70}")

    return log_file


def evaluate_hotpotqa_question_wrapper(sample, sample_idx, args, evaluator, shared_encoder):
    """
    Wrapper function for parallel evaluation with shared evaluator and encoder.
    """
    return evaluate_hotpotqa_question(sample, sample_idx, args, evaluator, shared_encoder)


def evaluate_hotpotqa_question(
    sample,
    sample_idx,
    args,
    evaluator,
    shared_encoder=None
):
    """
    Evaluate a single HotpotQA question by building memory and querying.

    Args:
        sample: LoCoMoSample (converted from HotpotQA)
        sample_idx: Question index
        args: Arguments
        evaluator: Evaluator instance

    Returns:
        Dict with results including EM, F1, and LLM judge metrics
    """
    # Setup cache directory for this question with batch subfolder (0-99, 100-199, etc.)
    model_name_normalized = args.model.replace(".", "_").replace("-", "_")
    embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
    batch_start = (sample_idx // 100) * 100
    batch_end = batch_start + 99
    batch_folder = f"{batch_start}_{batch_end}"
    cache_dir = f"./hotpotqa_cache_{model_name_normalized}/{batch_folder}/question{sample_idx}{embedding_suffix}"

    # Build memory from 10 documents
    logger.info(f"Building memory for question {sample_idx}...")

    builder = MemoryBuilder(
        cache_dir=cache_dir,
        llm_model=args.model,
        use_episodes=args.use_episodes,
        embedding_model=args.embedding_model,
        llm_backend=args.backend
    )

    # Replace encoder with shared one for thread safety
    if shared_encoder is not None:
        builder.trg.encoder = shared_encoder

    # Build memory (will be fresh each time unless cache exists)
    cache_file = Path(cache_dir) / "graph.json"
    if cache_file.exists() and not args.rebuild:
        logger.info("Loading cached memory...")
        builder.load()
        build_stats = {'event_count': 0, 'from_cache': True}
    else:
        logger.info("Building memory...")
        build_stats = builder.build_memory(sample)
        builder.save()

    logger.info(f"Memory built: {build_stats.get('event_count', 0)} events")

    # Setup query engine
    query_engine = QueryEngine(
        builder.trg,
        builder.node_index,
        entity_session_map=builder.entity_session_map if hasattr(builder, 'entity_session_map') else None,
        entity_dia_map=builder.entity_dia_map if hasattr(builder, 'entity_dia_map') else None
    )

    # Setup test harness
    use_simple = args.backend == "ollama"
    tester = TestHarness(builder, query_engine, evaluator=evaluator, use_simple_prompts=use_simple)

    # Setup best-of-N if requested
    if args.best_of_n > 1:
        from memory.best_of_n_selector import BestOfNSelector
        tester.best_of_n = args.best_of_n
        tester.best_of_n_method = args.best_of_n_method
        tester.best_of_n_selector = BestOfNSelector(
            n_attempts=args.best_of_n,
            selection_method=args.best_of_n_method
        )

    # Query the single question
    logger.info(f"Querying question {sample_idx}...")

    qa = sample.qa[0]

    query_context, answer_context = query_engine.query(qa.question, top_k=15)

    # Reformat context into clean plain-text (strips LoCoMo metadata for all backends)
    answer_context = reformat_context_for_qwen(answer_context)

    # Generate answer
    predicted = tester.answer_question(
        qa.question,
        answer_context,
        category=qa.category,
        expected=str(qa.final_answer)
    )

    # Evaluate answer
    is_correct = False
    metrics = None
    llm_score = 0.0

    if evaluator and qa.final_answer:
        eval_result = evaluator.evaluate_answer(
            qa.question,
            str(qa.final_answer),
            predicted,
            question_category=qa.category
        )
        metrics = eval_result.get('metrics')
        is_correct = eval_result.get('is_correct', False)
        llm_score = eval_result.get('llm_judge_score', 0.0)

    # Prepare result
    result = {
        'question_id': sample_idx,
        'sample_id': sample.sample_id,
        'question': qa.question,
        'gold_answer': qa.final_answer,
        'predicted_answer': predicted,
        'correct': is_correct,
        'metrics': metrics,
        'llm_judge_score': llm_score,
    }

    logger.info(f"Question {sample_idx}: EM={is_correct}, F1={metrics.get('f1', 0):.2f}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HAGE on HotpotQA")
    parser.add_argument("--dataset", default="data/hotpot_dev_distractor_v1.json",
                       help="Path to HotpotQA dataset")
    parser.add_argument("--num-questions", type=int, default=100,
                       help="Number of questions to evaluate (default: 100)")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="Starting question index (default: 0)")
    parser.add_argument("--model", type=str, default=None,
                       help="LLM model to use (e.g., gpt-4o-mini)")
    parser.add_argument("--embedding-model", type=str, default="minilm",
                       choices=["minilm", "openai"],
                       help="Embedding model: 'minilm' or 'openai'")
    parser.add_argument("--use-episodes", action="store_true",
                       help="Use episode-based segmentation")
    parser.add_argument("--backend", type=str, default="openai",
                       choices=["openai", "ollama"],
                       help="LLM backend")
    parser.add_argument("--use-azure", action="store_true", default=True,
                       help="Use Azure OpenAI (default: True)")
    parser.add_argument("--use-openai", action="store_true",
                       help="Use regular OpenAI instead of Azure")
    parser.add_argument("--best-of-n", type=int, default=4,
                       help="Best-of-N answer selection (default: 4)")
    parser.add_argument("--best-of-n-method", type=str, default="llm_judge",
                       choices=["llm_judge", "voting", "f1"],
                       help="Best-of-N selection method")
    parser.add_argument("--rebuild", action="store_true",
                       help="Force rebuild memory (ignore cache)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of questions to process in parallel (default: 1)")
    parser.add_argument("--judge-model", type=str, default=None,
                       help="Separate model for LLM judge (e.g., gpt-4o-mini). Defaults to --model.")
    parser.add_argument("--judge-backend", type=str, default=None,
                       choices=["openai", "ollama"],
                       help="Backend for judge model. Defaults to --backend.")
    args = parser.parse_args()

    # Configure API settings
    if args.backend == "ollama":
        use_azure = False
        if not args.model:
            args.model = "qwen2.5:3b"
    else:
        if args.use_openai:
            use_azure = False
        else:
            use_azure = True
            env_use_azure = os.getenv("USE_AZURE_OPENAI", "").lower()
            if env_use_azure == "false":
                use_azure = False

        if use_azure:
            os.environ['USE_AZURE_OPENAI'] = 'true'
            if not args.model:
                args.model = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini')
            os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = args.model
        else:
            os.environ['USE_AZURE_OPENAI'] = 'false'
            if not args.model:
                args.model = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')

    print("="*70)
    print("  HAGE HotpotQA Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Questions: {args.num_questions} (starting from {args.start_idx})")
    print(f"  Embedding: {args.embedding_model}")
    print(f"  Best-of-N: {args.best_of_n}")
    print(f"  Parallel: {args.parallel} workers")
    print("="*70)

    # Setup logging
    model_name_normalized = args.model.replace(".", "_").replace("-", "_")
    log_dir = f"logs_hotpotqa_{model_name_normalized}"
    log_file = setup_logging(log_dir, args.model)
    print(f"Logging to: {log_file}")
    print("="*70)

    # Load HotpotQA samples
    print(f"\nLoading HotpotQA samples...")
    samples = load_hotpotqa_dataset(
        args.dataset,
        num_samples=args.num_questions,
        start_idx=args.start_idx
    )

    # Pre-initialize shared encoder for thread safety
    print(f"\nInitializing shared encoder...")
    from memory.vector_db import VectorEncoder
    if args.embedding_model == "openai":
        shared_encoder = VectorEncoder(model_name='text-embedding-3-small', use_openai=True)
    else:
        shared_encoder = VectorEncoder(model_name='all-MiniLM-L6-v2', use_openai=False)
    print(f"Encoder initialized: {shared_encoder.dimension} dimensions")

    # Create a temporary builder to initialize shared llm_controller for evaluator
    print(f"\nInitializing evaluator...")
    judge_model = args.judge_model or args.model
    judge_backend = args.judge_backend or args.backend
    if judge_backend == "openai" and args.backend == "ollama":
        # Force Azure/OpenAI env for judge even when answerer uses ollama
        os.environ['USE_AZURE_OPENAI'] = 'true'
        os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = judge_model
    temp_cache = f"./temp_init_{judge_model.replace('.', '_').replace('-', '_')}_judge"
    temp_builder = MemoryBuilder(
        cache_dir=temp_cache,
        llm_model=judge_model,
        use_episodes=False,
        embedding_model=args.embedding_model,
        llm_backend=judge_backend
    )
    # Replace temp builder's encoder with shared one
    temp_builder.trg.encoder = shared_encoder
    evaluator = Evaluator(llm_controller=temp_builder.llm_controller, use_llm_judge=True)
    # Clean up temp directory
    import shutil
    if Path(temp_cache).exists():
        shutil.rmtree(temp_cache)

    # Evaluate each question
    results = []

    if args.parallel > 1:
        # Parallel processing using threads (I/O-bound task)
        print(f"\n🚀 Running {args.parallel} questions in parallel...\n")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx, sample in enumerate(samples):
                question_idx = args.start_idx + idx
                future = executor.submit(evaluate_hotpotqa_question_wrapper, sample, question_idx, args, evaluator, shared_encoder)
                future_to_idx[future] = (idx, question_idx)

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx, question_idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed = len(results)
                    print(f"✓ Completed Question {question_idx + 1} ({completed}/{args.num_questions})")
                except Exception as exc:
                    print(f"✗ Question {question_idx + 1} failed: {exc}")
                    logger.error(f"Question {question_idx + 1} failed: {exc}")

        # Sort results by question_id to maintain order
        results.sort(key=lambda x: x['question_id'])
    else:
        # Sequential processing
        for idx, sample in enumerate(samples):
            print(f"\n{'='*70}")
            print(f"Evaluating Question {args.start_idx + idx + 1}/{args.start_idx + args.num_questions}")
            print(f"{'='*70}")

            result = evaluate_hotpotqa_question(
                sample,
                args.start_idx + idx,
                args,
                evaluator,
                shared_encoder
            )
            results.append(result)

    # Calculate aggregate metrics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100 if total > 0 else 0

    f1_scores = [r['metrics']['f1'] for r in results if r.get('metrics')]
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

    bleu_scores = [r['metrics']['bleu1'] for r in results if r.get('metrics')]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) * 100 if bleu_scores else 0

    llm_scores = [r.get('llm_judge_score', 0) for r in results if 'llm_judge_score' in r]
    avg_llm = sum(llm_scores) / len(llm_scores) * 100 if llm_scores else 0

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"HOTPOTQA RESULTS ({total} questions)")
    print(f"{'='*70}")
    print(f"\nAccuracy Metrics:")
    print(f"  Exact Match: {accuracy:.1f}%")
    print(f"  F1 Score: {avg_f1:.1f}%")
    print(f"  BLEU-1: {avg_bleu:.1f}%")
    print(f"  LLM Judge: {avg_llm:.1f}%")

    # Save results
    results_dir = f"results_hotpotqa_{model_name_normalized}"
    os.makedirs(results_dir, exist_ok=True)
    output_file = f"{results_dir}/results_{args.start_idx}_{args.start_idx + args.num_questions - 1}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'embedding_model': args.embedding_model,
            'num_questions': total,
            'start_idx': args.start_idx,
            'best_of_n': args.best_of_n,
            'aggregate': {
                'exact_match': accuracy,
                'f1': avg_f1,
                'bleu1': avg_bleu,
                'llm_judge': avg_llm,
            },
            'results': results
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    logger.info(f"Evaluation completed successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
