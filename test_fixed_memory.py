#!/usr/bin/env python3
"""
Fixed TRG Memory System - Refactored with modular architecture

This script now uses the following modules:
- memory.memory_builder: Memory construction and indexing
- memory.query_engine: Query execution and retrieval
- memory.test_harness: Testing and evaluation

The core logic has been moved to proper modules under memory/ folder.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings

# Disable sentence-transformers progress bars
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_dataset import load_locomo_dataset
from memory.memory_builder import MemoryBuilder
from memory.query_engine import QueryEngine
from memory.test_harness import TestHarness
from memory.evaluator import Evaluator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def setup_logging(log_dir, model_name):
    """Setup logging to both console and file."""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (WARNING and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler (INFO and above - more detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info(f"{'='*70}")
    logger.info(f"Starting test run for model: {model_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*70}")

    return log_file

def score_only_mode(args):
    """
    Re-score existing results without rebuilding memory or querying.

    This mode loads existing result files and re-evaluates them with
    potentially different scoring models or methods.
    """
    import os
    from utils.memory_layer import LLMController
    from tqdm import tqdm

    print("\n" + "="*70)
    print("  SCORE-ONLY MODE: Re-evaluating existing results")
    print("="*70)

    # Check for API keys (Azure or OpenAI)
    # Command line flag takes precedence over environment variable
    if args.use_openai:
        use_azure = False
    else:
        # Default to Azure (--use-azure is True by default)
        use_azure = True
        # But check environment variable if set
        env_use_azure = os.getenv("USE_AZURE_OPENAI", "").lower()
        if env_use_azure == "false":
            use_azure = False

    if use_azure:
        # Set environment variable for openai_client module
        os.environ['USE_AZURE_OPENAI'] = 'true'
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if not api_key:
            print("Error: AZURE_OPENAI_API_KEY not found. Cannot perform LLM-based evaluation.")
            print("Please set AZURE_OPENAI_API_KEY in your .env file")
            print("Or use --use-openai to use regular OpenAI API instead")
            return 1
        # For Azure, use deployment name if not specified
        if not args.model:
            args.model = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4.1-mini')
        # Set deployment name for openai_client module
        os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = args.model
        print(f"Using Azure OpenAI - Deployment: {args.model}")
    else:
        # Set environment variable for openai_client module
        os.environ['USE_AZURE_OPENAI'] = 'false'
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY not found. Cannot perform LLM-based evaluation.")
            print("Please set OPENAI_API_KEY in your .env file")
            return 1
        # For OpenAI, use default model if not specified
        if not args.model:
            args.model = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')
        print(f"Using OpenAI API - Model: {args.model}")

    llm_controller = LLMController(
        backend='openai',  # This handles both Azure and regular OpenAI
        model=args.model,
        api_key=api_key
    )

    evaluator = Evaluator(llm_controller=llm_controller, use_llm_judge=True)
    print(f"Evaluator initialized with model: {args.model}\n")

    all_sample_results = []

    for sample_id in args.sample:
        print(f"\n{'='*70}")
        print(f"Re-scoring Sample {sample_id}")
        print(f"{'='*70}")

        if args.input_results:
            input_file = args.input_results
        else:
            model_name_normalized = args.model.replace(".", "_").replace("-", "_")
            model_specific_file = f"results_{model_name_normalized}/fixed_results_sample{sample_id}.json"
            default_file = f"results/fixed_results_sample{sample_id}.json"

            if Path(model_specific_file).exists():
                input_file = model_specific_file
            elif Path(default_file).exists():
                input_file = default_file
            else:
                input_file = default_file

        if not Path(input_file).exists():
            print(f"Error: Results file not found: {input_file}")
            print(f"Please run the full test first or specify --input-results")
            continue

        with open(input_file, 'r') as f:
            data = json.load(f)

        existing_results = data.get('results', [])
        print(f"Loaded {len(existing_results)} results from {input_file}")

        updated_results = []
        print("Re-evaluating answers...")

        for result in tqdm(existing_results, desc="Re-scoring"):
            question = result['question']
            expected = result['expected']
            predicted = result['predicted']
            category = result.get('category')

            eval_result = evaluator.evaluate_answer(
                question,
                str(expected) if expected else "",
                predicted,
                question_category=category
            )

            result['metrics'] = eval_result.get('metrics')
            result['correct'] = eval_result.get('is_correct', False)
            result['llm_judge_score'] = eval_result.get('llm_judge_score', 0.0)

            updated_results.append(result)

        total = len(updated_results)
        correct = sum(1 for r in updated_results if r['correct'])
        not_found = sum(1 for r in updated_results if r['predicted'] == "Information not found")

        f1_scores = [r['metrics']['f1'] for r in updated_results if r.get('metrics') and 'f1' in r['metrics']]
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

        bleu1_scores = [r['metrics']['bleu1'] for r in updated_results if r.get('metrics') and 'bleu1' in r['metrics']]
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100 if bleu1_scores else 0

        llm_scores = [r.get('llm_judge_score', 0) for r in updated_results if 'llm_judge_score' in r and r['llm_judge_score'] >= 0]
        avg_llm_score = sum(llm_scores) / len(llm_scores) * 100 if llm_scores else 0

        results_no_cat5 = [r for r in updated_results if r.get('category') != 5]
        if results_no_cat5:
            total_no_cat5 = len(results_no_cat5)
            correct_no_cat5 = sum(1 for r in results_no_cat5 if r['correct'])
            f1_no_cat5 = sum(r['metrics']['f1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            bleu1_no_cat5 = sum(r['metrics']['bleu1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            llm_no_cat5 = sum(r.get('llm_judge_score', 0) for r in results_no_cat5 if 'llm_judge_score' in r and r['llm_judge_score'] >= 0) / total_no_cat5 * 100 if total_no_cat5 > 0 else 0
        else:
            total_no_cat5 = correct_no_cat5 = f1_no_cat5 = bleu1_no_cat5 = llm_no_cat5 = 0

        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'f1': [], 'bleu1': [], 'llm': []})
        for r in updated_results:
            cat = r.get('category', 0)
            category_stats[cat]['total'] += 1
            if r['correct']:
                category_stats[cat]['correct'] += 1
            if r.get('metrics'):
                category_stats[cat]['f1'].append(r['metrics'].get('f1', 0))
                category_stats[cat]['bleu1'].append(r['metrics'].get('bleu1', 0))
            if 'llm_judge_score' in r and r['llm_judge_score'] >= 0:
                category_stats[cat]['llm'].append(r['llm_judge_score'])

        print(f"\n{'='*70}")
        print(f"Sample {sample_id} Re-scored Results (All Categories):")
        print(f"  Total Questions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Average F1: {avg_f1:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1:.1f}%")
        print(f"  Average LLM Judge Score: {avg_llm_score:.1f}%")
        print(f"  Information not found: {not_found} ({not_found/total*100:.1f}%)")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Re-scored Results WITHOUT Category 5:")
        if results_no_cat5:
            print(f"  Total Questions: {total_no_cat5}")
            print(f"  Correct: {correct_no_cat5}")
            print(f"  Accuracy: {correct_no_cat5/total_no_cat5*100:.1f}%")
            print(f"  Average F1: {f1_no_cat5:.1f}%")
            print(f"  Average BLEU-1: {bleu1_no_cat5:.1f}%")
            print(f"  Average LLM Judge Score: {llm_no_cat5:.1f}%")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Re-scored Results BY CATEGORY:")
        print(f"  {'Cat':<5} {'Total':<7} {'Correct':<8} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7}")
        print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            print(f"  {cat:<5} {stats['total']:<7} {stats['correct']:<8} {acc:<7.1f} {avg_f1_cat:<7.1f} {avg_bleu1_cat:<7.1f} {avg_llm_cat:<7.1f}")

        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/rescored_results_sample{sample_id}.json"
        category_breakdown = {}
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            category_breakdown[f"category_{cat}"] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': acc,
                'avg_f1': avg_f1_cat,
                'avg_bleu1': avg_bleu1_cat,
                'avg_llm': avg_llm_cat
            }

        with open(output_file, 'w') as f:
            json.dump({
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'rescoring_model': args.model,
                'original_file': input_file,
                'results': updated_results,
                'stats': {
                    'overall': {
                        'total': total,
                        'correct': correct,
                        'accuracy': correct/total*100,
                        'avg_f1': avg_f1,
                        'avg_bleu1': avg_bleu1,
                        'avg_llm': avg_llm_score,
                        'not_found': not_found
                    },
                    'without_category5': {
                        'total': total_no_cat5,
                        'correct': correct_no_cat5,
                        'accuracy': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
                        'avg_f1': f1_no_cat5,
                        'avg_bleu1': bleu1_no_cat5,
                        'avg_llm': llm_no_cat5
                    },
                    'category_breakdown': category_breakdown
                }
            }, f, indent=2, default=str)

        print(f"\nRe-scored results saved to {output_file}")

        wrong_answers = [r for r in test_results if r.get('llm_judge_score', 0) < 0.5]
        if wrong_answers:
            wrong_output_file = f"{results_dir}/fixed_results_sample{sample_id}_wrong.json"

            category_names = {
                1: "Multi-hop",
                2: "Temporal",
                3: "Open-domain",
                4: "Single-hop",
                5: "Adversarial"
            }

            formatted_wrong = []
            for wa in wrong_answers:
                formatted_wrong.append({
                    'q_id': wa['question_id'],
                    'category': f"{wa['category']} ({category_names.get(wa['category'], 'Unknown')})",
                    'question': wa['question'],
                    'expected': wa['expected'],
                    'predicted': wa['predicted'],
                    'f1': wa.get('metrics', {}).get('f1', 0),
                    'llm_score': wa.get('llm_judge_score', 0),
                })

            wrong_summary = {
                'sample_id': sample_id,
                'total_questions': len(test_results),
                'total_wrong': len(wrong_answers),
                'wrong_percentage': len(wrong_answers) / len(test_results) * 100,
                'wrong_questions': formatted_wrong
            }

            with open(wrong_output_file, 'w') as f:
                json.dump(wrong_summary, f, indent=2, default=str)

            print(f"Wrong answers ({len(wrong_answers)}) saved to {wrong_output_file}")

        all_sample_results.append({
            'sample_id': sample_id,
            'total': total,
            'correct': correct,
            'accuracy': correct/total*100,
            'avg_f1': avg_f1,
            'avg_bleu1': avg_bleu1,
            'avg_llm': avg_llm_score,
            'accuracy_no_cat5': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
            'category_breakdown': category_breakdown
        })

    if len(args.sample) > 1:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE RE-SCORED RESULTS ACROSS {len(args.sample)} SAMPLES")
        print(f"{'='*70}\n")

        avg_accuracy = sum(r['accuracy'] for r in all_sample_results) / len(all_sample_results)
        avg_f1_overall = sum(r['avg_f1'] for r in all_sample_results) / len(all_sample_results)
        avg_bleu1_overall = sum(r['avg_bleu1'] for r in all_sample_results) / len(all_sample_results)
        avg_llm_overall = sum(r['avg_llm'] for r in all_sample_results) / len(all_sample_results)
        avg_accuracy_no_cat5 = sum(r['accuracy_no_cat5'] for r in all_sample_results) / len(all_sample_results)

        print("Per-Sample Breakdown:")
        for r in all_sample_results:
            print(f"  Sample {r['sample_id']}: Acc={r['accuracy']:.1f}%, F1={r['avg_f1']:.1f}%, BLEU-1={r['avg_bleu1']:.1f}%, LLM={r['avg_llm']:.1f}%")

        print(f"\nAggregated Metrics (Average across all samples):")
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Average F1: {avg_f1_overall:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1_overall:.1f}%")
        print(f"  Average LLM Judge: {avg_llm_overall:.1f}%")
        print(f"  Average Accuracy (no Cat5): {avg_accuracy_no_cat5:.1f}%")

        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        aggregate_output = f"{results_dir}/rescored_results_aggregate_samples_{'_'.join(map(str, args.sample))}.json"
        with open(aggregate_output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'rescoring_model': args.model,
                'samples': args.sample,
                'per_sample_results': all_sample_results,
                'aggregate': {
                    'avg_accuracy': avg_accuracy,
                    'avg_f1': avg_f1_overall,
                    'avg_bleu1': avg_bleu1_overall,
                    'avg_llm': avg_llm_overall,
                    'avg_accuracy_no_cat5': avg_accuracy_no_cat5
                }
            }, f, indent=2)
        print(f"\nAggregate re-scored results saved to {aggregate_output}")

    return 0

def main():
    """Main test function - supports multiple samples"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--sample", type=int, nargs='+', default=[0],
                       help="Sample IDs to test (can specify multiple, e.g., --sample 0 1 2)")
    parser.add_argument("--max-questions", type=int, default=2000)
    parser.add_argument("--cache-dir", default="./locomo_trg_fixed")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild memory")
    parser.add_argument("--use-azure", action="store_true", default=True,
                       help="Use Azure OpenAI instead of regular OpenAI (default: True)")
    parser.add_argument("--use-openai", action="store_true",
                       help="Use regular OpenAI instead of Azure OpenAI (overrides --use-azure)")
    parser.add_argument("--backend", type=str, default=None,
                       choices=["openai", "ollama"],
                       help="LLM backend to use. 'openai' for GPT models, 'ollama' for local models like Qwen")
    parser.add_argument("--model", type=str, default=None,
                       help="Model to use. For OpenAI/Azure: gpt-4, gpt-4o-mini, etc. For Ollama: qwen2.5:3b, qwen2.5:7b, llama2, etc.")
    parser.add_argument("--max-top-k", type=int, default=None,
                       help="Cap top_k for retrieval (e.g. 10 for small models). Default: no cap (30 for Cat1, 15 for others)")
    parser.add_argument("--embedding-model", type=str, default="minilm",
                       choices=["minilm", "openai"],
                       help="Embedding model to use: 'minilm' (all-MiniLM-L6-v2, 384-dim) or 'openai' (text-embedding-3-small, 1536-dim)")
    parser.add_argument("--use-episodes", action="store_true",
                       help="Use episode-based segmentation instead of turn-based (groups related turns)")
    parser.add_argument("--score-only", action="store_true",
                       help="Only re-score existing results without rebuilding memory or querying")
    parser.add_argument("--input-results", type=str, default=None,
                       help="Input results file to re-score (used with --score-only)")
    parser.add_argument("--skip-category-5", action="store_true",
                       help="Skip category 5 (Adversarial) questions during testing")
    parser.add_argument("--category-to-test", type=str, default="1,2,3,4,5",
                       help="Comma-separated list of categories to test (e.g., '1,2,3,4' or '1,3'). Default: '1,2,3,4'")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel testing (parallel is enabled by default for 3x speedup)")
    parser.add_argument("--n-workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    parser.add_argument("--best-of-n", type=int, default=4,
                       help="Run each question N times and select best answer (default: 4 = best-of-4)")
    parser.add_argument("--best-of-n-method", type=str, default="llm_judge",
                       choices=["llm_judge", "voting", "f1"],
                       help="Method for selecting best answer: 'llm_judge', 'voting', or 'f1' (default: llm_judge)")
    parser.add_argument("--ablation", type=str, default=None,
                       choices=["basic_retrieval", "no_causal", "no_temporal", "flat_graph"],
                       help="Run ablation study with specific configuration")
    parser.add_argument("--edge-scorer", type=str, default="static",
                       choices=["static", "llm", "hybrid", "rl_transductive", "rl_inductive",
                                "rl_coevo", "rl_router_only", "rl_edge_only"],
                       help="Edge scoring method: 'static' (baseline), 'llm' (LLM-based), 'hybrid', "
                            "'rl_transductive' (Phase 2), 'rl_inductive' (Phase 2), "
                            "'rl_coevo' (Phase 3 joint), 'rl_router_only' (Phase 3 ablation), "
                            "'rl_edge_only' (Phase 3 ablation)")
    parser.add_argument("--edge-scorer-cache", type=str,
                       default="./edge_score_cache",
                       help="Cache directory for LLM edge scores")
    parser.add_argument("--edge-scorer-prompt", type=str, default="v1",
                       choices=["v1", "v2", "v3", "v4", "v5", "v6"],
                       help="Which prompt template to use for LLM scorer (v1=basic, v2=CoT, v3=few-shot)")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                       help="Path to RL model checkpoint (for rl_transductive/rl_inductive edge scorers)")
    parser.add_argument("--rl-graph-cache", type=str, default="locomo_trg_cache_gpt_4o_mini",
                       help="Graph cache directory for RL edge scorer")
    parser.add_argument("--rl-alpha", type=float, default=1.0,
                       help="RL/static blending weight: 1.0=pure RL, 0.5=50/50 blend, 0.0=pure static")
    parser.add_argument("--val-only", action="store_true",
                       help="Only evaluate on the 20%% held-out val QAs from transductive RL training (seed=42). "
                            "Use this for fair comparison with Trainable Edge results.")
    parser.add_argument("--val-seed", type=int, default=42,
                       help="Random seed used for the transductive train/val split (default: 42)")
    args = parser.parse_args()

    # Parallel is enabled by default
    args.parallel = not args.no_parallel

    # Determine backend and configure accordingly
    if args.backend == "ollama":
        # Using Ollama for local models
        api_type = "Ollama (Local)"
        if not args.model:
            args.model = "qwen2.5:3b"  # Default Ollama model
        # No API key needed for Ollama
        use_azure = False
    else:
        # Default to OpenAI/Azure
        args.backend = "openai"

        # Configure API settings (Azure or OpenAI)
        if args.use_openai:
            use_azure = False
        else:
            # Default to Azure (--use-azure is True by default)
            use_azure = True
            # But check environment variable if set
            env_use_azure = os.getenv("USE_AZURE_OPENAI", "").lower()
            if env_use_azure == "false":
                use_azure = False

        if use_azure:
            # Set environment variable for openai_client module
            os.environ['USE_AZURE_OPENAI'] = 'true'
            # For Azure, use deployment name if not specified
            if not args.model:
                args.model = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4.1-mini')
            # Set deployment name for openai_client module
            os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = args.model
            api_type = "Azure OpenAI"
        else:
            # Set environment variable for openai_client module
            os.environ['USE_AZURE_OPENAI'] = 'false'
            # For OpenAI, use default model if not specified
            if not args.model:
                args.model = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')
            api_type = "OpenAI"

    print("="*70)
    print("  Fixed TRG Memory System Test")
    print(f"  API: {api_type}")
    print(f"  Model: {args.model}")
    if args.score_only:
        print(f"  Mode: SCORE-ONLY (Re-evaluation)")
    else:
        print(f"  Mode: {'Episode-based' if args.use_episodes else 'Turn-based'}")
    print(f"  Samples: {args.sample}")

    # Parse categories to test
    categories_to_test = [int(c.strip()) for c in args.category_to_test.split(',')]

    # Handle legacy skip_category_5 flag
    if args.skip_category_5 and 5 in categories_to_test:
        categories_to_test.remove(5)

    print(f"  Categories to test: {sorted(categories_to_test)}")
    category_names = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}
    print(f"  Category types: {', '.join([f'{c}:{category_names.get(c, 'Unknown')}' for c in sorted(categories_to_test)])}")

    # Show parallel mode status
    # if args.parallel:
    #     print(f"  Parallel mode: ✓ ENABLED ({args.n_workers} workers, ~{args.n_workers}x speedup)")
    # else:
    #     print(f"  Parallel mode: ✗ DISABLED (use default for 3x speedup)")

    print("="*70)

    # Handle score-only mode
    if args.score_only:
        return score_only_mode(args)

    # Setup logging to file
    model_name_normalized = args.model.replace(".", "_").replace("-", "_")
    log_dir = f"logs_{model_name_normalized}"
    log_file = setup_logging(log_dir, args.model)
    print(f"Logging to: {log_file}")
    print("="*70)

    # Load dataset
    samples = load_locomo_dataset(args.dataset)

    # Validate sample IDs
    for sample_id in args.sample:
        if sample_id < 0 or sample_id >= len(samples):
            print(f"Error: Sample ID {sample_id} is out of range (0-{len(samples)-1})")
            return 1

    # Process each sample
    all_sample_results = []

    for sample_idx, sample_id in enumerate(args.sample, 1):
        sample = samples[sample_id]

        print(f"\n{'='*70}")
        print(f"Processing Sample {sample_id} ({sample_idx}/{len(args.sample)})")
        print(f"{'='*70}")

        # Auto-generate cache directory with sample number, embedding model, and LLM model
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Normalize model name for folder (replace dots and hyphens with underscores)
        model_name_normalized = args.model.replace(".", "_").replace("-", "_")

        if args.cache_dir == "./locomo_trg_fixed":
            # User didn't specify custom cache dir, use auto-naming with model name
            if args.use_episodes:
                cache_dir = f"./locomo_trg_episodes_{model_name_normalized}/sample{sample_id}{embedding_suffix}"
            else:
                cache_dir = f"./locomo_trg_cache_{model_name_normalized}/sample{sample_id}{embedding_suffix}"
            print(f"Auto cache directory: {cache_dir}")
            print(f"Embedding model: {args.embedding_model}")
        else:
            # User specified custom cache dir with sample ID in subdirectory
            cache_dir = f"{args.cache_dir}/sample{sample_id}{embedding_suffix}"
            print(f"Custom cache directory: {cache_dir}")
            print(f"Embedding model: {args.embedding_model}")

        # Initialize memory builder with backend support
        builder = MemoryBuilder(
            cache_dir=cache_dir,
            llm_model=args.model,
            use_episodes=args.use_episodes,
            embedding_model=args.embedding_model,
            llm_backend=args.backend
        )

        # Build or load memory
        cache_file = Path(cache_dir) / "graph.json"
        if cache_file.exists() and not args.rebuild:
            logger.info("Loading cached memory...")
            builder.load()
        else:
            logger.info("Building memory...")
            builder.build_memory(sample)
            builder.save()

        # Get memory stats
        mem_stats = builder.trg.get_statistics()
        print(f"\nMemory Statistics:")
        print(f"  Total nodes: {mem_stats['total_nodes']}")
        print(f"  Total links: {mem_stats['links_created']}")
        if mem_stats['total_nodes'] > 0:
            print(f"  Links per node: {mem_stats['links_created']/mem_stats['total_nodes']:.1f}")
        print(f"  Node types: {mem_stats['node_types']}")
        print(f"  Link types: {mem_stats['link_types']}")

        # Initialize query engine with entity-session mapping
        # Prepare ablation configuration
        ablation_config = {}
        if args.ablation:
            ablation_config[args.ablation] = True
            print(f"  ⚠️ ABLATION MODE: {args.ablation}")

        # Initialize edge scorer based on args
        from memory.edge_scorer import StaticEdgeScorer, HybridEdgeScorer
        from memory.edge_scorer_v2 import LLMEdgeScorerV2

        if args.edge_scorer == "static":
            edge_scorer = StaticEdgeScorer()
            print(f"  Edge Scorer: Static (baseline HAGE)")
        elif args.edge_scorer == "llm":
            # Use V2: O(1) LLM calls per query (pure, fast architecture)
            edge_scorer = LLMEdgeScorerV2(
                llm_controller=builder.llm_controller,
                cache_dir=args.edge_scorer_cache,
                use_cache=True,
                temperature=0.0,
                fallback_to_static=True
            )
            print(f"  Edge Scorer: LLM-V2 (O(1) per query, cache={args.edge_scorer_cache})")
        elif args.edge_scorer == "hybrid":
            # Hybrid mode uses V2 for LLM component
            llm_scorer = LLMEdgeScorerV2(
                llm_controller=builder.llm_controller,
                cache_dir=args.edge_scorer_cache,
                use_cache=True,
                temperature=0.0,
                fallback_to_static=True
            )
            static_scorer = StaticEdgeScorer()
            edge_scorer = HybridEdgeScorer(
                llm_scorer=llm_scorer,
                static_scorer=static_scorer,
                combination_method="weighted_average",
                llm_weight=0.7
            )
            print(f"  Edge Scorer: Hybrid (70% LLM + 30% Static, prompt={args.edge_scorer_prompt})")
        elif args.edge_scorer in ("rl_transductive", "rl_inductive",
                                    "rl_coevo", "rl_router_only", "rl_edge_only"):
            from memory.rl_edge_adapter import RLEdgeScorerAdapter
            # Map CLI names to adapter mode names
            rl_mode_map = {
                "rl_transductive": "transductive",
                "rl_inductive": "inductive",
                "rl_coevo": "coevo",
                "rl_router_only": "router_only",
                "rl_edge_only": "edge_only",
            }
            rl_mode = rl_mode_map[args.edge_scorer]
            # Default checkpoint paths if not specified
            if args.rl_checkpoint:
                rl_ckpt = args.rl_checkpoint
            elif rl_mode == "transductive":
                rl_ckpt = f"checkpoints/transductive_s{sample_id}/model_final.pt"
            elif rl_mode == "inductive":
                rl_ckpt = "checkpoints/inductive/model_final.pt"
            elif rl_mode == "coevo":
                # Best lambda sweep checkpoint (lambda=1.0, epoch 50)
                rl_ckpt = "results_lambda_sweep/ckpt_lambda_1.0/model_epoch_50.pt"
            elif rl_mode == "router_only":
                rl_ckpt = "results_coevo_ablation_v4/ckpt_router_only/model_final.pt"
            elif rl_mode == "edge_only":
                rl_ckpt = "results_coevo_ablation_v4/ckpt_edge_only/model_final.pt"
            edge_scorer = RLEdgeScorerAdapter.from_checkpoint(
                checkpoint_path=rl_ckpt,
                graph_cache_dir=args.rl_graph_cache,
                sample_id=sample_id,
                mode=rl_mode,
                alpha=args.rl_alpha,
            )
            alpha_str = f", alpha={args.rl_alpha}" if args.rl_alpha < 1.0 else ""
            print(f"  Edge Scorer: RL {rl_mode} (checkpoint={rl_ckpt}{alpha_str})")
        else:
            edge_scorer = StaticEdgeScorer()  # Fallback

        query_engine = QueryEngine(
            builder.trg,
            builder.node_index,
            entity_session_map=builder.entity_session_map if hasattr(builder, 'entity_session_map') else None,
            entity_dia_map=builder.entity_dia_map if hasattr(builder, 'entity_dia_map') else None,
            ablation_config=ablation_config,
            edge_scorer=edge_scorer  # NEW: Pass edge scorer
        )

        # Initialize evaluator (separate from memory and query)
        evaluator = Evaluator(
            llm_controller=builder.llm_controller,
            use_llm_judge=True
        )

        # Initialize test harness with evaluator
        # Auto-set max_top_k and simple prompts for small local models
        max_top_k = args.max_top_k
        use_simple_prompts = False
        if args.backend == "ollama":
            if max_top_k is None:
                max_top_k = 10  # Small models can't handle 30-node contexts
            use_simple_prompts = True
            print(f"  Ollama mode: max_top_k={max_top_k}, simple_prompts=True (override with --max-top-k)")
        tester = TestHarness(builder, query_engine, evaluator=evaluator, max_top_k=max_top_k, use_simple_prompts=use_simple_prompts)

        # Setup best-of-N if requested
        if args.best_of_n > 1:
            from memory.best_of_n_selector import BestOfNSelector
            # print(f"\n✓ Using Best-of-{args.best_of_n} selection (method: {args.best_of_n_method})")
            tester.best_of_n = args.best_of_n
            tester.best_of_n_method = args.best_of_n_method
            tester.best_of_n_selector = BestOfNSelector(
                n_attempts=args.best_of_n,
                selection_method=args.best_of_n_method
            )

        # --val-only: Filter to only the 20% held-out val QAs from transductive training
        if args.val_only:
            import numpy as np
            # Reproduce the same filtering as QADataset: skip QAs without 'answer' or ungrounded evidence
            # We match by question text since QADataset filters differently than sample.qa
            graph_cache_dir = args.rl_graph_cache if hasattr(args, 'rl_graph_cache') else "locomo_trg_cache_gpt_4o_mini"
            graph_json_path = os.path.join(graph_cache_dir, f"sample{sample_id}", "graph.json")

            if os.path.exists(graph_json_path):
                from memory.pytorch_graph import load_graph_json, build_pytorch_graph, PyTorchGraph
                graph_json = load_graph_json(graph_json_path)
                pg = build_pytorch_graph(graph_json, sample_id=sample_id)

                # Reproduce QADataset filtering for this sample
                locomo_data = json.load(open("data/locomo10.json"))
                grounded_questions = []
                for qa in locomo_data[sample_id].get("qa", []):
                    answer = qa.get("answer")
                    if not answer:
                        continue
                    evidence = qa.get("evidence", [])
                    target_indices = pg.resolve_evidence(evidence)
                    if not target_indices:
                        continue
                    grounded_questions.append(qa["question"])

                # Reproduce the same 80/20 permutation split
                n = len(grounded_questions)
                split = int(n * 0.8)
                np.random.seed(args.val_seed)
                perm = np.random.permutation(n).tolist()
                val_questions = set(grounded_questions[i] for i in perm[split:])

                before_count = len(sample.qa)
                sample.qa = [qa for qa in sample.qa if qa.question in val_questions]
                print(f"\n--val-only: Filtered to {len(sample.qa)} held-out val QAs (from {before_count} total, seed={args.val_seed})")
            else:
                print(f"\nWARNING: Graph cache not found at {graph_json_path}, cannot filter val-only. Running all QAs.")

        # Filter questions based on categories to test
        original_count = len(sample.qa)
        sample.qa = [qa for qa in sample.qa if qa.category in categories_to_test]
        filtered_count = len(sample.qa)

        if filtered_count < original_count:
            print(f"\nFiltered questions: {original_count} → {filtered_count}")
            print(f"Testing categories: {sorted(categories_to_test)}\n")
        else:
            print(f"\nTesting all {filtered_count} questions\n")

        # Test questions (parallel is now default)
        if args.parallel:
            results = tester.test_questions_parallel(sample, args.max_questions, n_workers=args.n_workers)
        else:
            results = tester.test_questions(sample, args.max_questions)

        # Calculate results for this sample
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        not_found = sum(1 for r in results if r['predicted'] == "Information not found")

        # Calculate average scores
        f1_scores = [r['metrics']['f1'] for r in results if r.get('metrics') and 'f1' in r['metrics']]
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0

        bleu1_scores = [r['metrics']['bleu1'] for r in results if r.get('metrics') and 'bleu1' in r['metrics']]
        avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100 if bleu1_scores else 0

        llm_scores = [r.get('llm_judge_score', 0) for r in results if 'llm_judge_score' in r and r['llm_judge_score'] >= 0]
        avg_llm_score = sum(llm_scores) / len(llm_scores) * 100 if llm_scores else 0

        # Calculate scores WITHOUT Category 5
        results_no_cat5 = [r for r in results if r.get('category') != 5]
        if results_no_cat5:
            total_no_cat5 = len(results_no_cat5)
            correct_no_cat5 = sum(1 for r in results_no_cat5 if r['correct'])
            f1_no_cat5 = sum(r['metrics']['f1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            bleu1_no_cat5 = sum(r['metrics']['bleu1'] for r in results_no_cat5 if r.get('metrics')) / total_no_cat5 * 100
            llm_no_cat5 = sum(r.get('llm_judge_score', 0) for r in results_no_cat5 if 'llm_judge_score' in r and r['llm_judge_score'] >= 0) / total_no_cat5 * 100 if total_no_cat5 > 0 else 0
        else:
            total_no_cat5 = correct_no_cat5 = f1_no_cat5 = bleu1_no_cat5 = llm_no_cat5 = 0

        # Calculate category-wise breakdown
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'f1': [], 'bleu1': [], 'llm': []})
        for r in results:
            cat = r.get('category', 0)
            category_stats[cat]['total'] += 1
            if r['correct']:
                category_stats[cat]['correct'] += 1
            if r.get('metrics'):
                category_stats[cat]['f1'].append(r['metrics'].get('f1', 0))
                category_stats[cat]['bleu1'].append(r['metrics'].get('bleu1', 0))
            if 'llm_judge_score' in r and r['llm_judge_score'] >= 0:
                category_stats[cat]['llm'].append(r['llm_judge_score'])

        # Print results for this sample
        print(f"\n{'='*70}")
        print(f"Sample {sample_id} Results (All Categories):")
        print(f"  Total Questions: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {correct/total*100:.1f}%")
        print(f"  Average F1: {avg_f1:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1:.1f}%")
        print(f"  Average LLM Judge Score: {avg_llm_score:.1f}%")
        print(f"  Information not found: {not_found} ({not_found/total*100:.1f}%)")

        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Results WITHOUT Category 5:")
        if results_no_cat5:
            print(f"  Total Questions: {total_no_cat5}")
            print(f"  Correct: {correct_no_cat5}")
            print(f"  Accuracy: {correct_no_cat5/total_no_cat5*100:.1f}%")
            print(f"  Average F1: {f1_no_cat5:.1f}%")
            print(f"  Average BLEU-1: {bleu1_no_cat5:.1f}%")
            print(f"  Average LLM Judge Score: {llm_no_cat5:.1f}%")

        # Print category breakdown
        print(f"\n{'-'*70}")
        print(f"Sample {sample_id} Results BY CATEGORY:")
        print(f"  {'Cat':<5} {'Total':<7} {'Correct':<8} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7}")
        print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            print(f"  {cat:<5} {stats['total']:<7} {stats['correct']:<8} {acc:<7.1f} {avg_f1_cat:<7.1f} {avg_bleu1_cat:<7.1f} {avg_llm_cat:<7.1f}")

        # Save per-sample results with model-specific directory
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Create model-specific results directory
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        output_file = f"{results_dir}/fixed_results_sample{sample_id}{embedding_suffix}.json"
        category_breakdown = {}
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_f1_cat = sum(stats['f1']) / len(stats['f1']) * 100 if stats['f1'] else 0
            avg_bleu1_cat = sum(stats['bleu1']) / len(stats['bleu1']) * 100 if stats['bleu1'] else 0
            avg_llm_cat = sum(stats['llm']) / len(stats['llm']) * 100 if stats['llm'] else 0
            category_breakdown[f"category_{cat}"] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': acc,
                'avg_f1': avg_f1_cat,
                'avg_bleu1': avg_bleu1_cat,
                'avg_llm': avg_llm_cat
            }

        with open(output_file, 'w') as f:
            json.dump({
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'embedding_model': args.embedding_model,
                'llm_model': args.model,
                'results': results,
                'stats': {
                    'overall': {
                        'total': total,
                        'correct': correct,
                        'accuracy': correct/total*100,
                        'avg_f1': avg_f1,
                        'avg_bleu1': avg_bleu1,
                        'avg_llm': avg_llm_score,
                        'not_found': not_found
                    },
                    'without_category5': {
                        'total': total_no_cat5,
                        'correct': correct_no_cat5,
                        'accuracy': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
                        'avg_f1': f1_no_cat5,
                        'avg_bleu1': bleu1_no_cat5,
                        'avg_llm': llm_no_cat5
                    },
                    'category_breakdown': category_breakdown
                }
            }, f, indent=2, default=str)

        print(f"Results saved to {output_file}")

        # Store for aggregation
        all_sample_results.append({
            'sample_id': sample_id,
            'total': total,
            'correct': correct,
            'accuracy': correct/total*100,
            'avg_f1': avg_f1,
            'avg_bleu1': avg_bleu1,
            'avg_llm': avg_llm_score,
            'accuracy_no_cat5': correct_no_cat5/total_no_cat5*100 if total_no_cat5 > 0 else 0,
            'category_breakdown': category_breakdown
        })

    # Print summary if multiple samples
    if len(args.sample) > 1:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE RESULTS ACROSS {len(args.sample)} SAMPLES")
        print(f"{'='*70}\n")

        logger.info(f"{'='*70}")
        logger.info(f"AGGREGATE RESULTS ACROSS {len(args.sample)} SAMPLES")
        logger.info(f"{'='*70}")

        # Calculate averages
        avg_accuracy = sum(r['accuracy'] for r in all_sample_results) / len(all_sample_results)
        avg_f1_overall = sum(r['avg_f1'] for r in all_sample_results) / len(all_sample_results)
        avg_bleu1_overall = sum(r['avg_bleu1'] for r in all_sample_results) / len(all_sample_results)
        avg_llm_overall = sum(r['avg_llm'] for r in all_sample_results) / len(all_sample_results)
        avg_accuracy_no_cat5 = sum(r['accuracy_no_cat5'] for r in all_sample_results) / len(all_sample_results)

        print("Per-Sample Breakdown:")
        for r in all_sample_results:
            print(f"  Sample {r['sample_id']}: Acc={r['accuracy']:.1f}%, F1={r['avg_f1']:.1f}%, BLEU-1={r['avg_bleu1']:.1f}%, LLM={r['avg_llm']:.1f}%")

        print(f"\nAggregated Metrics (Average across all samples):")
        print(f"  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"  Average F1: {avg_f1_overall:.1f}%")
        print(f"  Average BLEU-1: {avg_bleu1_overall:.1f}%")
        print(f"  Average LLM Judge: {avg_llm_overall:.1f}%")
        print(f"  Average Accuracy (no Cat5): {avg_accuracy_no_cat5:.1f}%")

        logger.info(f"Aggregated Metrics (Average across {len(args.sample)} samples):")
        logger.info(f"  Average Accuracy: {avg_accuracy:.1f}%")
        logger.info(f"  Average F1: {avg_f1_overall:.1f}%")
        logger.info(f"  Average BLEU-1: {avg_bleu1_overall:.1f}%")
        logger.info(f"  Average LLM Judge: {avg_llm_overall:.1f}%")
        logger.info(f"  Average Accuracy (no Cat5): {avg_accuracy_no_cat5:.1f}%")

        # Aggregate category breakdown across all samples
        print(f"\n{'-'*70}")
        print(f"AGGREGATE RESULTS BY CATEGORY (Average across {len(args.sample)} samples):")

        # Collect all categories across all samples
        all_categories = set()
        for r in all_sample_results:
            all_categories.update(r['category_breakdown'].keys())

        # Calculate average stats per category
        aggregate_category_stats = {}
        for cat_key in sorted(all_categories, key=lambda x: int(x.split('_')[1])):
            cat_num = int(cat_key.split('_')[1])
            total_samples_with_cat = 0
            sum_total = sum_correct = sum_acc = sum_f1 = sum_bleu = sum_llm = 0

            for r in all_sample_results:
                if cat_key in r['category_breakdown']:
                    cat_data = r['category_breakdown'][cat_key]
                    total_samples_with_cat += 1
                    sum_total += cat_data['total']
                    sum_correct += cat_data['correct']
                    sum_acc += cat_data['accuracy']
                    sum_f1 += cat_data['avg_f1']
                    sum_bleu += cat_data['avg_bleu1']
                    sum_llm += cat_data['avg_llm']

            if total_samples_with_cat > 0:
                aggregate_category_stats[cat_num] = {
                    'avg_total': sum_total / total_samples_with_cat,
                    'avg_correct': sum_correct / total_samples_with_cat,
                    'avg_accuracy': sum_acc / total_samples_with_cat,
                    'avg_f1': sum_f1 / total_samples_with_cat,
                    'avg_bleu1': sum_bleu / total_samples_with_cat,
                    'avg_llm': sum_llm / total_samples_with_cat,
                    'samples_count': total_samples_with_cat
                }

        print(f"  {'Cat':<5} {'Avg Total':<10} {'Avg Corr':<10} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7} {'#Samples':<9}")
        print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")

        logger.info(f"AGGREGATE RESULTS BY CATEGORY (Average across {len(args.sample)} samples):")
        logger.info(f"  {'Cat':<5} {'Avg Total':<10} {'Avg Corr':<10} {'Acc%':<7} {'F1%':<7} {'BLEU%':<7} {'LLM%':<7} {'#Samples':<9}")
        logger.info(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")

        for cat in sorted(aggregate_category_stats.keys()):
            stats = aggregate_category_stats[cat]
            print(f"  {cat:<5} {stats['avg_total']:<10.1f} {stats['avg_correct']:<10.1f} "
                  f"{stats['avg_accuracy']:<7.1f} {stats['avg_f1']:<7.1f} {stats['avg_bleu1']:<7.1f} "
                  f"{stats['avg_llm']:<7.1f} {stats['samples_count']:<9}")
            logger.info(f"  {cat:<5} {stats['avg_total']:<10.1f} {stats['avg_correct']:<10.1f} "
                       f"{stats['avg_accuracy']:<7.1f} {stats['avg_f1']:<7.1f} {stats['avg_bleu1']:<7.1f} "
                       f"{stats['avg_llm']:<7.1f} {stats['samples_count']:<9}")

        # Save aggregate results in model-specific directory
        embedding_suffix = "_openai" if args.embedding_model == "openai" else ""
        # Use same model-specific results directory
        model_name_normalized = args.model.replace(".", "_").replace("-", "_")
        results_dir = f"results_{model_name_normalized}"
        os.makedirs(results_dir, exist_ok=True)
        aggregate_output = f"{results_dir}/fixed_results_aggregate_samples_{'_'.join(map(str, args.sample))}{embedding_suffix}.json"
        with open(aggregate_output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'embedding_model': args.embedding_model,
                'llm_model': args.model,
                'samples': args.sample,
                'per_sample_results': all_sample_results,
                'aggregate': {
                    'avg_accuracy': avg_accuracy,
                    'avg_f1': avg_f1_overall,
                    'avg_bleu1': avg_bleu1_overall,
                    'avg_llm': avg_llm_overall,
                    'avg_accuracy_no_cat5': avg_accuracy_no_cat5,
                    'category_breakdown': aggregate_category_stats
                }
            }, f, indent=2)
        print(f"\nAggregate results saved to {aggregate_output}")
        logger.info(f"Aggregate results saved to {aggregate_output}")

    logger.info(f"{'='*70}")
    logger.info(f"Test run completed successfully")
    logger.info(f"{'='*70}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
