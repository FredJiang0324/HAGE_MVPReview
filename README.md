# HAGE: Harnessing Agentic Memory via RL-Driven Weighted Graph Evolution

**A weighted multi-relational memory framework with RL-driven graph traversal for long-horizon agentic reasoning.**

## Overview

**HAGE** is a principled memory system for long-term conversation memory and multi-hop reasoning. HAGE represents memory across four orthogonal relational graphs — **Semantic, Temporal, Causal, and Entity** — and introduces a co-evolutionary training framework that jointly optimizes trainable edge features and a query-conditioned QueryRouter MLP via policy-gradient reinforcement learning.

Key contributions:
1. **Weighted Multi-Relational Memory Graph**: each edge carries a trainable feature vector initialized by LLM-based scoring (Phase 1) and refined by RL (Phase 2).
2. **QueryRouter**: a 3-layer MLP that computes query-conditioned traversal scores without path-level supervision.
3. **Co-evolutionary Training**: REINFORCE jointly optimizes edge features and the router on evidence-hit rewards with zero additional LLM calls during training.

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Quick Start

### Testing with LoCoMo Dataset

```bash
# Test on LoCoMo dataset (10 samples included)
python test_fixed_memory.py --sample 0 --model gpt-4o-mini --max-questions 10 --category-to-test 1,2,3,4,5

# Test specific question categories
python test_fixed_memory.py --sample 0 --category-to-test 1  # Multi-hop only

# Test multiple samples
python test_fixed_memory.py --sample 0 1 2 --max-questions 50

# Full dataset path: data/locomo10.json
```

### Testing with HotpotQA Dataset

```bash
# Test on HotpotQA multi-hop questions
python test_hotpotqa.py --dataset data/hotpot_dev_distractor_v1.json --num-questions 100 --model gpt-4o-mini

# Parallel evaluation
python test_hotpotqa.py --num-questions 100 --parallel 10 --best-of-n 3
```

### RL Training (Phase 2)

```bash
# Train QueryRouter and edge features via REINFORCE
python train_rl_edge_scorer.py --dataset data/locomo10.json --epochs 100 --lr 1e-3
```

### 5-Fold Cross-Validation

```bash
# Run full 5-fold evaluation
python scripts/run_5fold_cv.py --dataset data/locomo10.json --model gpt-4o-mini
```

## Datasets

### 1. LoCoMo (Long Conversation Memory) — `data/locomo10.json`
- 10 conversation samples with extensive Q&A pairs
- 5 question categories: Multi-hop, Temporal, Open-domain, Single-hop, Adversarial
- **Status**: Included in repository (2.7MB)

### 2. HotpotQA — `data/hotpot_dev_distractor_v1.json`
- Multi-hop QA over 10 documents per question including distractors (7,405 dev questions)
- **Status**: Download from HuggingFace

```bash
mkdir -p data/ && cd data/
wget https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_dev_distractor_v1.json
cd ..
```

## Configuration

### Embedding Models

- **MiniLM** (default): Fast, offline, 384-dimensional embeddings
- **OpenAI**: Higher quality, requires API key, 1536-dimensional embeddings

```bash
python test_fixed_memory.py --embedding-model minilm   # default
python test_fixed_memory.py --embedding-model openai
```

### Supported LLM Backends

- `gpt-4o-mini` (default)
- `gpt-4.1-mini`
- `gpt-4o`
- Local models via compatible API (see `.env.example`)

## File Structure

```
HAGE/
├── memory/                       # Core memory and RL modules
│   ├── trg_memory.py             # Main memory engine
│   ├── memory_builder.py         # Memory construction pipeline
│   ├── query_engine.py           # Query processing and retrieval
│   ├── graph_db.py               # Graph database
│   ├── vector_db.py              # Vector database
│   ├── model.py                  # CoEvoMem model (QueryRouter + edge features)
│   ├── env.py                    # RL graph traversal environment
│   ├── rl_trainer.py             # REINFORCE training loop
│   ├── pytorch_graph.py          # PyTorch graph representation
│   ├── edge_scorer.py            # Phase 1 LLM-based edge scoring
│   ├── edge_scorer_v2.py         # Updated edge scorer
│   ├── rl_edge_adapter.py        # RL adapter for edge features
│   ├── llm_edge_prompts.py       # LLM prompts for edge scoring
│   ├── llm_judge.py              # LLM-as-a-Judge evaluator
│   ├── answer_formatter.py       # Answer formatting utilities
│   └── ...
├── utils/                        # Utility modules
│   ├── memory_layer.py           # LLM controller
│   ├── openai_client.py          # OpenAI/Azure client wrapper
│   └── utils.py                  # General utilities
├── scripts/                      # Evaluation and ablation scripts
│   ├── run_5fold_cv.py           # 5-fold cross-validation
│   ├── run_coevo_ablation.py     # Co-evolutionary ablation
│   ├── run_reward_ablation.py    # Reward function ablation
│   ├── run_hyperparams_ablation.py
│   ├── sweep_lambda.py           # Lambda hyperparameter sweep
│   ├── evaluate_phase2.py        # Phase 2 evaluation
│   └── measure_efficiency.py     # System efficiency measurement
├── data/                         # Datasets
│   ├── locomo10.json             # LoCoMo (included)
│   ├── hotpot_dev_distractor_v1.json  # HotpotQA (download required)
│   └── README.md                 # Dataset download instructions
├── examples/                     # Sample data for quick testing
│   └── locomo_sample.json
├── test_fixed_memory.py          # LoCoMo evaluation script
├── test_hotpotqa.py              # HotpotQA evaluation script
├── train_rl_edge_scorer.py       # RL training entry point
├── requirements.txt
└── .env.example
```

## Evaluation Metrics

- **Exact Match**: Binary correctness
- **F1 Score**: Token-level overlap (0–100%)
- **BLEU Score**: N-gram similarity (0–100%)
- **LLM Judge**: GPT-based semantic evaluation (0–100%)

## License

MIT License — see LICENSE file for details.
