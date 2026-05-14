<div align="center">

# HAGE

### Harnessing Agentic Memory via RL-Driven Weighted Graph Evolution

<small>A weighted multi-relational memory framework with RL-driven graph traversal for long-horizon agentic reasoning.</small>

<br/>

<p align="center">
  <a href="https://arxiv.org/abs/2605.09942"><img src="https://img.shields.io/badge/arXiv-2605.09942-b31b1b?style=flat&labelColor=555&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2605.09942"><img src="https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=flat&labelColor=555" alt="Hugging Face"></a>
  <a href="./HAGE_arxiv.pdf"><img src="https://img.shields.io/badge/Paper-PDF-EF4444?style=flat&labelColor=555" alt="Paper PDF"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EA44F?style=flat&labelColor=555" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat&labelColor=555&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=flat&labelColor=555&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat&labelColor=555" alt="PRs Welcome">
</p>

[🚀 Quick Start](#-quick-start) • [🌟 Overview](#-overview) • [📦 Installation](#-installation) • [📚 Datasets](#-datasets) • [⚙️ Configuration](#-configuration) • [📁 Project Structure](#-project-structure) • [📈 Evaluation](#-evaluation) • [📝 Citation](#-citation)

</div>

<br/>

---

## 🌟 Overview

**HAGE** is a principled memory system for long-term conversation memory and multi-hop reasoning. It represents memory across **four orthogonal relational graphs** — **Semantic**, **Temporal**, **Causal**, and **Entity** — and introduces a co-evolutionary training framework that jointly optimizes trainable edge features and a query-conditioned **QueryRouter** MLP via policy-gradient reinforcement learning.

<table>
<tr>
<td width="33%" align="center">

### 🕸️ Weighted Multi-Relational Graph
Each edge carries a **trainable feature vector**, initialized by LLM-based scoring (Phase 1) and refined by RL (Phase 2).

</td>
<td width="33%" align="center">

### 🧭 QueryRouter
A 3-layer MLP that computes **query-conditioned traversal scores** without path-level supervision.

</td>
<td width="33%" align="center">

### 🔄 Co-evolutionary Training
**REINFORCE** jointly optimizes edge features and the router on evidence-hit rewards with **zero extra LLM calls** during training.

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 📖 LoCoMo (Long Conversation Memory)

```bash
# Test on LoCoMo dataset (10 samples included)
python test_fixed_memory.py --sample 0 --model gpt-4o-mini --max-questions 10 --category-to-test 1,2,3,4,5

# Test specific question categories
python test_fixed_memory.py --sample 0 --category-to-test 1   # Multi-hop only

# Test multiple samples
python test_fixed_memory.py --sample 0 1 2 --max-questions 50
```

### 🔍 HotpotQA (Multi-hop QA)

```bash
# Test on HotpotQA multi-hop questions
python test_hotpotqa.py --dataset data/hotpot_dev_distractor_v1.json --num-questions 100 --model gpt-4o-mini

# Parallel evaluation
python test_hotpotqa.py --num-questions 100 --parallel 10 --best-of-n 3
```

### 🎯 RL Training (Phase 2)

```bash
# Train QueryRouter and edge features via REINFORCE
python train_rl_edge_scorer.py --dataset data/locomo10.json --epochs 100 --lr 1e-3
```

### 📊 5-Fold Cross-Validation

```bash
python scripts/run_5fold_cv.py --dataset data/locomo10.json --model gpt-4o-mini
```

---

## 📦 Installation

**Prerequisites:** Python 3.9+, a virtual environment (recommended), and an OpenAI-compatible API key.

```bash
# 1. Clone
git clone <repository-url>
cd HAGE

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## 📚 Datasets

| Dataset | Path | Description | Status |
|:--------|:-----|:------------|:------:|
| **LoCoMo** | `data/locomo10.json` | 10 long conversations with Q&A across 5 categories (Multi-hop, Temporal, Open-domain, Single-hop, Adversarial). | ✅ Included (2.7 MB) |
| **HotpotQA** | `data/hotpot_dev_distractor_v1.json` | Multi-hop QA over 10 documents per question with distractors (7,405 dev questions). | ⬇️ Download |

<details>
<summary><b>Download HotpotQA</b></summary>

```bash
mkdir -p data/ && cd data/
wget https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_dev_distractor_v1.json
cd ..
```

</details>

---

## ⚙️ Configuration

### Embedding Models

| Model | Dimensions | Notes |
|:------|:----------:|:------|
| **MiniLM** (default) | 384 | Fast, offline |
| **OpenAI** | 1536 | Higher quality, requires API key |

```bash
python test_fixed_memory.py --embedding-model minilm    # default
python test_fixed_memory.py --embedding-model openai
```

### Supported LLM Backends

- `gpt-4o-mini` *(default)*
- `gpt-4.1-mini`
- `gpt-4o`
- Local models via OpenAI-compatible API — see `.env.example`

---

## 📁 Project Structure

<details>
<summary><b>Click to expand</b></summary>

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
│   ├── hotpot_dev_distractor_v1.json   # HotpotQA (download required)
│   └── README.md                 # Dataset download instructions
├── examples/                     # Sample data for quick testing
│   └── locomo_sample.json
├── test_fixed_memory.py          # LoCoMo evaluation script
├── test_hotpotqa.py              # HotpotQA evaluation script
├── train_rl_edge_scorer.py       # RL training entry point
├── requirements.txt
└── .env.example
```

</details>

---

## 📈 Evaluation

| Metric | Range | What it measures |
|:-------|:-----:|:-----------------|
| **Exact Match** | 0 / 1 | Binary correctness |
| **F1 Score** | 0–100% | Token-level overlap |
| **BLEU** | 0–100% | N-gram similarity |
| **LLM Judge** | 0–100% | GPT-based semantic evaluation |

---

## 📝 Citation

If you find HAGE useful in your research, please consider citing:

```bibtex
@misc{jiang2026hageharnessingagenticmemory,
      title={HAGE: Harnessing Agentic Memory via RL-Driven Weighted Graph Evolution},
      author={Dongming Jiang and Yi Li and Guanpeng Li and Qiannan Li and Bingzhe Li},
      year={2026},
      eprint={2605.09942},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.09942},
}
```

---

## 📄 License

Released under the [MIT License](./LICENSE).
