# Dataset Information

This directory contains the evaluation datasets for the HAGE system.

## Datasets

### 1. LoCoMo (Long Conversation Memory) — `locomo10.json`
- **Status**: Included in repository (2.7MB)
- **Description**: Long-horizon conversational memory dataset with 10 conversation samples
- **Format**: JSON with conversation turns and Q&A pairs
- **Categories**: Multi-hop, Temporal, Open-domain, Single-hop, Adversarial
- **Size**: 10 samples, ~1000 Q&A pairs

### 2. HotpotQA — `hotpot_dev_distractor_v1.json`
- **Status**: Download required (45MB)
- **Description**: Multi-hop QA benchmark; each question requires reasoning over 10 documents including distractors
- **Format**: JSON with questions, supporting facts, and distractor paragraphs
- **Size**: 7,405 dev questions

#### Download Instructions:
```bash
mkdir -p data/
cd data/
wget https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_dev_distractor_v1.json
cd ..
```

## Dataset Formats

### LoCoMo Format
```json
{
  "samples": [
    {
      "conversation": [
        {"speaker": "User", "text": "..."},
        {"speaker": "Assistant", "text": "..."}
      ],
      "questions": [
        {
          "question": "What did the user mention?",
          "answer": "Expected answer",
          "category": 1
        }
      ]
    }
  ]
}
```

### HotpotQA Format
```json
{
  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "supporting_facts": [["Arthur's Magazine", 0], ["First for Women", 0]],
  "context": [
    ["Arthur's Magazine", ["Arthur's Magazine ...", "..."]],
    ["First for Women", ["First for Women ...", "..."]]
  ],
  "type": "comparison",
  "level": "medium"
}
```

## Sample Data

See the `examples/` directory for a small LoCoMo sample (`locomo_sample.json`) to test the system without downloading the full datasets.
