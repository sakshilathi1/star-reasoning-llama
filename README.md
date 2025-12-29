# 🌟 STaR: Self-Taught Reasoner with Llama-3.2-3B

<div align="center">

![Model](https://img.shields.io/badge/Model-Llama--3.2--3B--Instruct-purple.svg)
![Dataset](https://img.shields.io/badge/Dataset-GSM8K-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

**Implementing Self-Taught Reasoner (STaR) bootstrapping for mathematical reasoning on GSM8K**

[Overview](#-overview) • [Method](#-method) • [Results](#-results) • [Usage](#-usage)

</div>

---

## 📖 Overview

This project implements and compares three reasoning approaches on the **GSM8K** math word problem dataset using **Llama-3.2-3B-Instruct**:

1. **Zero-Shot Chain-of-Thought (CoT)** - Direct prompting with "Let's think step by step"
2. **Vanilla Supervised Fine-Tuning (SFT)** - Training on gold rationales from the dataset
3. **STaR (Self-Taught Reasoner)** - Bootstrapping reasoning by iteratively generating and refining rationales

### Key Concept

STaR iteratively improves reasoning by:
- Generating rationales for training examples
- Retrying with answer hints when incorrect
- Fine-tuning on successfully bootstrapped rationales

## 🧠 Method

### STaR Algorithm

```
For each question in training set:
    1. Generate rationale + answer using current model
    2. If answer is CORRECT → keep the rationale
    3. If answer is WRONG → regenerate with answer hint
    4. Collect all correct rationales → bootstrapped dataset
    5. Fine-tune model on bootstrapped dataset
    6. Repeat for multiple iterations
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STaR Training Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    Generate     ┌──────────────┐                 │
│   │ Question │───────────────▶│   Rationale   │                 │
│   └──────────┘                 │   + Answer    │                 │
│                                └───────┬───────┘                 │
│                                        │                         │
│                               Check Answer                       │
│                                        │                         │
│                    ┌───────────────────┴───────────────────┐    │
│                    ▼                                       ▼    │
│              ┌──────────┐                           ┌──────────┐│
│              │ Correct! │                           │  Wrong!  ││
│              │  Keep    │                           │  Retry   ││
│              └────┬─────┘                           └────┬─────┘│
│                   │                                      │      │
│                   │                    ┌─────────────────┘      │
│                   │                    ▼                        │
│                   │         ┌─────────────────────┐             │
│                   │         │ Regenerate with     │             │
│                   │         │ Answer Hint         │             │
│                   │         └──────────┬──────────┘             │
│                   │                    │                        │
│                   ▼                    ▼                        │
│              ┌─────────────────────────────┐                    │
│              │   Bootstrapped Dataset      │                    │
│              │   (Correct Rationales)      │                    │
│              └──────────────┬──────────────┘                    │
│                             │                                   │
│                             ▼                                   │
│              ┌─────────────────────────────┐                    │
│              │   Fine-tune with LoRA/QLoRA │                    │
│              └─────────────────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Results

### Performance Comparison (Exact Match Accuracy)

| Method | Accuracy |
|--------|----------|
| **Zero-Shot CoT** | **34.00%** |
| Vanilla SFT | 30.00% |
| STaR (Iteration 0) | 30.00% |

### Analysis

**When STaR Helps:**
- Answer-conditioned correction filters spurious CoT paths
- Error-driven curriculum focuses on harder examples
- Produces short, correct rationales that transfer well

**When STaR Hurts:**
- Template overfitting on small models (3B parameters)
- Noisy or verbose bootstrapped chains
- Decoding sensitivity across methods

## 🔧 Setup

### Prerequisites

- Python ≥ 3.10
- Google Colab with GPU (T4/A100)
- Hugging Face account with access to Llama models

### Installation

```bash
# Clone repository
git clone https://github.com/sakshilathi1/star-reasoning-llama.git
cd star-reasoning-llama

# Install dependencies
pip install -U pip
pip install transformers datasets torch accelerate peft bitsandbytes
pip install huggingface_hub
```

### Authentication

```python
from huggingface_hub import login
login(token="<YOUR_HF_TOKEN>")
```

## 🚀 Usage

### Running on Google Colab (Recommended)

1. Upload `Sakshi_Lathi_STar_Small_Project_1.ipynb` to Google Colab
2. Enable GPU runtime (T4 or A100)
3. Run cells sequentially:

```
Step 1: Environment Setup
Step 2: Load GSM8K Dataset
Step 3: Zero-Shot CoT Evaluation
Step 4: Vanilla SFT Training + Evaluation
Step 5: STaR Bootstrap + Training + Evaluation
```

### Notebook Sections

| Section | Description |
|---------|-------------|
| **Setup** | Imports, model loading, seeds, decoding params |
| **Data Creation** | Load GSM8K via `datasets` library |
| **Zero-Shot CoT** | Test inference with CoT prompting |
| **Vanilla SFT** | Train on gold rationales → evaluate |
| **STaR** | Bootstrap → train → evaluate |

## 📝 Prompts Used

### Zero-Shot CoT (Test Inference)

```
Question: {question}
Let's think step by step.
```

### Rationale Generation - Without Hint

```
Question: {question}
Let's think step by step to solve this problem.
```

### Rationale Generation - With Hint (STaR Train-Time Only)

```
Question: {question}
The answer is {correct_answer}. Let's think step by step to explain how we get this answer.
```

## 📁 Project Structure

```
star-reasoning-llama/
├── Sakshi_Lathi_STar_Small_Project_1.ipynb   # Main notebook with all code
├── README.md                                   # This file
└── requirements.txt                            # Dependencies (optional)
```

## 🔬 Technical Details

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Llama-3.2-3B-Instruct |
| Fine-tuning | LoRA/QLoRA |
| Train Set | GSM8K train (7,473 examples) |
| Test Set | GSM8K test (1,319 examples) |
| Evaluation Metric | Exact Match (EM) |

### Key Implementation Details

- **Answer Extraction**: Regex-based parsing for "Final Answer:" format
- **Normalization**: Strip whitespace, remove commas/units
- **Decoding**: Consistent temperature and max_tokens across methods
- **Training**: Parameter-efficient fine-tuning with LoRA adapters

## 👤 Author

**Sakshi Lathi**  
Arizona State University  
CSE 576: Natural Language Processing

---

<div align="center">

*Implementing self-improvement in language models through bootstrapped reasoning*

</div>
