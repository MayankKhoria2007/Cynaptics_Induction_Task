# 🧠 GPT From Scratch – Tokenization & Architecture Experiments

A clean, from-scratch implementation of a GPT-style language model, exploring different tokenization strategies and attention mechanism improvements using the Tiny Shakespeare dataset.

---

## 🚀 Project Highlights

- 🔤 Built a GPT model from scratch
- ⚙️ Implemented two tokenization approaches
- 🧩 Designed a custom tokenizer pipeline
- 📈 Introduced an attention architecture enhancement
- 📚 Trained on Tiny Shakespeare dataset

---

## 📂 Repository Structure

.
├── gpt_using_tiktoken.py
├── gpt_using_own_tokenizer.py
├── tokenizer.py
├── tokenizer.json

---

## 🧩 Implementations

### 🔹 1. GPT with tiktoken Tokenizer

- Uses a prebuilt tokenizer
- Vocabulary size: 50,000

#### 🔧 Architectural Enhancement

Key (K) and Query (Q) vector dimensions are scaled to 8× the head size.

#### 💡 Motivation

- Improves expressiveness of attention
- Captures richer token relationships
- Enables higher-dimensional attention space

---

### 🔹 2. GPT with Custom Tokenizer

- tokenizer.py → trains tokenizer
- tokenizer.json → saved tokenizer
- Vocabulary size: 12,000

---

## ⚙️ Setup & Execution

### Requirements

- Python 3.8+
- PyTorch
- tiktoken

Install:

pip install torch tiktoken

---

### ▶️ Run (tiktoken version)

python gpt_using_tiktoken.py

---

### ▶️ Run (custom tokenizer)

Step 1: Ensure tokenizer.json exists

python tokenizer.py

Step 2: Run

python gpt_using_own_tokenizer.py

---

## 🏗️ Model Architecture

- Token Embeddings
- Positional Embeddings
- Transformer Blocks:
  - Multi-Head Attention
  - Feed Forward Network
- LayerNorm
- Output Layer

---

## 🔍 Attention Change

Query/Key dimension = 8 × head size

---

## 🔤 Tokenization Comparison

tiktoken (50k vocab) vs Custom (12k vocab)

---

## 📈 Key Insights

- Tokenization affects performance and efficiency
- Larger Q/K dimension improves attention quality
- Custom tokenizer improves domain adaptation

---

## 🛠️ Future Work

- Add evaluation metrics
- Improve sampling
- Optimize memory
- Add inference UI
