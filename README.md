SCM-Query: Domain-Specific NL-to-SQL Generation using LoRA

SCM-Query is a specialized Natural Language Interface (NLI) designed to bridge the gap between business instructions and relational databases in the **Supply Chain Management (SCM)** domain. By fine-tuning a Large Language Model using Parameter-Efficient Fine-Tuning (PEFT), this system accurately translates complex SCM queries into executable SQL.

## 🚀 Key Features
- **Domain-Specific Logic:** Understands SCM-specific schema relationships including inventory, procurement, and logistics.
- **LoRA Optimization:** Utilizes Low-Rank Adaptation (LoRA) to adapt the Qwen3:4b model with minimal computational overhead.
- **Advanced SQL Support:** Robust generation of complex constructs including **CTEs, multi-table JOINs, and Window Functions**.
- **Rigorous Evaluation:** A comprehensive benchmarking suite measuring Exact Match (EM) and Partial Credit (F1) scores across simple, medium, and complex query categories.

## 🛠️ Tech Stack
- **Language:** Python
- **Model:** Qwen3:4b (Base)
- **Frameworks:** Hugging Face (Transformers, PEFT, TRL)
- **Optimization:** BitsAndBytes (4-bit/8-bit quantization), LoRA
- **Evaluation:** Scikit-learn, Regex-based SQL construct tracking
- **Data:** JSONL-formatted SCM instruction datasets

## 📊 Technical Workflow

### 1. Fine-Tuning (PEFT)
The model is trained using **Low-Rank Adaptation (LoRA)** to modify a small subset of the model's weights. This allows for high-performance domain adaptation while maintaining a low memory footprint.
- **Precision:** FP16/BF16
- **Strategy:** Instruction-based supervised fine-tuning (SFT)

### 2. Inference Pipeline

- **Weight Merging:** Adapters are merged with the base model (`merge_and_unload`) for low-latency inference.
- **Deterministic Decoding:** Temperature is set to 0.0 to ensure reproducible and reliable SQL generation.

### 3. Evaluation Suite
The system produces detailed reports on model performance:
- **Exact Match (EM):** Strict normalization and comparison against "Gold" SQL.
- **Partial Credit (F1):** Measures token overlap to reward partially correct queries.
- **Construct Analysis:** Tracks accuracy specifically for `JOIN`, `WHERE`, `GROUP BY`, and `WINDOW_FUNC`.

## 📁 Project Structure
```text
├── train_lora.py          # Script for PEFT fine-tuning
├── evaluate_model.py      # Multi-metric evaluation suite
├── data/
│   ├── scm_train.jsonl    # Training instructions
│   └── scm_test.jsonl     # Benchmarking dataset
├── results/
│   ├── predictions.jsonl  # Log of model outputs
│   └── accuracy_report.tsv# Detailed performance breakdown
└── README.md
