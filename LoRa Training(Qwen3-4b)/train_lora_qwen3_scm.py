import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — edit these paths and hyperparameters
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────
    model_path: str        = "./qwen3_4b_base"          # local safetensors folder
    train_file: str        = "./data/scm_nl2sql_train_1187.jsonl"
    val_file: str          = "./data/scm_nl2sql_val_1187.jsonl"
    output_dir: str        = "./outputs/lora_scm_nl2sql"

    # ── LoRA hyperparameters ──────────────────────────────────────────
    lora_r: int            = 16          # rank — 8 is lighter, 32 is stronger
    lora_alpha: int        = 32          # scaling = alpha / r  → keep 2×r
    lora_dropout: float    = 0.05
    # which linear layers to inject LoRA into (all attention + FFN)
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # ── Training hyperparameters ───────────────────────────────────────
    num_train_epochs: int  = 5
    per_device_train_batch_size: int = 2    # increase to 4 if VRAM allows
    per_device_eval_batch_size: int  = 2
    gradient_accumulation_steps: int = 8   # effective batch = 2 × 8 = 16
    learning_rate: float   = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float    = 0.10
    weight_decay: float    = 0.01
    max_grad_norm: float   = 1.0

    # ── Sequence length ────────────────────────────────────────────────
    max_seq_length: int    = 1024         # covers all samples incl. complex CTEs

    # ── Precision ─────────────────────────────────────────────────────
    # fp16 for older GPUs (RTX 30xx), bf16 for A100/H100/RTX 40xx
    fp16: bool             = True
    bf16: bool             = False        # set True + fp16=False for A100/H100

    # ── Checkpointing ─────────────────────────────────────────────────
    save_strategy: str     = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    save_total_limit: int  = 3

    # ── Logging ───────────────────────────────────────────────────────
    logging_steps: int     = 20
    report_to: str         = "none"       # change to "wandb" to enable tracking
    seed: int              = 42


CFG = Config()

# ─────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
#  ALPACA PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def build_prompt(sample: dict) -> str:
    """Full prompt including the SQL output (for training)."""
    prompt = PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample["input"],
    )
    return prompt + sample["output"]


def build_inference_prompt(sample: dict) -> str:
    """Prompt without the output (for inference/evaluation)."""
    return PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample["input"],
    )


# ─────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    log.info(f"Loaded {len(samples)} samples from {path}")
    return samples


def tokenize_dataset(samples: list[dict], tokenizer, max_length: int) -> Dataset:
    """
    Tokenise each sample as a full prompt+response string.
    Labels are set to -100 for the prompt tokens so loss is computed
    only on the SQL response tokens — standard Causal-LM SFT setup.
    """
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for s in samples:
        full_text   = build_prompt(s)
        prompt_text = build_inference_prompt(s)

        full_enc   = tokenizer(full_text,   truncation=True, max_length=max_length)
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length)

        prompt_len = len(prompt_enc["input_ids"])
        input_ids  = full_enc["input_ids"]
        attn_mask  = full_enc["attention_mask"]

        # Mask prompt tokens from loss
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Pad/truncate to max_length
        pad_id   = tokenizer.pad_token_id or tokenizer.eos_token_id
        pad_len  = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_id] * pad_len
            attn_mask = attn_mask + [0]      * pad_len
            labels    = labels    + [-100]   * pad_len
        else:
            input_ids = input_ids[:max_length]
            attn_mask = attn_mask[:max_length]
            labels    = labels[:max_length]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attn_mask)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids":      input_ids_list,
        "attention_mask": attention_mask_list,
        "labels":         labels_list,
    })


# ─────────────────────────────────────────────────────────────────────
#  MODEL + TOKENIZER LOADING  (from local safetensors)
# ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
    log.info(f"Loading tokenizer from: {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        padding_side="right",   # right-padding for causal LM
    )

    # Qwen3 may not have a pad token — set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("pad_token set to eos_token")

    log.info(f"Loading base model from: {cfg.model_path}")
    log.info("This may take 2–5 minutes for a 4B model …")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16 if cfg.fp16 else torch.bfloat16,
        device_map="auto",          # spreads across available GPUs automatically
        trust_remote_code=True,
        use_safetensors=True,       # explicitly use safetensors format
        low_cpu_mem_usage=True,     # stream weights to save RAM during loading
    )

    log.info(f"Model loaded — parameters: {model.num_parameters():,}")
    log.info(f"Device map: {model.hf_device_map}")

    # Disable KV-cache during training (not needed, wastes memory)
    model.config.use_cache = False

    # Gradient checkpointing saves VRAM at the cost of ~20% slower backward pass
    model.enable_input_require_grads()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
#  LoRA CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

def apply_lora(model, cfg: Config):
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = cfg.lora_r,
        lora_alpha     = cfg.lora_alpha,
        lora_dropout   = cfg.lora_dropout,
        target_modules = cfg.lora_target_modules,
        bias           = "none",
        inference_mode = False,
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"LoRA applied — trainable params: {trainable:,} / {total:,} "
             f"({100 * trainable / total:.2f}%)")

    return model


# ─────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # 2. Load and tokenise datasets
    log.info("Tokenising training data …")
    train_samples = load_jsonl(cfg.train_file)
    val_samples   = load_jsonl(cfg.val_file)

    train_dataset = tokenize_dataset(train_samples, tokenizer, cfg.max_seq_length)
    val_dataset   = tokenize_dataset(val_samples,   tokenizer, cfg.max_seq_length)
    log.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir                  = cfg.output_dir,
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        per_device_eval_batch_size  = cfg.per_device_eval_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        warmup_ratio                = cfg.warmup_ratio,
        weight_decay                = cfg.weight_decay,
        max_grad_norm               = cfg.max_grad_norm,
        fp16                        = cfg.fp16,
        bf16                        = cfg.bf16,
        save_strategy               = cfg.save_strategy,
        evaluation_strategy         = cfg.evaluation_strategy,
        load_best_model_at_end      = cfg.load_best_model_at_end,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = cfg.save_total_limit,
        logging_steps               = cfg.logging_steps,
        report_to                   = cfg.report_to,
        seed                        = cfg.seed,
        dataloader_num_workers      = 0,
        remove_unused_columns       = False,
        gradient_checkpointing      = True,
    )

    # 4. Data collator — pads batches dynamically
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,       # slight CUDA efficiency boost
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = data_collator,
        tokenizer       = tokenizer,
    )

    # 6. Train
    log.info("=" * 60)
    log.info("Starting LoRA fine-tuning …")
    log.info(f"  Epochs              : {cfg.num_train_epochs}")
    log.info(f"  Effective batch size: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    log.info(f"  Learning rate       : {cfg.learning_rate}")
    log.info(f"  Max seq length      : {cfg.max_seq_length}")
    log.info(f"  LoRA rank           : {cfg.lora_r}  alpha: {cfg.lora_alpha}")
    log.info("=" * 60)

    train_result = trainer.train()

    # 7. Save adapter (only LoRA weights, not the full base model)
    log.info("Saving LoRA adapter …")
    model.save_pretrained(cfg.output_dir)          # saves adapter_model.safetensors
    tokenizer.save_pretrained(cfg.output_dir)

    # 8. Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["val_samples"]   = len(val_dataset)

    with open(os.path.join(cfg.output_dir, "training_log.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Training complete!")
    log.info(f"Adapter saved to: {cfg.output_dir}")
    log.info(f"Train loss: {metrics.get('train_loss', 'N/A'):.4f}")

    return trainer, metrics


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # GPU sanity check
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU detected — training on CPU will be extremely slow!")
    else:
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / 1e9
            log.info(f"GPU {i}: {props.name}  |  VRAM: {vram:.1f} GB")

        total_vram = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(n_gpu)
        ) / 1e9
        log.info(f"Total VRAM available: {total_vram:.1f} GB")

        if total_vram < 14:
            log.warning(
                "Less than 14 GB VRAM detected. "
                "Reduce per_device_train_batch_size to 1 in Config."
            )

    trainer, metrics = train(CFG)
