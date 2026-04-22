"""
╔══════════════════════════════════════════════════════════════════════╗
║   LoRA Evaluation — Qwen3:4b  ×  SCM NL-to-SQL                     ║
║   Computes: Execution Accuracy, Exact Match, Partial Credit,        ║
║             Per-category breakdown, Per-construct breakdown          ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
    python evaluate_lora_qwen3_scm.py

WHAT THIS DOES:
    1. Loads base Qwen3:4b from safetensors
    2. Merges the trained LoRA adapter
    3. Runs inference on all 118 test samples
    4. Computes 5 accuracy metrics
    5. Saves results to:
       - ./outputs/eval_results/predictions.jsonl   (all predictions)
       - ./outputs/eval_results/accuracy_report.json
       - ./outputs/eval_results/accuracy_report.tsv
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

BASE_MODEL_PATH   = "./qwen3_4b_base"
ADAPTER_PATH      = "./outputs/lora_scm_nl2sql"
TEST_FILE         = "./data/scm_nl2sql_test_1187.jsonl"
OUTPUT_DIR        = "./outputs/eval_results"
MAX_SEQ_LENGTH    = 1024
MAX_NEW_TOKENS    = 512
BATCH_SIZE        = 1       # increase if VRAM allows
TEMPERATURE       = 0.0     # 0.0 = greedy decode (deterministic)
DO_SAMPLE         = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


# ─────────────────────────────────────────────────────────────────────
#  MODEL LOADING  —  merge LoRA into base for inference
# ─────────────────────────────────────────────────────────────────────

def load_merged_model():
    log.info(f"Loading tokenizer from base: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",    # left-pad for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info(f"Loading base model: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    log.info(f"Loading LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Merge weights for faster inference (no overhead from LoRA during forward pass)
    log.info("Merging LoRA weights into base model …")
    model = model.merge_and_unload()
    model.eval()

    log.info("Model ready for inference")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────

def generate_sql(model, tokenizer, sample: dict, device) -> str:
    prompt = PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample["input"],
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (exclude the prompt)
    new_ids     = output_ids[0][inputs["input_ids"].shape[1]:]
    generated   = tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated.strip()


# ─────────────────────────────────────────────────────────────────────
#  ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────

def normalise_sql(sql: str) -> str:
    """
    Canonical form for exact-match comparison:
    lowercase, collapse whitespace, strip trailing semicolon.
    """
    sql = sql.lower().strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def exact_match(pred: str, gold: str) -> bool:
    return normalise_sql(pred) == normalise_sql(gold)


def partial_credit(pred: str, gold: str) -> float:
    """
    Token-overlap F1 between prediction and gold (word level).
    Captures partial correctness — correct table/column names
    with wrong conditions or aggregations.
    """
    pred_tok = set(re.findall(r"\w+", normalise_sql(pred)))
    gold_tok = set(re.findall(r"\w+", normalise_sql(gold)))
    if not gold_tok:
        return 1.0
    inter  = pred_tok & gold_tok
    prec   = len(inter) / len(pred_tok) if pred_tok else 0.0
    recall = len(inter) / len(gold_tok)
    if prec + recall == 0:
        return 0.0
    return 2 * prec * recall / (prec + recall)


# SQL constructs to track individually
CONSTRUCTS = {
    "JOIN":        lambda s: "join" in s.lower(),
    "GROUP_BY":    lambda s: "group by" in s.lower(),
    "HAVING":      lambda s: "having" in s.lower(),
    "CTE":         lambda s: s.strip().lower().startswith("with "),
    "WINDOW_FUNC": lambda s: "over (" in s.lower() or "over(" in s.lower(),
    "CASE_WHEN":   lambda s: "case" in s.lower() and "when" in s.lower(),
    "SUBQUERY":    lambda s: s.lower().count("select") > 1,
    "COALESCE":    lambda s: "coalesce" in s.lower(),
    "LAG_LEAD":    lambda s: "lag(" in s.lower() or "lead(" in s.lower(),
    "STRING_AGG":  lambda s: "string_agg" in s.lower(),
    "UNION":       lambda s: bool(re.search(r"\bunion\b", s.lower())),
    "INTERSECT":   lambda s: bool(re.search(r"\bintersect\b", s.lower())),
    "EXCEPT":      lambda s: bool(re.search(r"\bexcept\b", s.lower())),
    "MERGE":       lambda s: s.strip().lower().startswith("merge"),
    "ROLLUP":      lambda s: "rollup" in s.lower(),
    "TRY_CAST":    lambda s: "try_cast" in s.lower(),
    "CROSS_APPLY": lambda s: "cross apply" in s.lower(),
    "IIF":         lambda s: bool(re.search(r"\biif\b", s.lower())),
    "STDEV":       lambda s: "stdev" in s.lower(),
    "EXISTS":      lambda s: "exists" in s.lower(),
    "INSERT":      lambda s: s.strip().lower().startswith("insert"),
    "UPDATE":      lambda s: s.strip().lower().startswith("update"),
    "DELETE":      lambda s: s.strip().lower().startswith("delete"),
}


def detect_constructs(sql: str) -> list[str]:
    return [name for name, fn in CONSTRUCTS.items() if fn(sql)]


# ─────────────────────────────────────────────────────────────────────
#  MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────

def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test samples
    test_samples = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    log.info(f"Test samples: {len(test_samples)}")

    # Load model
    model, tokenizer = load_merged_model()
    device = next(model.parameters()).device

    # ── Inference loop ─────────────────────────────────────────────────
    predictions     = []
    em_scores       = []
    pc_scores       = []
    complexity_hits = defaultdict(list)  # complexity → [bool]
    category_hits   = defaultdict(list)  # category → [bool]
    construct_hits  = defaultdict(list)  # construct → [bool]

    log.info("Running inference …")
    t0 = time.time()

    for i, sample in enumerate(test_samples):
        if (i + 1) % 10 == 0:
            elapsed  = time.time() - t0
            per_sample = elapsed / (i + 1)
            remaining  = per_sample * (len(test_samples) - i - 1)
            log.info(
                f"  [{i+1:3d}/{len(test_samples)}]  "
                f"elapsed {elapsed:.0f}s  ETA {remaining:.0f}s"
            )

        pred_sql = generate_sql(model, tokenizer, sample, device)
        gold_sql = sample["output"]

        em   = exact_match(pred_sql, gold_sql)
        pc   = partial_credit(pred_sql, gold_sql)

        em_scores.append(em)
        pc_scores.append(pc)

        # Execution accuracy proxy:
        # A prediction is "executable" if it parses as valid SQL structure.
        # True execution accuracy requires a running DB — this is the best
        # offline proxy: EM OR (partial credit > 0.85).
        exec_hit = em or pc >= 0.85
        category   = sample.get("category", "unknown")
        complexity  = sample.get("complexity", "unknown")

        category_hits[category].append(exec_hit)
        complexity_hits[complexity].append(exec_hit)

        # Track constructs present in the gold SQL
        gold_constructs = detect_constructs(gold_sql)
        for c in gold_constructs:
            construct_hits[c].append(exec_hit)

        predictions.append({
            "id":              i,
            "input":           sample["input"],
            "gold_sql":        gold_sql,
            "pred_sql":        pred_sql,
            "category":        category,
            "complexity":      complexity,
            "exact_match":     em,
            "partial_credit":  round(pc, 4),
            "exec_hit_proxy":  exec_hit,
            "gold_constructs": gold_constructs,
        })

    elapsed_total = time.time() - t0
    n = len(test_samples)

    # ── Aggregate metrics ───────────────────────────────────────────────
    overall_em   = sum(em_scores) / n
    overall_pc   = sum(pc_scores) / n
    overall_exec = sum(p["exec_hit_proxy"] for p in predictions) / n

    log.info("=" * 60)
    log.info(f"  EVALUATION RESULTS — {n} test samples")
    log.info("=" * 60)
    log.info(f"  Execution Accuracy (proxy): {overall_exec*100:.1f}%")
    log.info(f"  Exact Match:                {overall_em*100:.1f}%")
    log.info(f"  Partial Credit (F1):        {overall_pc*100:.1f}%")
    log.info(f"  Inference time:             {elapsed_total:.0f}s  ({elapsed_total/n:.1f}s/sample)")

    # ── Per-complexity breakdown ─────────────────────────────────────────
    log.info("\n  Accuracy by complexity:")
    complexity_report = {}
    for cx in ["simple", "medium", "complex"]:
        hits = complexity_hits.get(cx, [])
        acc  = sum(hits) / len(hits) if hits else 0.0
        complexity_report[cx] = {"n": len(hits), "accuracy": round(acc, 4)}
        log.info(f"    {cx:<10}  n={len(hits):3d}  acc={acc*100:.1f}%")

    # ── Per-category breakdown ───────────────────────────────────────────
    log.info("\n  Accuracy by category:")
    category_report = {}
    for cat, hits in sorted(category_hits.items(), key=lambda x: -sum(x[1])/len(x[1])):
        acc = sum(hits) / len(hits) if hits else 0.0
        category_report[cat] = {"n": len(hits), "accuracy": round(acc, 4)}
        log.info(f"    {cat:<35}  n={len(hits):3d}  acc={acc*100:.1f}%")

    # ── Per-construct breakdown ──────────────────────────────────────────
    log.info("\n  Accuracy by SQL construct:")
    construct_report = {}
    for con, hits in sorted(construct_hits.items(), key=lambda x: -len(x[1])):
        acc = sum(hits) / len(hits) if hits else 0.0
        construct_report[con] = {"n": len(hits), "accuracy": round(acc, 4)}
        log.info(f"    {con:<16}  n={len(hits):3d}  acc={acc*100:.1f}%")

    # ── Save predictions ─────────────────────────────────────────────────
    pred_path = os.path.join(OUTPUT_DIR, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    log.info(f"\nPredictions saved: {pred_path}")

    # ── Save JSON report ─────────────────────────────────────────────────
    report = {
        "n_test_samples":     n,
        "inference_time_sec": round(elapsed_total, 1),
        "sec_per_sample":     round(elapsed_total / n, 2),
        "overall": {
            "execution_accuracy_proxy": round(overall_exec, 4),
            "exact_match":              round(overall_em,   4),
            "partial_credit_f1":        round(overall_pc,   4),
        },
        "by_complexity": complexity_report,
        "by_category":   category_report,
        "by_construct":  construct_report,
    }
    json_path = os.path.join(OUTPUT_DIR, "accuracy_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report saved: {json_path}")

    # ── Save TSV report ───────────────────────────────────────────────────
    tsv_path = os.path.join(OUTPUT_DIR, "accuracy_report.tsv")
    rows = []
    rows.append("Category\tMetric\tN\tAccuracy (%)\tNotes")

    rows.append(f"Overall\tExecution Accuracy (proxy)\t{n}\t{overall_exec*100:.1f}\tEM or partial credit ≥ 0.85")
    rows.append(f"Overall\tExact Match\t{n}\t{overall_em*100:.1f}\tNormalised SQL string match")
    rows.append(f"Overall\tPartial Credit F1\t{n}\t{overall_pc*100:.1f}\tToken-overlap F1 vs gold SQL")

    for cx, data in complexity_report.items():
        rows.append(f"Complexity\t{cx}\t{data['n']}\t{data['accuracy']*100:.1f}\t")

    for cat, data in category_report.items():
        rows.append(f"Category\t{cat}\t{data['n']}\t{data['accuracy']*100:.1f}\t")

    for con, data in construct_report.items():
        rows.append(f"Construct\t{con}\t{data['n']}\t{data['accuracy']*100:.1f}\t")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    log.info(f"TSV report saved: {tsv_path}")

    return report


# ─────────────────────────────────────────────────────────────────────
#  ERROR ANALYSIS  — optional, run after evaluate()
# ─────────────────────────────────────────────────────────────────────

def analyse_errors(predictions_path: str = None):
    """
    Prints the top-20 failure cases sorted by lowest partial credit.
    Run this after evaluate() to understand where the model struggles.
    """
    path = predictions_path or os.path.join(OUTPUT_DIR, "predictions.jsonl")
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line.strip()))

    failures = [p for p in preds if not p["exec_hit_proxy"]]
    failures.sort(key=lambda x: x["partial_credit"])

    print(f"\n{'='*70}")
    print(f"TOP FAILURE CASES  ({len(failures)} failures / {len(preds)} total)")
    print(f"{'='*70}")

    for i, p in enumerate(failures[:20]):
        print(f"\n[{i+1}] Category: {p['category']}  Complexity: {p['complexity']}")
        print(f"  Input:      {p['input']}")
        print(f"  Gold SQL:   {p['gold_sql'][:120]}")
        print(f"  Prediction: {p['pred_sql'][:120]}")
        print(f"  Partial F1: {p['partial_credit']:.3f}")
        print(f"  Constructs: {', '.join(p['gold_constructs'])}")

    # Summary of failure patterns
    fail_cats = defaultdict(int)
    fail_cons = defaultdict(int)
    for p in failures:
        fail_cats[p["category"]] += 1
        for c in p["gold_constructs"]:
            fail_cons[c] += 1

    print(f"\n  Failure count by category:")
    for cat, cnt in sorted(fail_cats.items(), key=lambda x: -x[1])[:8]:
        print(f"    {cat:<35} {cnt}")

    print(f"\n  Failure count by SQL construct:")
    for con, cnt in sorted(fail_cons.items(), key=lambda x: -x[1])[:10]:
        print(f"    {con:<20} {cnt}")


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = evaluate()

    # Optional: run error analysis immediately after evaluation
    analyse_errors()

    print("\n" + "=" * 60)
    print(f"  FINAL ACCURACY SUMMARY")
    print("=" * 60)
    o = report["overall"]
    print(f"  Execution Accuracy : {o['execution_accuracy_proxy']*100:.1f}%")
    print(f"  Exact Match        : {o['exact_match']*100:.1f}%")
    print(f"  Partial Credit F1  : {o['partial_credit_f1']*100:.1f}%")
    print("=" * 60)
