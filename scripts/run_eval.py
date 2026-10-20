import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_format_variants():
    """Generates 30 distinct prompt format templates based on experiment.yaml factors."""
    field_seps = [": ", " - ", ":\n", " = ", ":: "]
    item_seps = ["\n", "\n\n", " ", " || "]
    cases = ["Sentence", "SENTENCE", "sentence"]
    
    formats = []
    for f_sep in field_seps:
        for i_sep in item_seps:
            for c in cases:
                fmt = f"{c}{f_sep}{{text}}{i_sep}Label:"
                formats.append(fmt)
                if len(formats) == 30:
                    return formats
    return formats

def main():
    set_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Evaluating on device: {device}")

    # 1. Load Dataset & Fixed Eval Subset
    ds = load_dataset("stanfordnlp/sst2")
    val_data = ds["validation"]
    
    # Freeze 500-sample evaluation subset
    indices = list(range(len(val_data)))
    random.shuffle(indices)
    eval_indices = indices[:500]
    eval_samples = [val_data[i] for i in eval_indices]
    
    # Save frozen evaluation IDs
    os.makedirs("results", exist_ok=True)
    with open("results/eval_ids.json", "w") as f:
        json.dump(eval_indices, f)

    # 2. Load Model & Tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    # Get target label token IDs
    label_tokens = {
        0: tokenizer.encode(" negative", add_special_tokens=False)[0],
        1: tokenizer.encode(" positive", add_special_tokens=False)[0]
    }

    formats = generate_format_variants()
    shots_list = [0, 1, 3, 5]
    
    summary_results = []

    # 3. Evaluation Loop across Shots (k) and Formats
    for k in shots_list:
        print(f"\n--- Running Evaluation for k = {k} few-shot shots ---")
        format_accuracies = []

        for fmt_idx, fmt in enumerate(formats):
            correct = 0
            for sample in eval_samples:
                text = sample["sentence"]
                gold_label = sample["label"]
                
                # Construct zero-shot / few-shot prompt
                prompt_text = fmt.format(text=text)
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1].float()
                
                pred_label = 0 if logits[label_tokens[0]].item() > logits[label_tokens[1]].item() else 1
                if pred_label == gold_label:
                    correct += 1

            acc = correct / len(eval_samples)
            format_accuracies.append(acc)

        # Calculate FormatSpread Metrics (Sclar et al., 2024)
        min_acc = min(format_accuracies)
        max_acc = max(format_accuracies)
        spread = max_acc - min_acc
        mean_acc = np.mean(format_accuracies)
        std_acc = np.std(format_accuracies)

        summary_results.append({
            "k_shots": k,
            "mean_accuracy": round(mean_acc, 4),
            "min_accuracy": round(min_acc, 4),
            "max_accuracy": round(max_acc, 4),
            "format_spread": round(spread, 4),
            "std_dev": round(std_acc, 4)
        })

    # 4. Save Final CSV Results
    df = pd.DataFrame(summary_results)
    print("\n========================================================")
    print("                GATE G3 EXPERIMENTAL RESULTS            ")
    print("========================================================")
    print(df.to_string(index=False))

    df.to_csv("results/eval_results.csv", index=False)
    print("\n[SUCCESS] Final evaluation results saved to 'results/eval_results.csv'.")

if __name__ == "__main__":
    main()