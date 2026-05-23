#!/usr/bin/env python3
"""
evaluate_fewshot_results.py
Compare all 6 configs across 0/1/2/4/8 shots for all 3 datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score

RESULTS_DIR = Path("results/label_prediction")
N_SPLITS    = 5

CONFIGS = [
    {"label": "Rule+MMed",       "llm": "MMed",    "refiner": "rule"},
    {"label": "Rule+Mistral",    "llm": "Mistral", "refiner": "rule"},
    {"label": "Mistral+Mistral", "llm": "Mistral", "refiner": "mistral"},
    {"label": "Mistral+MMed",    "llm": "MMed",    "refiner": "mistral"},
    {"label": "MMed+MMed",       "llm": "MMed",    "refiner": "mmed"},
    {"label": "MMed+Mistral",    "llm": "Mistral", "refiner": "mmed"},
]

PAPER_BACC = {"PH2": 78.07, "Derm7pt": 78.56, "HAM10000": 76.00}
N_SHOTS    = [0, 1, 2, 4, 8]

def csv_path(cfg, dataset, n_demos, split=None):
    llm     = cfg["llm"]
    refiner = cfg["refiner"]
    suffix  = (f"raw_values_False_gt_concepts_False"
               f"_model_extractor_Explicd_n_demos_{n_demos}_refiner_{refiner}")
    if split is not None:
        name = f"{dataset}_split_{split}_{llm}_diagnostic_report_validation_{suffix}.csv"
    else:
        name = f"{dataset}_{llm}_diagnostic_report_validation_{suffix}.csv"
    return RESULTS_DIR / name

def compute_metrics(path):
    if not Path(path).exists():
        return None
    df     = pd.read_csv(path)
    y_true = df["gt_response"].str.lower().str.strip()
    y_pred = df["llm_response"].str.lower().str.strip()
    bacc   = balanced_accuracy_score(y_true, y_pred) * 100
    sens   = recall_score(y_true, y_pred, pos_label="melanoma", zero_division=0) * 100
    spec   = recall_score(y_true, y_pred, pos_label="nevus",    zero_division=0) * 100
    return {"bacc": bacc, "sens": sens, "spec": spec}

def avg_ph2(cfg, n_demos):
    results = [compute_metrics(csv_path(cfg, "PH2", n_demos, s)) for s in range(N_SPLITS)]
    valid   = [r for r in results if r]
    if not valid:
        return None
    return {k: np.mean([r[k] for r in valid]) for k in ("bacc", "sens", "spec")}

def pct(v):
    return f"{v:.2f}" if v is not None else "  —  "

def delta(v, baseline):
    if v is None:
        return "  —  "
    d = v - baseline
    return f"{'+' if d >= 0 else ''}{d:.2f}"

def main():
    print("\n" + "="*110)
    print("  FEW-SHOT COMPARISON — All 6 Configs x All Datasets")
    print("="*110)

    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        paper = PAPER_BACC[dataset]

        print(f"\n{'='*110}")
        print(f"  DATASET: {dataset}   (paper baseline = {paper:.2f}%)")
        print(f"{'='*110}")

        # Header
        shots_header = "".join(f"  {n:>5}-shot" for n in N_SHOTS)
        print(f"  {'Config':<20}{shots_header}")
        print(f"  {'(BAcc %)':<20}" + "  " + "-"*90)

        for cfg in CONFIGS:
            row = f"  {cfg['label']:<20}"
            best_bacc = 0
            for n in N_SHOTS:
                if dataset == "PH2":
                    m = avg_ph2(cfg, n)
                else:
                    m = compute_metrics(csv_path(cfg, dataset, n))

                if m:
                    row += f"  {pct(m['bacc']):>8}"
                    best_bacc = max(best_bacc, m['bacc'])
                else:
                    row += f"  {'MISSING':>8}"

            row += f"   (best Δ vs paper: {delta(best_bacc, paper) if best_bacc > 0 else 'N/A':>6}%)"
            print(row)

        print(f"  {'Paper baseline':<20}" + f"  {paper:>8.2f}" * len(N_SHOTS))
        print(f"{'='*110}")

    # Also print full detail per config
    print("\n\n" + "="*110)
    print("  DETAILED VIEW — BAcc / Sens / Spec per n-shot")
    print("="*110)

    for dataset in ["PH2", "Derm7pt", "HAM10000"]:
        paper = PAPER_BACC[dataset]
        print(f"\n{'='*110}")
        print(f"  DATASET: {dataset}")
        print(f"{'='*110}")
        print(f"  {'Config':<20}  {'n':>4}  {'BAcc':>7}  {'Sens':>7}  {'Spec':>7}  {'Δ Paper':>8}")
        print(f"  {'-'*60}")

        for cfg in CONFIGS:
            for n in N_SHOTS:
                if dataset == "PH2":
                    m = avg_ph2(cfg, n)
                else:
                    m = compute_metrics(csv_path(cfg, dataset, n))

                label = cfg['label'] if n == N_SHOTS[0] else ""
                if m:
                    print(f"  {label:<20}  {n:>4}  "
                          f"{pct(m['bacc']):>7}  {pct(m['sens']):>7}  "
                          f"{pct(m['spec']):>7}  {delta(m['bacc'], paper):>8}%")
                else:
                    print(f"  {label:<20}  {n:>4}  {'MISSING':>7}  "
                          f"{'MISSING':>7}  {'MISSING':>7}  {'N/A':>8}")
            print()

    print()

if __name__ == "__main__":
    main()
