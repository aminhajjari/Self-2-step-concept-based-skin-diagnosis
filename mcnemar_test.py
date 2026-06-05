#!/usr/bin/env python3
"""
mcnemar_test.py
Statistical significance testing using McNemar's test.
Compares your best config against:
  1. No-refinement baseline
  2. Paper baseline (if you have those result files)

Run after the full pipeline completes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    print("Install statsmodels: pip install statsmodels --break-system-packages")
    exit(1)

RESULTS_DIR = Path("results/label_prediction")
N_SPLITS = 5

def load_predictions(dataset, llm, refiner, n_demos=0, split=None):
    suffix = (f"raw_values_False_gt_concepts_False"
              f"_model_extractor_Explicd_n_demos_{n_demos}_refiner_{refiner}")
    if split is not None:
        name = f"{dataset}_split_{split}_{llm}_diagnostic_report_validation_{suffix}.csv"
    else:
        name = f"{dataset}_{llm}_diagnostic_report_validation_{suffix}.csv"
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[["image_id", "gt_response", "llm_response"]]

def run_mcnemar(df_a, df_b, label_a, label_b):
    """McNemar's test between two systems."""
    merged = df_a.merge(df_b, on="image_id", suffixes=("_a", "_b"))
    
    correct_a = merged["llm_response_a"].str.lower() == merged["gt_response_a"].str.lower()
    correct_b = merged["llm_response_b"].str.lower() == merged["gt_response_b"].str.lower()
    
    n00 = (~correct_a & ~correct_b).sum()  # both wrong
    n01 = (~correct_a &  correct_b).sum()  # a wrong, b right
    n10 = ( correct_a & ~correct_b).sum()  # a right, b wrong
    n11 = ( correct_a &  correct_b).sum()  # both right
    
    table = [[n11, n10], [n01, n00]]
    
    # Use exact test for small samples, chi2 for large
    exact = (n01 + n10) < 25
    result = mcnemar(table, exact=exact, correction=True)
    
    acc_a = correct_a.mean() * 100
    acc_b = correct_b.mean() * 100
    
    return {
        'system_a': label_a,
        'system_b': label_b,
        'acc_a': acc_a,
        'acc_b': acc_b,
        'n10': n10, 'n01': n01,
        'pvalue': result.pvalue,
        'significant': result.pvalue < 0.05,
        'test_type': 'exact' if exact else 'chi2'
    }

def avg_ph2_predictions(llm, refiner, n_demos=0):
    """Combine all 5 PH2 splits into one dataframe."""
    dfs = []
    for split in range(N_SPLITS):
        df = load_predictions("PH2", llm, refiner, n_demos, split)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs).drop_duplicates(subset="image_id")

def main():
    print("\n" + "="*85)
    print("  McNEMAR'S SIGNIFICANCE TESTS")
    print("="*85)
    
    COMPARISONS = [
        # Compare refiners against each other
        ("Derm7pt", 0,
         "Rule+Mistral(0-shot)", "Mistral", "rule",
         "MMed+Mistral(0-shot)", "Mistral", "mmed"),
        ("Derm7pt", 1,
         "Rule+Mistral(1-shot)", "Mistral", "rule",
         "MMed+Mistral(1-shot)", "Mistral", "mmed"),
        ("Derm7pt", 1,
         "Rule+MMed(1-shot)", "MMed", "rule",
         "MMed+MMed(1-shot)", "MMed", "mmed"),
        ("HAM10000", 0,
         "Rule+MMed(0-shot)", "MMed", "rule",
         "MMed+MMed(0-shot)", "MMed", "mmed"),
        ("HAM10000", 1,
         "Rule+MMed(1-shot)", "MMed", "rule",
         "MMed+MMed(1-shot)", "MMed", "mmed"),
   ]
    
    print(f"\n  {'Comparison':<45} {'Acc A':>6} {'Acc B':>6} "
          f"{'p-value':>9} {'Sig?':>6}")
    print("  " + "-"*75)
    
    all_results = []
    for (dataset, n_shots, label_a, llm_a, ref_a, 
                            label_b, llm_b, ref_b) in COMPARISONS:
        
        if dataset == "PH2":
            df_a = avg_ph2_predictions(llm_a, ref_a, n_shots)
            df_b = avg_ph2_predictions(llm_b, ref_b, n_shots)
        else:
            df_a = load_predictions(dataset, llm_a, ref_a, n_shots)
            df_b = load_predictions(dataset, llm_b, ref_b, n_shots)
        
        if df_a is None or df_b is None:
            print(f"  {label_a[:20]} vs {label_b[:20]}: MISSING FILES")
            continue
        
        result = run_mcnemar(df_a, df_b, label_a, label_b)
        all_results.append(result)
        
        sig_marker = "✓ YES" if result['significant'] else "  no"
        comp_str = f"{label_a[:22]} vs {label_b[:22]}"
        print(f"  {comp_str:<45} "
              f"{result['acc_a']:>6.2f} {result['acc_b']:>6.2f} "
              f"{result['pvalue']:>9.4f} {sig_marker:>6}")
    
    print("="*85)
    print(f"\n  * p < 0.05 considered statistically significant")
    print(f"  * Exact test used when n01+n10 < 25, chi-square otherwise")
    
    if all_results:
        out = pd.DataFrame(all_results)
        out.to_csv("results/mcnemar_results.csv", index=False)
        print(f"\n  Saved to results/mcnemar_results.csv")

if __name__ == "__main__":
    main()
