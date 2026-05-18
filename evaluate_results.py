#!/usr/bin/env python3
"""
evaluate_results.py
-------------------
Reads label-prediction CSVs and prints clean, separated comparison tables
for PH2 (per-split + average), Derm7pt, and HAM10000.

Expected CSV files (produced by c_to_y() in run_x_to_c_to_y.py):
  results/label_prediction/PH2_split_{0..4}_{MODEL}_Explicd_raw_values_False_gt_concepts_False_n_demos_0.csv
  results/label_prediction/Derm7pt_{MODEL}_Explicd_raw_values_False_gt_concepts_False_n_demos_0.csv
  results/label_prediction/HAM10000_{MODEL}_Explicd_raw_values_False_gt_concepts_False_n_demos_0.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
)

# ── configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("results/label_prediction")
MODEL_NAME    = "MMed-Llama-3-8B-EnIns"   # basename of your checkpoint folder
CONCEPT_MODEL = "Explicd"
N_DEMOS       = 0
RAW_VALUES    = False
GT_CONCEPTS   = False
N_SPLITS      = 5

# Paper benchmarks (Table 3) for quick comparison
PAPER = {
    "PH2":      {"BAcc": 78.07, "Sens": None,  "Spec": None},
    "Derm7pt":  {"BAcc": 78.56, "Sens": None,  "Spec": None},
    "HAM10000": {"BAcc": 76.00, "Sens": None,  "Spec": None},
}

# ── table drawing helpers ──────────────────────────────────────────────────────

def _col_widths(headers, rows):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    return [w + 2 for w in widths]   # +2 padding

def _hline(widths, left="├", mid="┼", right="┤", fill="─"):
    return left + mid.join(fill * w for w in widths) + right

def _row(cells, widths):
    parts = []
    for cell, w in zip(cells, widths):
        s = str(cell)
        parts.append(" " + s + " " * (w - len(s) - 1))
    return "│" + "│".join(parts) + "│"

def print_table(title, headers, rows):
    widths = _col_widths(headers, rows)
    total  = sum(widths) + len(widths) + 1

    print()
    print("┌" + "─" * (total - 2) + "┐")
    print("│" + title.center(total - 2) + "│")
    print("├" + _hline(widths, "─", "┬", "─")[1:-1] + "┤")
    print(_row(headers, widths))
    print(_hline(widths))
    for r in rows:
        print(_row(r, widths))
    print("└" + _hline(widths, "─", "┴", "─", "─")[1:-1] + "┘")

def pct(v, decimals=2):
    return f"{v:.{decimals}f}%" if v is not None else "N/A"

# ── metrics calculation ────────────────────────────────────────────────────────

def compute_metrics(csv_path: Path):
    """Return dict of metrics from a label-prediction CSV, or None if missing."""
    if not csv_path.exists():
        return None

    df    = pd.read_csv(csv_path)
    y_true = df["gt_response"].str.lower().str.strip()
    y_pred = df["llm_response"].str.lower().str.strip()

    bacc  = balanced_accuracy_score(y_true, y_pred) * 100
    sens  = recall_score(y_true, y_pred, pos_label="melanoma", zero_division=0) * 100
    spec  = recall_score(y_true, y_pred, pos_label="nevus",    zero_division=0) * 100
    cm    = confusion_matrix(y_true, y_pred, labels=["melanoma", "nevus"])

    tn, fp, fn, tp = cm.ravel()
    return {
        "n"    : len(df),
        "bacc" : bacc,
        "sens" : sens,
        "spec" : spec,
        "TP"   : int(tp), "TN": int(tn),
        "FP"   : int(fp), "FN": int(fn),
    }

def csv_path(dataset, split=None):
    suffix = (f"raw_values_{RAW_VALUES}_gt_concepts_{GT_CONCEPTS}"
              f"_model_extractor_{CONCEPT_MODEL}_n_demos_{N_DEMOS}")
    if split is not None:
        name = f"{dataset}_split_{split}_{MODEL_NAME}_diagnostic_report_validation_{suffix}.csv"
    else:
        name = f"{dataset}_{MODEL_NAME}_diagnostic_report_validation_{suffix}.csv"
    return RESULTS_DIR / name

# ── table printers ─────────────────────────────────────────────────────────────

def table_ph2_per_split(results):
    """Table 1 — PH2 results for every split."""
    headers = ["Split", "Samples", "BAcc (%)", "Sensitivity (%)", "Specificity (%)",
               "TP", "TN", "FP", "FN"]
    rows = []
    for s, m in enumerate(results):
        if m:
            rows.append([f"Split {s}", m["n"],
                         pct(m["bacc"]), pct(m["sens"]), pct(m["spec"]),
                         m["TP"], m["TN"], m["FP"], m["FN"]])
        else:
            rows.append([f"Split {s}", "—", "MISSING", "MISSING", "MISSING",
                         "—", "—", "—", "—"])
    print_table(f"TABLE 1 — PH2  Per-Split Results  [{MODEL_NAME}]", headers, rows)


def table_ph2_average(results):
    """Table 2 — PH2 average across splits."""
    valid = [m for m in results if m is not None]
    if not valid:
        print("\n[WARNING] No PH2 results found — cannot compute average.\n")
        return

    def stats(key):
        vals = [m[key] for m in valid]
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    paper_bacc = PAPER["PH2"]["BAcc"]

    headers = ["Metric", "Mean", "Std", "Min", "Max", f"Paper (0-shot)"]
    rows = []
    for label, key in [("BAcc (%)", "bacc"), ("Sensitivity (%)", "sens"), ("Specificity (%)", "spec")]:
        mean, std, mn, mx = stats(key)
        paper_val = pct(paper_bacc) if key == "bacc" else "—"
        rows.append([label, pct(mean), pct(std), pct(mn), pct(mx), paper_val])

    print_table(
        f"TABLE 2 — PH2  Average across {len(valid)}/{N_SPLITS} Splits  [{MODEL_NAME}]",
        headers, rows
    )


def table_single_dataset(dataset, m):
    """Table 3 / 4 — one row for Derm7pt or HAM10000."""
    paper_bacc = PAPER[dataset]["BAcc"]
    headers = ["Dataset", "Samples", "BAcc (%)", "Sensitivity (%)", "Specificity (%)",
               "TP", "TN", "FP", "FN", "Paper BAcc (0-shot)"]
    if m:
        rows = [[dataset, m["n"],
                 pct(m["bacc"]), pct(m["sens"]), pct(m["spec"]),
                 m["TP"], m["TN"], m["FP"], m["FN"],
                 pct(paper_bacc)]]
    else:
        rows = [[dataset, "—", "MISSING", "MISSING", "MISSING",
                 "—", "—", "—", "—", pct(paper_bacc)]]

    tbl_num = "3" if dataset == "Derm7pt" else "4"
    print_table(f"TABLE {tbl_num} — {dataset}  Results  [{MODEL_NAME}]", headers, rows)


def table_combined_summary(ph2_results, derm_m, ham_m):
    """Table 5 — all datasets side-by-side with paper comparison."""
    valid_ph2 = [m for m in ph2_results if m]
    if valid_ph2:
        ph2_bacc = np.mean([m["bacc"] for m in valid_ph2])
        ph2_sens = np.mean([m["sens"] for m in valid_ph2])
        ph2_spec = np.mean([m["spec"] for m in valid_ph2])
    else:
        ph2_bacc = ph2_sens = ph2_spec = None

    headers = ["Dataset", "BAcc (%)", "Sensitivity (%)", "Specificity (%)",
               "Paper BAcc (%)", "Δ vs Paper"]

    def delta(our, paper):
        if our is None or paper is None:
            return "N/A"
        d = our - paper
        return f"{'+' if d >= 0 else ''}{d:.2f}%"

    rows = [
        ["PH2 (avg)",
         pct(ph2_bacc), pct(ph2_sens), pct(ph2_spec),
         pct(PAPER["PH2"]["BAcc"]),
         delta(ph2_bacc, PAPER["PH2"]["BAcc"])],

        ["Derm7pt",
         pct(derm_m["bacc"]) if derm_m else "MISSING",
         pct(derm_m["sens"]) if derm_m else "MISSING",
         pct(derm_m["spec"]) if derm_m else "MISSING",
         pct(PAPER["Derm7pt"]["BAcc"]),
         delta(derm_m["bacc"] if derm_m else None, PAPER["Derm7pt"]["BAcc"])],

        ["HAM10000",
         pct(ham_m["bacc"]) if ham_m else "MISSING",
         pct(ham_m["sens"]) if ham_m else "MISSING",
         pct(ham_m["spec"]) if ham_m else "MISSING",
         pct(PAPER["HAM10000"]["BAcc"]),
         delta(ham_m["bacc"] if ham_m else None, PAPER["HAM10000"]["BAcc"])],
    ]

    print_table(
        f"TABLE 5 — COMBINED SUMMARY  (vs Paper Table 3, 0-shot)  [{MODEL_NAME}]",
        headers, rows
    )

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 78
    print()
    print(sep)
    print("  RESULTS SUMMARY — Two-Step Concept-Based Skin Lesion Diagnosis")
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Concept model : {CONCEPT_MODEL}   |  n_demos={N_DEMOS}")
    print(f"  Results dir   : {RESULTS_DIR}")
    print(sep)

    # ── load all results ──────────────────────────────────────────────────────
    ph2_results = [compute_metrics(csv_path("PH2", s)) for s in range(N_SPLITS)]
    derm_m      = compute_metrics(csv_path("Derm7pt"))
    ham_m       = compute_metrics(csv_path("HAM10000"))

    # ── PH2 section ───────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  SECTION 1 — PH2 Dataset (5-fold cross-validation)")
    print("─" * 78)
    table_ph2_per_split(ph2_results)
    table_ph2_average(ph2_results)

    # ── Derm7pt section ───────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  SECTION 2 — Derm7pt Dataset")
    print("─" * 78)
    table_single_dataset("Derm7pt", derm_m)

    # ── HAM10000 section ──────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  SECTION 3 — HAM10000 Dataset")
    print("─" * 78)
    table_single_dataset("HAM10000", ham_m)

    # ── combined summary ──────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  SECTION 4 — Combined Summary  (all datasets vs paper)")
    print("─" * 78)
    table_combined_summary(ph2_results, derm_m, ham_m)

    print()
    print(sep)
    print("  END OF SUMMARY")
    print(sep)
    print()


if __name__ == "__main__":
    main()
