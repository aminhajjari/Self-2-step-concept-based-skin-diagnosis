#!/usr/bin/env python
"""Multi-Expert ensemble over existing per-classifier prediction CSVs.

Reads results/label_prediction/, votes across classifiers (the "experts"),
and writes results/multi_expert/. It NEVER modifies the core pipeline or any
existing output; delete this file and results/multi_expert/ to fully remove it.

Usage:
    python multi_expert_classifier.py --n_demos 0 --retrieval rices
"""

import os
import re
import glob
import argparse
from collections import Counter

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

PRED_DIR = "results/label_prediction"
OUT_DIR = "results/multi_expert"
EXPERTS = ["MMed", "Mistral", "MedGemma"]   # set the committee here
TIE_BREAK = "melanoma"                       # used only if a vote ties (even #experts)


def parse_name(fn):
    """Extract fields from a prediction filename, or None if it doesn't match."""
    m = re.match(
        r"(?P<dataset>PH2|Derm7pt|HAM10000)(?:_split_(?P<split>\d+))?_"
        r"(?P<clf>MMed|Mistral|MedGemma)_diagnostic_report_validation_raw_values_False_"
        r"gt_concepts_False_model_extractor_Explicd_n_demos_(?P<demos>\d+)_"
        r"refiner_(?P<refiner>rule|mmed|mistral)_retrieval_(?P<ret>rices|random)\.csv",
        fn)
    return m.groupdict() if m else None


def metrics(y_true, y_pred):
    bacc = balanced_accuracy_score(y_true, y_pred) * 100
    sens = recall_score(y_true, y_pred, pos_label="melanoma", zero_division=0) * 100
    spec = recall_score(y_true, y_pred, pos_label="nevus", zero_division=0) * 100
    return len(y_true), bacc, sens, spec


def vote(preds):
    counts = Counter(preds).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:   # tie
        return TIE_BREAK
    return counts[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_demos", default="0")
    ap.add_argument("--retrieval", default="rices")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Group files by (dataset, split, refiner) for the chosen shot/retrieval.
    groups = {}
    for path in glob.glob(f"{PRED_DIR}/*.csv"):
        info = parse_name(os.path.basename(path))
        if not info:
            continue
        if info["demos"] != args.n_demos or info["ret"] != args.retrieval:
            continue
        key = (info["dataset"], info["split"], info["refiner"])
        groups.setdefault(key, {})[info["clf"]] = path

    rows = []
    for (dataset, split, refiner), files in sorted(groups.items()):
        if not all(e in files for e in EXPERTS):     # need every expert present
            continue
        dfs = {e: pd.read_csv(files[e]).set_index("image_id") for e in EXPERTS}
        ids = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))

        gt, ens = [], []
        for i in ids:
            gt.append(dfs[EXPERTS[0]].loc[i, "gt_response"])
            ens.append(vote([dfs[e].loc[i, "llm_response"] for e in EXPERTS]))

        out = pd.DataFrame({"image_id": ids, "gt_response": gt,
                            "ensemble_response": ens})
        for e in EXPERTS:
            out[f"{e}_response"] = [dfs[e].loc[i, "llm_response"] for i in ids]
        tag = dataset + (f"_split_{split}" if split else "")
        out.to_csv(f"{OUT_DIR}/{tag}_ENSEMBLE_refiner_{refiner}_ndemos_{args.n_demos}.csv",
                   index=False)

        n, bacc, sens, spec = metrics(gt, ens)
        rows.append({"dataset": dataset, "split": split, "refiner": refiner,
                     "n": n, "BAcc": bacc, "Sens": sens, "Spec": spec})

    res = pd.DataFrame(rows)
    if res.empty:
        print("No complete expert groups found. Need all of "
              f"{EXPERTS} for the same (dataset, split, refiner).")
        return

    print(f"\n{'='*70}")
    print(f"  MULTI-EXPERT ENSEMBLE  ({'+'.join(EXPERTS)})  "
          f"[n_demos={args.n_demos}, {args.retrieval}]")
    print('='*70)
    print(f"  {'Dataset':<12}{'Refiner':<10}{'n':>6}{'BAcc':>9}{'Sens':>9}{'Spec':>9}")
    print("  " + "-"*54)

    # PH2 averaged over its splits; Derm7pt/HAM10000 single.
    summ = (res.groupby(["dataset", "refiner"])[["n", "BAcc", "Sens", "Spec"]]
              .mean().reset_index())
    summ.to_csv(f"{OUT_DIR}/ensemble_summary_ndemos_{args.n_demos}_{args.retrieval}.csv",
                index=False)
    for _, r in summ.iterrows():
        print(f"  {r['dataset']:<12}{r['refiner']:<10}{int(r['n']):>6}"
              f"{r['BAcc']:>9.2f}{r['Sens']:>9.2f}{r['Spec']:>9.2f}")
    print()


if __name__ == "__main__":
    main()
