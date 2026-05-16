#!/usr/bin/env python3
"""
Evaluate classification results and compare with paper benchmarks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score
)

def calculate_metrics(csv_file):
    """Calculate comprehensive metrics from results CSV."""
    df = pd.read_csv(csv_file)
    
    # Extract ground truth and predictions
    y_true = df['gt_response'].str.lower().str.strip()
    y_pred = df['llm_response'].str.lower().str.strip()
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    # Calculate sensitivity and specificity
    # Assuming 'melanoma' is positive class
    sensitivity = recall_score(y_true, y_pred, pos_label='melanoma', zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label='nevus', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['melanoma', 'nevus'])
    
    return {
        'file': Path(csv_file).name,
        'n_samples': len(df),
        'accuracy': acc * 100,
        'balanced_accuracy': bacc * 100,
        'sensitivity': sensitivity * 100,
        'specificity': specificity * 100,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }

def print_detailed_results(metrics, dataset_name):
    """Print detailed results for a dataset."""
    print(f"\n{'='*70}")
    print(f"📊 {dataset_name}")
    print(f"{'='*70}")
    print(f"Samples: {metrics['n_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    print(f"Sensitivity (TPR): {metrics['sensitivity']:.2f}%")
    print(f"Specificity (TNR): {metrics['specificity']:.2f}%")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Melanoma  Nevus")
    print(f"True Melanoma    {metrics['confusion_matrix'][0,0]:3d}     {metrics['confusion_matrix'][0,1]:3d}")
    print(f"     Nevus       {metrics['confusion_matrix'][1,0]:3d}     {metrics['confusion_matrix'][1,1]:3d}")
    
    # Class distribution
    print(f"\n📊 Class Distribution:")
    true_counts = metrics['y_true'].value_counts()
    pred_counts = metrics['y_pred'].value_counts()
    print(f"  Ground Truth: Melanoma={true_counts.get('melanoma', 0)}, Nevus={true_counts.get('nevus', 0)}")
    print(f"  Predictions:  Melanoma={pred_counts.get('melanoma', 0)}, Nevus={pred_counts.get('nevus', 0)}")

def compare_with_paper():
    """Compare results with paper's Table 3 benchmarks."""
    paper_results = {
        'PH2': {
            'ExpLICD + MMed (0-shot)': 78.07,
            'ExpLICD + Mistral (0-shot)': 77.44,
            'ExpLICD + MMed (1-shot)': 85.05,
        },
        'Derm7pt': {
            'ExpLICD + MMed (0-shot)': 78.56,
            'ExpLICD + Mistral (0-shot)': 79.10,
            'ExpLICD + MMed (8-shot)': 79.78,
        },
        'HAM10000': {
            'ExpLICD + MMed (0-shot)': 76.00,
            'ExpLICD + Mistral (0-shot)': 75.50,
            'ExpLICD + MMed (2-shot)': 75.00,
        }
    }
    
    print(f"\n{'='*70}")
    print(f"📋 COMPARISON WITH PAPER (Table 3)")
    print(f"{'='*70}")
    
    return paper_results

# Main evaluation
if __name__ == "__main__":
    results_dir = Path("results/label_prediction")
    
    print("\n" + "="*70)
    print("🔬 MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Evaluate PH2 (average across 5 splits)
    print("\n" + "🔵 " * 35)
    print("DATASET 1: PH2 (5-fold Cross Validation)")
    print("🔵 " * 35)
    
    ph2_metrics = []
    for split in range(5):
        csv_file = results_dir / f"PH2_split_{split}_MMed-Llama-3-8B_Explicd_raw_values_False_gt_concepts_False_n_demos_0.csv"
        if csv_file.exists():
            metrics = calculate_metrics(csv_file)
            ph2_metrics.append(metrics)
            print_detailed_results(metrics, f"PH2 Split {split}")
    
    # Average PH2 results
    if ph2_metrics:
        avg_bacc = np.mean([m['balanced_accuracy'] for m in ph2_metrics])
        avg_sens = np.mean([m['sensitivity'] for m in ph2_metrics])
        avg_spec = np.mean([m['specificity'] for m in ph2_metrics])
        
        print(f"\n{'='*70}")
        print(f"📊 PH2 AVERAGE ACROSS 5 SPLITS")
        print(f"{'='*70}")
        print(f"Balanced Accuracy: {avg_bacc:.2f}%")
        print(f"Sensitivity: {avg_sens:.2f}%")
        print(f"Specificity: {avg_spec:.2f}%")
    
    # Evaluate Derm7pt
    print("\n" + "🟢 " * 35)
    print("DATASET 2: Derm7pt")
    print("🟢 " * 35)
    
    derm7pt_file = results_dir / "Derm7pt_MMed-Llama-3-8B_Explicd_raw_values_False_gt_concepts_False_n_demos_0.csv"
    if derm7pt_file.exists():
        derm7pt_metrics = calculate_metrics(derm7pt_file)
        print_detailed_results(derm7pt_metrics, "Derm7pt")
    
    # Evaluate HAM10000
    print("\n" + "🟣 " * 35)
    print("DATASET 3: HAM10000")
    print("🟣 " * 35)
    
    ham_file = results_dir / "HAM10000_MMed-Llama-3-8B_Explicd_raw_values_False_gt_concepts_False_n_demos_0.csv"
    if ham_file.exists():
        ham_metrics = calculate_metrics(ham_file)
        print_detailed_results(ham_metrics, "HAM10000")
    
    # Compare with paper
    paper_results = compare_with_paper()
    
    print(f"\n📈 Your Results vs Paper:")
    if ph2_metrics:
        print(f"  PH2:      Your={avg_bacc:.2f}% | Paper={paper_results['PH2']['ExpLICD + MMed (0-shot)']}%")
    if derm7pt_file.exists():
        print(f"  Derm7pt:  Your={derm7pt_metrics['balanced_accuracy']:.2f}% | Paper={paper_results['Derm7pt']['ExpLICD + MMed (0-shot)']}%")
    if ham_file.exists():
        print(f"  HAM10000: Your={ham_metrics['balanced_accuracy']:.2f}% | Paper={paper_results['HAM10000']['ExpLICD + MMed (0-shot)']}%")
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70 + "\n")
