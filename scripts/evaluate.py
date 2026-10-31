"""
Comprehensive Model Evaluation on Unseen Test Set.
Computes Top-1, Top-3, Precision, Recall, Macro/Weighted F1, Per-Class metrics,
unknown rejection performance, and generates publication-grade visualizations.
"""

import os
import sys
import argparse
import joblib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from app.service import FaceService
from scripts.extract_features import extract_features_stream


def evaluate_test_set(
    model_path: str,
    test_manifest: str,
    root_dir: str = '.',
    cache_dir: str = 'data/embeddings',
    reports_dir: str = 'reports',
    chunk_size: int = 256
):
    reports_p = Path(reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)
    cache_p = Path(cache_dir)
    test_cache = cache_p / 'test_embeddings.npz'

    print("=====================================================================")
    print("  STAGE 1: EXTRACTING TEST EMBEDDINGS (Unseen Held-Out Test Split)")
    print("=====================================================================")
    X_test, y_test, paths_test, test_audit = extract_features_stream(
        manifest_path=test_manifest,
        root_dir=root_dir,
        cache_path=test_cache,
        chunk_size=chunk_size
    )

    if len(X_test) == 0:
        raise RuntimeError("No valid test embeddings could be extracted.")

    print(f"\nTest embeddings shape: {X_test.shape} across {len(np.unique(y_test))} identities.")

    print("\n=====================================================================")
    print("  STAGE 2: LOADING TRAINED CLASSIFIER AND COMPUTING METRICS")
    print("=====================================================================")
    clf = joblib.load(model_path)
    classes = clf.estimator.classes_
    threshold = clf.threshold

    probs = clf.estimator.predict_proba(X_test)
    raw_preds = classes[np.argmax(probs, axis=1)]
    max_confidences = np.max(probs, axis=1)

    # Filter by trained classes
    valid_mask = np.isin(y_test, classes)
    if not np.all(valid_mask):
        print(f"Warning: {np.sum(~valid_mask)} test samples belong to classes not in training set.")
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]
        raw_preds = raw_preds[valid_mask]
        probs = probs[valid_mask]
        max_confidences = max_confidences[valid_mask]

    # Metrics
    top1_acc = float(accuracy_score(y_test, raw_preds))
    top3_acc = float(top_k_accuracy_score(y_test, probs, k=min(3, len(classes)), labels=classes))
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, raw_preds, average='macro', zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, raw_preds, average='weighted', zero_division=0
    )

    # Threshold-based Rejection Analysis
    accepted_mask = max_confidences >= threshold
    unknown_rejection_rate = float(np.mean(~accepted_mask))
    acc_on_accepted = float(accuracy_score(y_test[accepted_mask], raw_preds[accepted_mask])) if np.sum(accepted_mask) > 0 else 0.0

    print(f"\n--- Closed-Set Metrics (N={len(y_test)}) ---")
    print(f"Top-1 Accuracy (Closed-set):   {top1_acc * 100:.2f}%")
    print(f"Top-3 Accuracy:                {top3_acc * 100:.2f}%")
    print(f"Macro Precision:               {prec_macro * 100:.2f}%")
    print(f"Macro Recall:                  {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:                {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:             {f1_weighted * 100:.2f}%")
    print(f"\n--- Open-Set / Rejection Metrics (Threshold = {threshold:.4f}) ---")
    print(f"Rejection Rate on Test Faces:  {unknown_rejection_rate * 100:.2f}%")
    print(f"Accuracy on Accepted Faces:    {acc_on_accepted * 100:.2f}%")

    # Per-Class Accuracy
    per_class = {}
    for cls in np.unique(y_test):
        cls_mask = y_test == cls
        cls_total = int(np.sum(cls_mask))
        cls_correct = int(np.sum(raw_preds[cls_mask] == cls))
        per_class[str(cls)] = {
            'total': cls_total,
            'correct': cls_correct,
            'accuracy': float(cls_correct / cls_total) if cls_total > 0 else 0.0
        }

    # Save summary report JSON
    evaluation_summary = {
        'test_samples': len(y_test),
        'num_classes_in_test': len(np.unique(y_test)),
        'threshold': float(threshold),
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'macro_precision': float(prec_macro),
        'macro_recall': float(rec_macro),
        'macro_f1': float(f1_macro),
        'weighted_f1': float(f1_weighted),
        'unknown_rejection_rate': unknown_rejection_rate,
        'accuracy_on_accepted': acc_on_accepted,
        'per_class_summary': per_class
    }

    with open(reports_p / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2)

    # ----------------- Visualizations -----------------
    # 1. Per-Class Accuracy Distribution Histogram
    plt.figure(figsize=(8, 5))
    acc_values = [v['accuracy'] * 100 for v in per_class.values()]
    plt.hist(acc_values, bins=20, color='#1f77b4', edgecolor='black', alpha=0.8)
    plt.axvline(np.mean(acc_values), color='red', linestyle='--', label=f'Mean Acc: {np.mean(acc_values):.1f}%')
    plt.title('Distribution of Test Per-Class Accuracies')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Number of Player Classes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_p / 'per_class_accuracy_dist.png', dpi=200)
    plt.close()

    # 2. Confusion Matrix of Top-20 Most Frequent Test Classes
    top20_classes = [c for c, _ in sorted(per_class.items(), key=lambda x: x[1]['total'], reverse=True)[:20]]
    mask_top20 = np.isin(y_test, top20_classes) & np.isin(raw_preds, top20_classes)
    if np.sum(mask_top20) > 10:
        cm = confusion_matrix(y_test[mask_top20], raw_preds[mask_top20], labels=top20_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=top20_classes, yticklabels=top20_classes)
        plt.title('Confusion Matrix (Top-20 Represented Players in Test Split)')
        plt.xlabel('Predicted Player')
        plt.ylabel('Ground Truth Player')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(reports_p / 'confusion_matrix_top20.png', dpi=200)
        plt.close()

    # 3. Confidence Score Distribution (Correct vs Incorrect)
    plt.figure(figsize=(8, 5))
    correct_scores = max_confidences[raw_preds == y_test]
    incorrect_scores = max_confidences[raw_preds != y_test]
    plt.hist(correct_scores, bins=30, alpha=0.7, color='green', label=f'Correct (N={len(correct_scores)})')
    if len(incorrect_scores) > 0:
        plt.hist(incorrect_scores, bins=30, alpha=0.7, color='red', label=f'Incorrect (N={len(incorrect_scores)})')
    plt.axvline(threshold, color='black', linestyle=':', linewidth=2, label=f'Threshold ({threshold:.2f})')
    plt.title('Calibrated Confidence Distribution: Correct vs Incorrect Predictions')
    plt.xlabel('Predicted Confidence Score')
    plt.ylabel('Sample Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_p / 'confidence_distribution.png', dpi=200)
    plt.close()

    print(f"\nVisualizations and reports successfully written to: {reports_p.resolve()}")
    return evaluation_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate face identification model.")
    parser.add_argument('--manifest', '--test-manifest', dest='test_manifest', required=True, help="Test manifest CSV")
    parser.add_argument('--model', default='models/face_index.joblib', help="Path to trained model")
    parser.add_argument('--root', default='.', help="Dataset root directory")
    parser.add_argument('--cache-dir', default='data/embeddings', help="Cache directory")
    parser.add_argument('--reports-dir', default='reports', help="Reports output directory")
    parser.add_argument('--chunk-size', type=int, default=256, help="Chunk size")
    args = parser.parse_args()

    evaluate_test_set(
        model_path=args.model,
        test_manifest=args.test_manifest,
        root_dir=args.root,
        cache_dir=args.cache_dir,
        reports_dir=args.reports_dir,
        chunk_size=args.chunk_size
    )


if __name__ == '__main__':
    main()
