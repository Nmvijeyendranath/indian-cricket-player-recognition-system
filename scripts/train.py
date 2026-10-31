"""
Train Calibrated Linear SVM Face Classifier on ArcFace Embeddings.
Uses chunked streaming, incremental checkpoint caching, and validation-based threshold calibration.
"""

import os
import sys
import argparse
import joblib
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report

from app.classifier import IdentityClassifier
from app.config import INSIGHTFACE_MODEL, FACE_DET_SIZE, MODEL_PATH
from scripts.extract_features import extract_features_stream


def train_pipeline(
    train_manifest: str,
    val_manifest: str = None,
    root_dir: str = '.',
    out_model_path: str = MODEL_PATH,
    cache_dir: str = 'data/embeddings',
    chunk_size: int = 256,
    default_threshold: float = 0.55,
    tune_threshold: bool = True
):
    cache_p = Path(cache_dir)
    cache_p.mkdir(parents=True, exist_ok=True)
    train_cache = cache_p / 'train_embeddings.npz'
    val_cache = cache_p / 'val_embeddings.npz'

    print("=====================================================================")
    print("  STAGE 1: EXTRACTING TRAINING EMBEDDINGS (ArcFace 512D on GPU)")
    print("=====================================================================")
    X_train, y_train, paths_train, train_audit = extract_features_stream(
        manifest_path=train_manifest,
        root_dir=root_dir,
        cache_path=train_cache,
        chunk_size=chunk_size
    )

    if len(X_train) == 0:
        raise RuntimeError("No valid single-face images extracted from training manifest.")

    unique_classes_train = np.unique(y_train)
    print(f"\nTraining embedding matrix shape: {X_train.shape} across {len(unique_classes_train)} identities.")

    # Validation embeddings
    X_val, y_val = None, None
    if val_manifest and Path(val_manifest).exists():
        print("\n=====================================================================")
        print("  STAGE 2: EXTRACTING VALIDATION EMBEDDINGS")
        print("=====================================================================")
        X_val, y_val, paths_val, val_audit = extract_features_stream(
            manifest_path=val_manifest,
            root_dir=root_dir,
            cache_path=val_cache,
            chunk_size=chunk_size
        )
        print(f"Validation embedding matrix shape: {X_val.shape} across {len(np.unique(y_val))} identities.")

    print("\n=====================================================================")
    print("  STAGE 3: FITTING CALIBRATED LINEAR SVM (Balanced Weights)")
    print("=====================================================================")
    t_fit0 = time.time()
    clf = IdentityClassifier(threshold=default_threshold)
    clf.fit(X_train, y_train, cv=3)
    t_fit = time.time() - t_fit0
    print(f"SVM training and 3-fold sigmoid calibration completed in {t_fit:.2f}s.")

    # Validation & Rejection Threshold Tuning
    validation_metrics = {}
    if X_val is not None and len(X_val) > 0 and tune_threshold:
        print("\n=====================================================================")
        print("  STAGE 4: TUNING REJECTION THRESHOLD ON HELD-OUT VALIDATION SET")
        print("=====================================================================")
        optimal_threshold = clf.tune_threshold_on_validation(X_val, y_val)
        print(f"Optimal tuned validation threshold: {optimal_threshold:.4f} (default was {default_threshold})")

        # Evaluate on validation set
        val_probs = clf.estimator.predict_proba(X_val)
        val_preds_raw = clf.estimator.classes_[np.argmax(val_probs, axis=1)]
        val_scores = np.max(val_probs, axis=1)

        val_top1 = accuracy_score(y_val, val_preds_raw)
        val_top3 = top_k_accuracy_score(
            y_val,
            val_probs,
            k=min(3, len(clf.estimator.classes_)),
            labels=clf.estimator.classes_
        )

        val_accepted = val_scores >= optimal_threshold
        val_rejected_rate = float(np.mean(~val_accepted))

        validation_metrics = {
            'val_samples': len(y_val),
            'val_top1_accuracy_unfiltered': float(val_top1),
            'val_top3_accuracy_unfiltered': float(val_top3),
            'val_rejected_rate_at_threshold': val_rejected_rate,
            'tuned_threshold': float(optimal_threshold)
        }
        print(f"Validation Top-1 Accuracy: {val_top1 * 100:.2f}%")
        print(f"Validation Top-3 Accuracy: {val_top3 * 100:.2f}%")
        print(f"Validation Rejection Rate at threshold: {val_rejected_rate * 100:.2f}%")

    # Serialize complete model bundle
    out_p = Path(out_model_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    clf.metadata.update({
        'model_name': 'ArcFace_BuffaloL_LinearSVM_Calibrated',
        'model_version': '1.0.0',
        'embedding_dim': 512,
        'embedding_backbone': INSIGHTFACE_MODEL,
        'det_size': FACE_DET_SIZE,
        'classes': list(clf.estimator.classes_),
        'num_classes': len(clf.estimator.classes_),
        'training_samples': len(X_train),
        'validation_metrics': validation_metrics,
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

    joblib.dump(clf, out_p)
    print(f"\nModel artifact successfully saved to: {out_p.resolve()}")
    print("=====================================================================\n")
    return clf


def main():
    parser = argparse.ArgumentParser(description="Train face identification model.")
    parser.add_argument('--manifest', '--train-manifest', dest='train_manifest', required=True, help="Training manifest CSV")
    parser.add_argument('--val-manifest', default=None, help="Validation manifest CSV")
    parser.add_argument('--root', default='.', help="Root directory for dataset image paths")
    parser.add_argument('--out', default=MODEL_PATH, help="Output path for joblib model artifact")
    parser.add_argument('--cache-dir', default='data/embeddings', help="Directory to store checkpointed embeddings")
    parser.add_argument('--chunk-size', type=int, default=256, help="Streaming batch size")
    parser.add_argument('--threshold', type=float, default=0.55, help="Default rejection threshold")
    parser.add_argument('--no-tune-threshold', action='store_true', help="Skip threshold tuning on validation set")

    args = parser.parse_args()
    train_pipeline(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        root_dir=args.root,
        out_model_path=args.out,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        default_threshold=args.threshold,
        tune_threshold=not args.no_tune_threshold
    )


if __name__ == '__main__':
    main()
