import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score


class IdentityClassifier:
    """Production Linear SVM with Sigmoid Probability Calibration and Dynamic Rejection Thresholding."""

    def __init__(self, estimator=None, threshold: float = 0.55, metadata: dict = None):
        self.estimator = estimator
        self.threshold = float(threshold)
        self.metadata = metadata or {}

    @property
    def classes_(self):
        if self.estimator is not None and hasattr(self.estimator, 'classes_'):
            return self.estimator.classes_
        return None

    def fit(self, X, y, cv: int = 3):
        """Fit Linear SVM with probability calibration."""
        y_arr = np.asarray(y)
        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            raise ValueError('Need at least two distinct identities to fit classifier.')

        # Linear SVC with balanced class weights, calibrated via Platt sigmoid scaling
        base_svc = SVC(kernel='linear', class_weight='balanced')
        # CalibratedClassifierCV wraps the base estimator
        self.estimator = CalibratedClassifierCV(
            estimator=base_svc,
            method='sigmoid',
            cv=cv
        )
        self.estimator.fit(np.asarray(X, dtype=np.float32), y_arr)
        self.metadata['num_classes'] = len(unique_classes)
        self.metadata['num_samples'] = len(y_arr)
        return self

    def tune_threshold_on_validation(
        self,
        val_X: np.ndarray,
        val_y: np.ndarray,
        unknown_X: np.ndarray = None,
        search_range: np.ndarray = None
    ) -> float:
        """
        Tune the rejection threshold using validation data to maximize balanced F1 score
        or maintain a bounded False Acceptance Rate (FAR).
        """
        if search_range is None:
            search_range = np.linspace(0.10, 0.90, 81)

        val_X = np.asarray(val_X, dtype=np.float32)
        val_y = np.asarray(val_y)
        probs = self.estimator.predict_proba(val_X)
        max_probs = np.max(probs, axis=1)
        pred_indices = np.argmax(probs, axis=1)
        pred_classes = self.estimator.classes_[pred_indices]

        best_score = -1.0
        best_threshold = self.threshold

        # If non-roster / unknown faces are provided, optimize open-set trade-off
        if unknown_X is not None and len(unknown_X) > 0:
            unk_probs = self.estimator.predict_proba(np.asarray(unknown_X, dtype=np.float32))
            unk_max_probs = np.max(unk_probs, axis=1)
        else:
            unk_max_probs = None

        for t in search_range:
            # Known predictions accepted if score >= t
            accepted_mask = max_probs >= t
            preds = np.where(accepted_mask, pred_classes, '__UNKNOWN__')
            
            # Ground truth: val_y are known classes
            acc_known = np.mean(preds == val_y)
            
            if unk_max_probs is not None:
                # Unknown faces should be rejected (< t)
                unk_rejected = np.mean(unk_max_probs < t)
                combined_score = 0.6 * acc_known + 0.4 * unk_rejected
            else:
                # If only closed-set validation, maximize known F1 with penalty for over-rejection
                coverage = np.mean(accepted_mask)
                f1_known = f1_score(val_y, preds, labels=self.estimator.classes_, average='macro', zero_division=0)
                combined_score = f1_known * (0.5 + 0.5 * coverage)

            if combined_score > best_score:
                best_score = combined_score
                best_threshold = float(t)

        self.threshold = best_threshold
        self.metadata['tuned_threshold'] = best_threshold
        self.metadata['threshold_tuning_score'] = float(best_score)
        return best_threshold

    def predict(self, embedding: np.ndarray, top_k: int = 3) -> dict:
        """Predict identity from a 512D embedding with calibrated confidence and rejection."""
        if self.estimator is None:
            raise RuntimeError('No classifier loaded')

        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        probs = self.estimator.predict_proba(emb)[0]
        
        # Rank candidate classes
        top_k = min(top_k, len(probs))
        order = np.argsort(probs)[::-1][:top_k]
        top_idx = order[0]
        top_confidence = float(probs[top_idx])
        top_identity = str(self.estimator.classes_[top_idx])

        is_unknown = top_confidence < self.threshold
        predicted_identity = None if is_unknown else top_identity

        candidates = [
            {'identity': str(self.estimator.classes_[i]), 'confidence': float(probs[i])}
            for i in order
        ]

        return {
            'identity': predicted_identity,
            'confidence': top_confidence,
            'unknown': is_unknown,
            'threshold': self.threshold,
            'top_k': candidates
        }
