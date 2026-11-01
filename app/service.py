import os
import joblib
from pathlib import Path
from app.config import MODEL_PATH, INSIGHTFACE_MODEL, FACE_DET_SIZE
from app.embeddings import InsightFaceEmbedder, FacePipelineError


class FaceService:
    """Singleton-style service managing embedder and classifier inference lifecycle."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._embedder = None
        self._classifier = None

    def _resolve_model_path(self) -> Path:
        """Resolve model path whether run from repo root, parent directory, or package directory."""
        p = Path(self.model_path)
        if p.exists():
            return p
        pkg_relative = Path(__file__).resolve().parent.parent / self.model_path
        if pkg_relative.exists():
            return pkg_relative
        candidate = Path('cricket-face-id') / self.model_path
        if candidate.exists():
            return candidate
        pkg_models = Path(__file__).resolve().parent.parent / 'models' / 'face_index.joblib'
        if pkg_models.exists():
            return pkg_models
        return p

    @property
    def embedder(self) -> InsightFaceEmbedder:
        if self._embedder is None:
            self._embedder = InsightFaceEmbedder(
                model_name=INSIGHTFACE_MODEL,
                det_size=(FACE_DET_SIZE, FACE_DET_SIZE),
                prefer_gpu=True
            )
        return self._embedder

    @property
    def classifier(self):
        if self._classifier is None:
            resolved_path = self._resolve_model_path()
            if not resolved_path.exists():
                raise RuntimeError(f"Trained model artifact not found at '{self.model_path}'. Run training first.")
            self._classifier = joblib.load(resolved_path)
        return self._classifier

    def identify(self, image, top_k: int = 3) -> dict:
        """
        End-to-end identification:
        1. Validates image.
        2. Detects and aligns face using InsightFace on GPU.
        3. Generates L2-normalized 512D ArcFace embedding.
        4. Predicts identity, calibrated confidence, unknown rejection flag, and top-k candidates.
        """
        embedding = self.embedder.embed(image)
        result = self.classifier.predict(embedding, top_k=top_k)
        return result

    def get_status(self) -> dict:
        """Return operational status and metadata."""
        resolved_path = self._resolve_model_path()
        is_model_loaded = resolved_path.exists()
        classes_attr = getattr(self.classifier, 'classes_', None)
        if classes_attr is None and hasattr(self.classifier, 'estimator'):
            classes_attr = getattr(self.classifier.estimator, 'classes_', [])
        classes_count = len(classes_attr) if (is_model_loaded and classes_attr is not None) else 0
        threshold = self.classifier.threshold if is_model_loaded else None
        return {
            'status': 'ready' if is_model_loaded else 'needs_training',
            'model_path': str(resolved_path),
            'classes_count': classes_count,
            'rejection_threshold': threshold,
            'gpu_accelerated': 'CUDAExecutionProvider' in getattr(self.embedder, 'providers', [])
        }
