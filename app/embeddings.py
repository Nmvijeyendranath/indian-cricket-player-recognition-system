import os
import sys
from pathlib import Path
import numpy as np

# Automatically discover Torch CUDA runtime DLLs on Windows
if sys.platform == 'win32':
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / 'lib'
        if torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib))
            except (AttributeError, OSError):
                pass
            os.environ['PATH'] = str(torch_lib) + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass


class FacePipelineError(RuntimeError):
    """Exception raised when face detection or alignment fails."""
    pass


class InsightFaceEmbedder:
    """Production InsightFace ArcFace embeddings (buffalo_l) with GPU acceleration and CPU fallback."""

    def __init__(self, model_name: str = 'buffalo_l', det_size: tuple = (640, 640), prefer_gpu: bool = True):
        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort
        except ImportError as e:
            raise FacePipelineError(
                'Install insightface, onnxruntime-gpu (or onnxruntime), and opencv-python.'
            ) from e

        available_providers = ort.get_available_providers()
        providers = []
        ctx_id = -1

        if prefer_gpu and 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            ctx_id = 0
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')

        if not providers:
            providers = ['CPUExecutionProvider']
            ctx_id = -1

        # Restrict to detection and recognition only, avoiding heavy unnecessary 3D/attribute models
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.providers = providers
        self.ctx_id = ctx_id

    def detect_faces(self, image):
        """Run face detection on a PIL Image or RGB numpy array."""
        import cv2
        if not hasattr(image, 'convert'):
            rgb = np.asarray(image)
        else:
            rgb = np.asarray(image.convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.app.get(bgr)

    def embed(self, image):
        """Extract a 512D L2-normalized embedding for an image with exactly one face."""
        faces = self.detect_faces(image)
        if len(faces) == 0:
            raise FacePipelineError('No face detected in image.')
        if len(faces) > 1:
            raise FacePipelineError(f'Expected exactly one face; detected {len(faces)} faces.')

        v = np.asarray(faces[0].normed_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        return v / max(norm, 1e-12)

    def extract_face_crop(self, image, target_size: tuple = (224, 224)):
        """Extract and align the single detected face crop as a PIL Image."""
        import cv2
        from PIL import Image
        from insightface.utils import face_align

        if not hasattr(image, 'convert'):
            rgb = np.asarray(image)
        else:
            rgb = np.asarray(image.convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if len(faces) != 1:
            raise FacePipelineError(f'Expected exactly one face for crop, detected {len(faces)}')

        face = faces[0]
        aligned_bgr = face_align.norm_crop(bgr, landmark=face.kps, image_size=target_size[0])
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(aligned_rgb)
