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


def _get_quantized_model_paths():
    """Resolve paths to INT8-quantized ONNX models shipped with the app."""
    base = Path(__file__).resolve().parent.parent / 'models'
    det_path = base / 'det_10g_int8.onnx'
    rec_path = base / 'w600k_r50_int8.onnx'
    return det_path, rec_path


class InsightFaceEmbedder:
    """
    Memory-optimised face embedder using INT8-quantized ArcFace ONNX models.
    Falls back to standard InsightFace buffalo_l if quantized models are absent.
    Total RAM footprint: ~107 MB (vs ~370 MB for full float32 models).
    """

    def __init__(self, model_name: str = 'buffalo_l', det_size: tuple = (320, 320), prefer_gpu: bool = True):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise FacePipelineError('Install onnxruntime and opencv-python.') from e

        det_path, rec_path = _get_quantized_model_paths()

        # Use memory-optimised INT8 path when quantized models are present
        if det_path.exists() and rec_path.exists():
            self._load_quantized(det_path, rec_path, det_size, ort)
        else:
            # Fallback: standard InsightFace buffalo_l (works locally with full models)
            self._load_insightface(model_name, det_size, prefer_gpu, ort)

    def _load_quantized(self, det_path, rec_path, det_size, ort):
        """Load INT8 quantized ONNX models directly with memory-saving session options."""
        import cv2
        from insightface.model_zoo.retinaface import RetinaFace
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX

        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']

        det_session = ort.InferenceSession(str(det_path), sess_options=opts, providers=providers)
        self._det_model = RetinaFace(model_file=str(det_path), session=det_session)
        self._det_model.prepare(ctx_id=-1, input_size=det_size, det_thresh=0.5)

        rec_session = ort.InferenceSession(str(rec_path), sess_options=opts, providers=providers)
        self._rec_model = ArcFaceONNX(model_file=str(rec_path), session=rec_session)
        self._rec_model.prepare(ctx_id=-1)

        self.providers = providers
        self.ctx_id = -1
        self.app = None
        self._det_size = det_size
        self._mode = 'quantized'

    def _load_insightface(self, model_name, det_size, prefer_gpu, ort):
        """Fallback: standard InsightFace FaceAnalysis pipeline."""
        from insightface.app import FaceAnalysis

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

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.providers = providers
        self.ctx_id = ctx_id
        self._mode = 'standard'

    def detect_faces(self, image):
        """Run face detection on a PIL Image or RGB numpy array."""
        import cv2
        if not hasattr(image, 'convert'):
            rgb = np.asarray(image)
        else:
            rgb = np.asarray(image.convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self._mode == 'quantized':
            bboxes, kpss = self._det_model.detect(bgr, input_size=self._det_size)
            # Build face-like objects compatible with embed()
            faces = []
            if bboxes is not None:
                for i, bbox in enumerate(bboxes):
                    face = type('Face', (), {})()
                    face.bbox = bbox[:4]
                    face.det_score = float(bbox[4])
                    face.kps = kpss[i] if kpss is not None else None
                    face._bgr = bgr
                    faces.append(face)
            return faces
        else:
            return self.app.get(bgr)

    def embed(self, image):
        """Extract a 512D L2-normalized embedding for an image with exactly one face."""
        faces = self.detect_faces(image)
        if len(faces) == 0:
            raise FacePipelineError('No face detected in image.')
        if len(faces) > 1:
            raise FacePipelineError(f'Expected exactly one face; detected {len(faces)} faces.')

        if self._mode == 'quantized':
            import cv2
            from insightface.utils import face_align
            face = faces[0]
            if face.kps is not None:
                aligned = face_align.norm_crop(face._bgr, landmark=face.kps, image_size=112)
            else:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                crop = face._bgr[y1:y2, x1:x2]
                aligned = cv2.resize(crop, (112, 112))
            embedding = self._rec_model.get_feat(aligned).flatten()
            v = np.asarray(embedding, dtype=np.float32)
        else:
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

        if self._mode == 'quantized':
            bboxes, kpss = self._det_model.detect(bgr, input_size=self._det_size)
            if bboxes is None or len(bboxes) != 1:
                raise FacePipelineError(f'Expected exactly one face for crop.')
            aligned_bgr = face_align.norm_crop(bgr, landmark=kpss[0], image_size=target_size[0])
        else:
            faces = self.app.get(bgr)
            if len(faces) != 1:
                raise FacePipelineError(f'Expected exactly one face for crop, detected {len(faces)}')
            aligned_bgr = face_align.norm_crop(bgr, landmark=faces[0].kps, image_size=target_size[0])

        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(aligned_rgb)
