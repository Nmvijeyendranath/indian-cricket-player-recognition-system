import os
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.config import MAX_UPLOAD_MB, TOP_K
from app.embeddings import FacePipelineError
from app.service import FaceService

app = FastAPI(
    title='Indian Cricket Team Member Face Identification',
    description='Production-quality closed-set face identification with unknown-person rejection.'
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

service = FaceService()


@app.on_event('startup')
def startup_event():
    """Warm up classifier and InsightFace embedder so the first request is fast."""
    try:
        _ = service.classifier
        _ = service.embedder
    except Exception as e:
        print(f'Startup model loading warning: {e}')


@app.get('/health')
def health():
    """Health check and model status."""
    return service.get_status()


@app.post('/identify')
def identify(file: UploadFile = File(...), top_k: int = TOP_K):
    """
    Identify cricket player from an uploaded facial image.
    Rejects images with no face, multiple faces, or confidence below calibrated threshold.
    """
    if not 1 <= top_k <= 20:
        raise HTTPException(status_code=400, detail='top_k must be between 1 and 20')

    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Image exceeds upload limit (12MB)')

    try:
        image = Image.open(BytesIO(data))
        image.load()
        if image.width * image.height > 25_000_000:
            raise HTTPException(status_code=413, detail='Image dimensions exceed 25 megapixels')
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail='Invalid or unreadable image file')

    try:
        result = service.identify(image, top_k=top_k)
        return result
    except FacePipelineError as fpe:
        raise HTTPException(status_code=422, detail=str(fpe))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference failure: {str(e)}')


# Resolve directories relative to package root
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / 'static'
PLAYER_IMAGES_DIR = BASE_DIR / 'data' / 'archive (3)' / 'indian cricketer'
REPORTS_DIR = BASE_DIR / 'reports'

if PLAYER_IMAGES_DIR.exists():
    app.mount('/player-images', StaticFiles(directory=str(PLAYER_IMAGES_DIR)), name='player-images')

if REPORTS_DIR.exists():
    app.mount('/reports', StaticFiles(directory=str(REPORTS_DIR)), name='reports')

if STATIC_DIR.exists():
    app.mount('/', StaticFiles(directory=str(STATIC_DIR), html=True), name='static')
