import os
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH=os.getenv('MODEL_PATH','models/face_index.joblib')
INSIGHTFACE_MODEL=os.getenv('INSIGHTFACE_MODEL','buffalo_l')
FACE_DET_SIZE=int(os.getenv('FACE_DET_SIZE','640'))
MAX_UPLOAD_MB=int(os.getenv('MAX_UPLOAD_MB','12'))
TOP_K=int(os.getenv('TOP_K','5'))
