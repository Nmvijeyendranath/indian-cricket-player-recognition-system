# Next agent handoff

This is a backend-first FastAPI starter in `/home/ubuntu/work/cricket-face-id`. No user dataset or face weights are included. ArcFace via optional InsightFace is the recommended embedder; SVM calibration and threshold are only starter defaults and require validation calibration. Training is streamed in bounded manifest/image batches but stores embeddings in memory for SVM.

Before extending: inspect README; keep consent/privacy constraints; split by identity/source/person/session before train/eval and guard against burst/duplicate leakage; do not claim accuracy without representative tests. Validate APIs/tests (`pytest`), improve calibration/open-set metrics as data become available, and never include dataset/model files in repository ZIP. Static page is served separately in repo; mount `static/` if serving through FastAPI is desired. Exact run instructions are in README. `joblib` artifacts are trusted-only.
