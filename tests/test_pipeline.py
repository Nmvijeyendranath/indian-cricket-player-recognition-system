import os
import tempfile
import pytest
import numpy as np
import joblib
from pathlib import Path
from fastapi.testclient import TestClient

from scripts.prepare_dataset import clean_player_name, is_all_null_bytes
from scripts.extract_features import save_cache, load_cache
from app.classifier import IdentityClassifier
from app.main import app


def test_player_name_normalization():
    assert clean_player_name("M.S. Dhoni1") == "M.S. Dhoni"
    assert clean_player_name("Rahul Dravid2") == "Rahul Dravid"
    assert clean_player_name("Virat Kohli") == "Virat Kohli"
    assert clean_player_name("Sachin Tendulkar") == "Sachin Tendulkar"


def test_null_byte_detection(tmp_path):
    null_file = tmp_path / "corrupt.jpg"
    null_file.write_bytes(b"\x00" * 4096)
    assert is_all_null_bytes(null_file) is True

    valid_file = tmp_path / "valid.jpg"
    valid_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    assert is_all_null_bytes(valid_file) is False


def test_classifier_fit_and_rejection():
    # Synthetic 512D unit vectors for 3 classes
    np.random.seed(42)
    dim = 512
    n_per_class = 20

    # Cluster centroids
    c0 = np.random.randn(dim)
    c0 /= np.linalg.norm(c0)
    c1 = np.random.randn(dim)
    c1 /= np.linalg.norm(c1)
    c2 = np.random.randn(dim)
    c2 /= np.linalg.norm(c2)

    X0 = c0 + np.random.randn(n_per_class, dim) * 0.05
    X1 = c1 + np.random.randn(n_per_class, dim) * 0.05
    X2 = c2 + np.random.randn(n_per_class, dim) * 0.05

    X = np.vstack([X0, X1, X2])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array(['Player_A'] * n_per_class + ['Player_B'] * n_per_class + ['Player_C'] * n_per_class)

    clf = IdentityClassifier(threshold=0.60)
    clf.fit(X, y, cv=3)

    assert len(clf.classes_) == 3

    # Prediction on close sample
    res_known = clf.predict(c0, top_k=3)
    assert res_known['identity'] == 'Player_A'
    assert res_known['confidence'] > 0.60
    assert res_known['unknown'] is False
    assert len(res_known['top_k']) == 3

    # Prediction on orthogonal/unknown random noise
    random_noise = np.random.randn(dim)
    random_noise /= np.linalg.norm(random_noise)
    res_unknown = clf.predict(random_noise, top_k=3)
    assert res_unknown['confidence'] < 0.85
    assert 'unknown' in res_unknown


def test_classifier_serialization(tmp_path):
    X = np.random.randn(30, 512).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array(['Player_A'] * 15 + ['Player_B'] * 15)

    clf = IdentityClassifier(threshold=0.72, metadata={'version': '1.0.0'})
    clf.fit(X, y, cv=2)

    model_file = tmp_path / "model.joblib"
    joblib.dump(clf, model_file)

    loaded = joblib.load(model_file)
    assert loaded.threshold == 0.72
    assert loaded.metadata['version'] == '1.0.0'
    assert list(loaded.classes_) == ['Player_A', 'Player_B']


def test_feature_caching_and_atomic_save(tmp_path):
    cache_file = tmp_path / "test_cache.npz"
    cache_data = {
        'img1.jpg': {'identity': 'Player_A', 'embedding': np.zeros(512, dtype=np.float32)},
        'img2.jpg': {'identity': 'Player_B', 'embedding': np.ones(512, dtype=np.float32)}
    }
    save_cache(cache_file, cache_data)
    assert cache_file.exists()

    loaded = load_cache(cache_file)
    assert len(loaded) == 2
    assert 'img1.jpg' in loaded
    assert np.allclose(loaded['img1.jpg']['embedding'], np.zeros(512))


def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data


def test_api_invalid_image_upload():
    client = TestClient(app)
    response = client.post(
        "/identify",
        files={"file": ("fake.jpg", b"not-an-image-content", "image/jpeg")}
    )
    assert response.status_code in (400, 422)

