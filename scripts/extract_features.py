"""
Chunk-based, streaming facial feature extraction with checkpointing and resumability.
Processes dataset manifests in bounded chunks without loading raw images into RAM simultaneously.
"""

import os
import sys
import gc
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageOps

from app.manifest import iter_manifest
from app.embeddings import InsightFaceEmbedder, FacePipelineError
from app.config import INSIGHTFACE_MODEL, FACE_DET_SIZE


def load_cache(cache_path: Path) -> Dict[str, dict]:
    """Load existing checkpoint cache if it exists."""
    if not cache_path.exists():
        return {}
    try:
        data = np.load(cache_path, allow_pickle=True)
        paths = data['paths'].tolist()
        identities = data['identities'].tolist()
        embeddings = data['embeddings']
        cache = {
            p: {'identity': idn, 'embedding': emb}
            for p, idn, emb in zip(paths, identities, embeddings)
        }
        print(f"Loaded existing feature checkpoint with {len(cache)} cached embeddings from {cache_path}")
        return cache
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}. Starting fresh.")
        return {}


def save_cache(cache_path: Path, cache: Dict[str, dict]):
    """Atomically persist feature cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp.npz')
    
    paths = list(cache.keys())
    identities = [cache[p]['identity'] for p in paths]
    embeddings = np.array([cache[p]['embedding'] for p in paths], dtype=np.float32)

    np.savez_compressed(
        tmp_path,
        paths=np.array(paths),
        identities=np.array(identities),
        embeddings=embeddings
    )
    if tmp_path.exists():
        if cache_path.exists():
            cache_path.unlink()
        tmp_path.rename(cache_path)


def extract_features_stream(
    manifest_path: str,
    root_dir: str,
    cache_path: Path,
    chunk_size: int = 256,
    det_size: int = FACE_DET_SIZE,
    prefer_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """
    Stream manifest entries in chunks, extract ArcFace embeddings on GPU,
    checkpoint incrementally, and accumulate a compact embedding matrix.
    """
    embedder = InsightFaceEmbedder(
        model_name=INSIGHTFACE_MODEL,
        det_size=(det_size, det_size),
        prefer_gpu=prefer_gpu
    )
    print(f"Feature Extractor active providers: {embedder.providers} (ctx_id={embedder.ctx_id})")

    cache = load_cache(cache_path)
    audit_stats = {
        'total_images_in_manifest': 0,
        'cached_skipped': 0,
        'successfully_extracted': 0,
        'rejected_no_face': 0,
        'rejected_multi_face': 0,
        'unreadable_or_corrupt': 0,
        'rejections': []
    }

    t_start = time.time()
    chunk_count = 0

    for batch in iter_manifest(manifest_path, root=root_dir, chunk_size=chunk_size):
        chunk_count += 1
        chunk_t0 = time.time()
        new_in_chunk = 0

        for r in batch:
            audit_stats['total_images_in_manifest'] += 1
            rel_path = r['path']

            if rel_path in cache:
                audit_stats['cached_skipped'] += 1
                continue

            abs_path = r['absolute_path']
            identity = r['identity']

            try:
                with Image.open(abs_path) as im:
                    rgb_image = ImageOps.exif_transpose(im).convert('RGB')
                    embedding = embedder.embed(rgb_image)
                    cache[rel_path] = {
                        'identity': identity,
                        'embedding': embedding
                    }
                    new_in_chunk += 1
                    audit_stats['successfully_extracted'] += 1
            except FacePipelineError as fpe:
                err_msg = str(fpe)
                if 'No face' in err_msg:
                    audit_stats['rejected_no_face'] += 1
                elif 'Expected exactly one face' in err_msg:
                    audit_stats['rejected_multi_face'] += 1
                audit_stats['rejections'].append({'path': rel_path, 'identity': identity, 'error': err_msg})
            except Exception as e:
                audit_stats['unreadable_or_corrupt'] += 1
                audit_stats['rejections'].append({'path': rel_path, 'identity': identity, 'error': str(e)})

        # Checkpoint to disk after each chunk if new embeddings were extracted
        if new_in_chunk > 0:
            save_cache(cache_path, cache)

        chunk_time = max(time.time() - chunk_t0, 1e-4)
        rate = len(batch) / chunk_time
        print(
            f"Chunk {chunk_count:02d} processed: {len(batch)} images in {chunk_time:.2f}s "
            f"({rate:.1f} img/s) | Total cached: {len(cache)} | New: {new_in_chunk} | "
            f"NoFace: {audit_stats['rejected_no_face']} | MultiFace: {audit_stats['rejected_multi_face']}",
            flush=True
        )

        # Free memory and trigger garbage collection between chunks
        del batch
        gc.collect()

    total_time = max(time.time() - t_start, 1e-4)
    print(
        f"\nFeature extraction complete in {total_time:.2f}s. "
        f"Total valid embeddings: {len(cache)} / {audit_stats['total_images_in_manifest']}"
    )

    # Convert cached entries to compact NumPy matrices
    paths = list(cache.keys())
    identities = np.array([cache[p]['identity'] for p in paths])
    embeddings = np.array([cache[p]['embedding'] for p in paths], dtype=np.float32)

    return embeddings, identities, paths, audit_stats


def main():
    parser = argparse.ArgumentParser(description="Extract facial embeddings with chunking and caching.")
    parser.add_argument('--manifest', required=True, help="Path to manifest CSV")
    parser.add_argument('--root', default='.', help="Root directory for relative paths")
    parser.add_argument('--cache', required=True, help="Path to save/load .npz cache")
    parser.add_argument('--chunk-size', type=int, default=256, help="Manifest batch size")
    parser.add_argument('--cpu-only', action='store_true', help="Force CPU extraction")
    args = parser.parse_args()

    extract_features_stream(
        manifest_path=args.manifest,
        root_dir=args.root,
        cache_path=Path(args.cache),
        chunk_size=args.chunk_size,
        prefer_gpu=not args.cpu_only
    )


if __name__ == '__main__':
    main()

