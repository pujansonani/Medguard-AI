#!/usr/bin/env python3
"""
MEDGUARD AI: Clinical Text Note Encoder & Embedding Cache

Pre-encodes unstructured clinical text using ClinicalBERT / Transformer models.
Caches embeddings to data/cache/text_embeddings/ to prevent redundant tokenization and forward passes.

Usage:
    python scripts/encode_notes.py --mode standard
    python scripts/encode_notes.py --device cpu --batch-size 32
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.models.clinical_nlp import ClinicalNLPModel
from medguard.utils.config import get_project_root, load_all_configs
from medguard.utils.logging import get_logger

logger = get_logger("medguard.encode_notes")


def get_cache_key(model_name: str, texts: list[str], max_len: int = 256) -> str:
    """Generate deterministic MD5 hash key for cached embeddings."""
    hasher = hashlib.md5()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(max_len).encode("utf-8"))
    for t in texts[:50]: # Sample hash
        hasher.update(t.encode("utf-8"))
    hasher.update(str(len(texts)).encode("utf-8"))
    return hasher.hexdigest()[:16]


def encode_and_cache_notes(
    proc_dir: Path,
    cache_dir: Path,
    device: str = "cpu",
    batch_size: int = 32,
    mode: str = "standard",
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Texts
    train_f = proc_dir / "texts_train.json"
    val_f = proc_dir / "texts_val.json"
    test_f = proc_dir / "texts_test.json"

    if not train_f.exists():
        logger.error(f"Text files not found in {proc_dir}. Run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    with open(train_f, "r") as f:
        texts_train = json.load(f)
    with open(val_f, "r") as f:
        texts_val = json.load(f)
    with open(test_f, "r") as f:
        texts_test = json.load(f)

    logger.info(f"Loaded {len(texts_train)} train, {len(texts_val)} val, {len(texts_test)} test note sets.")

    # 2. Initialize Model
    model_name = "ClinicalBERT"
    nlp_model = ClinicalNLPModel(device=device, batch_size=batch_size)

    # 3. Check / Compute Train Embeddings
    train_key = get_cache_key(model_name, texts_train)
    train_cache_file = cache_dir / f"emb_train_{train_key}.npy"

    if train_cache_file.exists():
        logger.info(f"Loaded cached train embeddings from {train_cache_file}")
        emb_train = np.load(train_cache_file)
    else:
        logger.info(f"Computing embeddings for {len(texts_train)} training notes (Device: {device})...")
        emb_train = nlp_model.encode_texts(texts_train)
        np.save(train_cache_file, emb_train)
        logger.info(f"Saved train embeddings to {train_cache_file} (Shape: {emb_train.shape})")

    # 4. Check / Compute Val & Test Embeddings
    val_key = get_cache_key(model_name, texts_val)
    val_cache_file = cache_dir / f"emb_val_{val_key}.npy"
    if val_cache_file.exists():
        emb_val = np.load(val_cache_file)
    else:
        emb_val = nlp_model.encode_texts(texts_val)
        np.save(val_cache_file, emb_val)

    test_key = get_cache_key(model_name, texts_test)
    test_cache_file = cache_dir / f"emb_test_{test_key}.npy"
    if test_cache_file.exists():
        emb_test = np.load(test_cache_file)
    else:
        emb_test = nlp_model.encode_texts(texts_test)
        np.save(test_cache_file, emb_test)

    # Save manifest
    manifest = {
        "model_name": model_name,
        "train_key": train_key,
        "val_key": val_key,
        "test_key": test_key,
        "train_shape": list(emb_train.shape),
        "val_shape": list(emb_val.shape),
        "test_shape": list(emb_test.shape),
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Clinical text embedding cache ready under {cache_dir.resolve()}!")


def main():
    parser = argparse.ArgumentParser(description="Encode clinical notes and cache embeddings.")
    parser.add_argument("--mode", type=str, default="standard", choices=["quick", "standard", "full"])
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = get_project_root()
    configs = load_all_configs(root)
    cfg = configs["global"]

    # Detect hardware
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    proc_dir = root / cfg.get("paths", {}).get("processed_dir", "data/processed")
    cache_dir = root / "data" / "cache" / "text_embeddings"

    encode_and_cache_notes(
        proc_dir=proc_dir,
        cache_dir=cache_dir,
        device=device,
        batch_size=args.batch_size,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
