from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    metadata_csv = Path(cfg["data"]["metadata_csv"])
    embeddings_path = Path(cfg["data"]["embeddings_path"])
    encoder_name = cfg["text"]["encoder_name"]

    df = pd.read_csv(metadata_csv)
    prompts = sorted(df["prompt"].astype(str).unique().tolist())

    print(f"Loading frozen text encoder: {encoder_name}")
    model = SentenceTransformer(encoder_name, device="cpu")
    embeddings = {}
    for prompt in tqdm(prompts, desc="Encoding prompts"):
        emb = model.encode(prompt, convert_to_tensor=True, normalize_embeddings=True).cpu()
        embeddings[prompt] = emb

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, embeddings_path)
    print(f"Saved {len(embeddings)} text embeddings to {embeddings_path}")


if __name__ == "__main__":
    main()
