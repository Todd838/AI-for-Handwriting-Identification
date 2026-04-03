import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data_anyscript import (
    PageRecord,
    build_records,
    group_by_author,
    random_query_gallery_split,
)
from data_anyscript_vision import default_transform
from modeling_writer import (
    WriterEmbeddingHead,
    add_vision_backbone_cli_args,
    encode_batch,
    get_backbone_hidden_size,
    load_vision_backbone,
    vision_backbone_kwargs_from_args,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--query_ratio", type=float, default=0.1)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Pages decoded from disk per forward pass (streaming).",
    )
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--no_unsloth", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Seed before query/gallery shuffle so row order matches list-style id JSON across runs.",
    )
    p.add_argument(
        "--random_shuffle",
        action="store_true",
        help="Do not seed RNG; split order changes each run (use dict id JSON, not lists).",
    )
    add_vision_backbone_cli_args(p)
    return p.parse_args()


def embed_records(
    records: List[PageRecord],
    model,
    head,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode images and embed in batches (constant RAM vs. dataset size; scales to huge galleries)."""
    tfm = default_transform(image_size)
    n = len(records)
    d = head.proj[-1].out_features
    if n == 0:
        return np.zeros((0, d), dtype=np.float32), np.array([], dtype=object)

    out = np.empty((n, d), dtype=np.float32)
    meta: List[tuple] = []
    for start in tqdm(range(0, n, batch_size), desc="embedding records"):
        end = min(n, start + batch_size)
        batch_recs = records[start:end]
        tensors = [tfm(Image.open(r.page_path).convert("RGB")) for r in batch_recs]
        batch = torch.stack(tensors, dim=0)
        del tensors
        with torch.no_grad():
            emb = encode_batch(model=model, head=head, images=batch, device=device)
        out[start:end] = emb.cpu().numpy().astype(np.float32)
        for r in batch_recs:
            meta.append((r.author_id, r.book_id, r.page_path))
    return out, np.array(meta, dtype=object)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model_name or ckpt.get("model_name", "THUDM/glm-ocr")
    model, _, _ = load_vision_backbone(
        model_name=model_name,
        load_in_4bit=args.load_in_4bit,
        prefer_unsloth=not args.no_unsloth,
        **vision_backbone_kwargs_from_args(args),
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = get_backbone_hidden_size(model, fallback=1024)
    head = WriterEmbeddingHead(input_dim=hidden_size, embed_dim=args.embed_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    records = build_records(args.data_root)
    by_author = group_by_author(records)
    all_records = [r for pages in by_author.values() for r in pages]
    all_records.sort(key=lambda r: (r.author_id, r.book_id, r.page_path))
    if not args.random_shuffle:
        random.seed(args.shuffle_seed)
    query_records, gallery_records = random_query_gallery_split(all_records, query_ratio=args.query_ratio)

    query_embs, query_meta = embed_records(
        query_records, model, head, args.image_size, args.batch_size, device
    )
    gallery_embs, gallery_meta = embed_records(
        gallery_records, model, head, args.image_size, args.batch_size, device
    )

    np.save(os.path.join(args.out_dir, "query_embs.npy"), query_embs)
    np.save(os.path.join(args.out_dir, "query_meta.npy"), query_meta)
    np.save(os.path.join(args.out_dir, "gallery_embs.npy"), gallery_embs)
    np.save(os.path.join(args.out_dir, "gallery_meta.npy"), gallery_meta)
    print(f"Saved split embeddings/meta to: {args.out_dir}")


if __name__ == "__main__":
    main()
