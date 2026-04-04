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
    coerce_cli_data_root,
    expand_colab_out_template,
    group_by_author,
    random_query_gallery_split,
    resolve_training_data_root,
)
from data_anyscript_vision import default_transform
from modeling_writer import (
    WriterEmbeddingHead,
    add_vision_backbone_cli_args,
    encode_batch,
    get_backbone_hidden_size,
    glm_vision_inputs_from_pils,
    load_vision_backbone,
    vision_backbone_kwargs_from_args,
    vision_uses_glm_image_processor,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Folder whose children are author IDs, or 'auto' (Colab: search Drive in-process).",
    )
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
    p.add_argument(
        "--all_pages",
        action="store_true",
        help="Embed every page from data_root before query/gallery split. Default: only authors with 2+ pages.",
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
    hf_processor=None,
    use_glm: bool = False,
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
        pils = [Image.open(r.page_path).convert("RGB") for r in batch_recs]
        if use_glm:
            batch, grid = glm_vision_inputs_from_pils(hf_processor, pils)
        else:
            tensors = [tfm(p) for p in pils]
            batch = torch.stack(tensors, dim=0)
            del tensors
            grid = None
        del pils
        with torch.no_grad():
            emb = encode_batch(
                model=model, head=head, images=batch, device=device, image_grid_thw=grid
            )
        out[start:end] = emb.cpu().numpy().astype(np.float32)
        for r in batch_recs:
            meta.append((r.author_id, r.book_id, r.page_path))
    return out, np.array(meta, dtype=object)


def main():
    args = parse_args()
    args.data_root = coerce_cli_data_root(args.data_root)
    args.checkpoint = expand_colab_out_template(args.checkpoint)
    args.out_dir = expand_colab_out_template(args.out_dir)
    args.data_root = resolve_training_data_root(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model_name or ckpt.get("model_name", "zai-org/GLM-OCR")
    model, hf_processor, _ = load_vision_backbone(
        model_name=model_name,
        load_in_4bit=args.load_in_4bit,
        prefer_unsloth=not args.no_unsloth,
        **vision_backbone_kwargs_from_args(args),
    )
    model = model.to(device)
    use_glm = vision_uses_glm_image_processor(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = get_backbone_hidden_size(model, fallback=1024)
    head = WriterEmbeddingHead(input_dim=hidden_size, embed_dim=args.embed_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    records = build_records(args.data_root)
    if args.all_pages:
        all_records = list(records)
    else:
        by_author = group_by_author(records)
        all_records = [r for pages in by_author.values() for r in pages]
    all_records.sort(key=lambda r: (r.author_id, r.book_id, r.page_path))
    if not args.random_shuffle:
        random.seed(args.shuffle_seed)
    query_records, gallery_records = random_query_gallery_split(all_records, query_ratio=args.query_ratio)

    query_embs, query_meta = embed_records(
        query_records,
        model,
        head,
        args.image_size,
        args.batch_size,
        device,
        hf_processor=hf_processor,
        use_glm=use_glm,
    )
    gallery_embs, gallery_meta = embed_records(
        gallery_records,
        model,
        head,
        args.image_size,
        args.batch_size,
        device,
        hf_processor=hf_processor,
        use_glm=use_glm,
    )

    np.save(os.path.join(args.out_dir, "query_embs.npy"), query_embs)
    np.save(os.path.join(args.out_dir, "query_meta.npy"), query_meta)
    np.save(os.path.join(args.out_dir, "gallery_embs.npy"), gallery_embs)
    np.save(os.path.join(args.out_dir, "gallery_meta.npy"), gallery_meta)
    print(f"Saved split embeddings/meta to: {args.out_dir}")


if __name__ == "__main__":
    main()
