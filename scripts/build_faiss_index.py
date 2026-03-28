import argparse
import os
from typing import List

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data_anyscript import PageRecord, build_records, group_by_author
from data_anyscript_vision import default_transform
from modeling_writer import WriterEmbeddingHead, encode_batch, get_backbone_hidden_size, load_vision_backbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--index_out", type=str, required=True)
    p.add_argument("--meta_out", type=str, required=True)
    p.add_argument("--no_unsloth", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    return p.parse_args()


def _load_pages(records: List[PageRecord], image_size: int):
    tfm = default_transform(image_size)
    imgs = []
    meta = []
    for r in records:
        img = Image.open(r.page_path).convert("RGB")
        imgs.append(tfm(img))
        meta.append((r.author_id, r.book_id, r.page_path))
    return imgs, meta


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.index_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = args.model_name or ckpt.get("model_name", "THUDM/glm-ocr")
    model, _, _ = load_vision_backbone(
        model_name=model_name,
        load_in_4bit=args.load_in_4bit,
        prefer_unsloth=not args.no_unsloth,
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
    all_records: List[PageRecord] = [r for pages in by_author.values() for r in pages]
    imgs, meta = _load_pages(all_records, image_size=args.image_size)

    embeddings = []
    for i in tqdm(range(0, len(imgs), args.batch_size), desc="embedding pages"):
        batch = torch.stack(imgs[i : i + args.batch_size], dim=0)
        with torch.no_grad():
            emb = encode_batch(model=model, head=head, images=batch, device=device)
        embeddings.append(emb.cpu().numpy())

    embs = np.concatenate(embeddings, axis=0).astype("float32")
    index = faiss.IndexFlatIP(args.embed_dim)
    index.add(embs)

    faiss.write_index(index, args.index_out)
    np.save(args.meta_out, np.array(meta, dtype=object))
    print(f"Saved FAISS index: {args.index_out}")
    print(f"Saved metadata: {args.meta_out}")


if __name__ == "__main__":
    main()
