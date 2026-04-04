"""
Build dense ICDAR 2026 AnyScript submission CSVs:

- intra_book: every query page x every training page (cosine similarity from embeddings)
- extra_book: every query book x every training page (query book from mean L2-normalized page embeddings)

Official platform IDs come from JSON (list aligned with embedding order, or dict keyed by
page_relative_key / book_key). See README.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from data_anyscript import (
    book_key,
    build_records,
    expand_colab_out_template,
    iter_dense_submission_rows,
    page_relative_key,
    resolve_competition_ids,
    write_anyscript_submission_csv,
)
from modeling_writer import add_vision_backbone_cli_args


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["intra_book", "extra_book"], required=True)
    p.add_argument("--out_csv", type=str, required=True)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Directory with query_embs.npy, query_meta.npy, gallery_embs.npy, gallery_meta.npy",
    )
    src.add_argument("--checkpoint", type=str, default=None, help="Embed pages from checkpoint")

    p.add_argument("--gallery_data_root", type=str, default=None, help="Training tree (checkpoint mode)")
    p.add_argument("--query_data_root", type=str, default=None, help="Held-out query tree (checkpoint mode)")

    p.add_argument(
        "--gallery_key_root",
        type=str,
        default=None,
        help="Root for page_relative_key() on gallery pages; default gallery_data_root or required with --embeddings_dir",
    )
    p.add_argument(
        "--query_key_root",
        type=str,
        default=None,
        help="Root for page_relative_key() on query pages; default query_data_root or required with --embeddings_dir",
    )

    p.add_argument("--query_ids_json", type=str, default=None)
    p.add_argument("--gallery_ids_json", type=str, default=None)
    p.add_argument(
        "--allow_synthetic_ids",
        action="store_true",
        help="Placeholder query_*/train_* ids when official JSON is not available yet",
    )

    p.add_argument("--query_chunk", type=int, default=16, help="Query batch size for similarity blocks")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--no_unsloth", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    add_vision_backbone_cli_args(p)
    return p.parse_args()


def aggregate_book_embeddings(page_embs: np.ndarray, meta: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    by_book: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(meta)):
        a, b, _ = meta[i]
        by_book[book_key(str(a), str(b))].append(i)
    keys = sorted(by_book.keys())
    stacks = []
    for k in keys:
        idxs = by_book[k]
        v = page_embs[idxs].mean(axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            v = v / n
        stacks.append(v)
    return np.stack(stacks, axis=0), keys


def meta_page_keys(meta: np.ndarray, key_root: str) -> List[str]:
    out: List[str] = []
    for i in range(len(meta)):
        _, _, page_path = meta[i]
        out.append(page_relative_key(key_root, str(page_path)))
    return out


def load_checkpoint_bundle(args):
    import torch
    from export_embeddings_split import embed_records
    from modeling_writer import (
        WriterEmbeddingHead,
        get_backbone_hidden_size,
        load_vision_backbone,
        vision_backbone_kwargs_from_args,
        vision_uses_glm_image_processor,
    )

    if not args.gallery_data_root or not args.query_data_root:
        raise ValueError("--checkpoint requires --gallery_data_root and --query_data_root")
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

    gallery_records = build_records(args.gallery_data_root)
    query_records = build_records(args.query_data_root)
    if not gallery_records:
        raise ValueError("No pages under gallery_data_root")
    if not query_records:
        raise ValueError("No pages under query_data_root")

    qe, qm = embed_records(
        query_records,
        model,
        head,
        args.image_size,
        args.batch_size,
        device,
        hf_processor=hf_processor,
        use_glm=use_glm,
    )
    ge, gm = embed_records(
        gallery_records,
        model,
        head,
        args.image_size,
        args.batch_size,
        device,
        hf_processor=hf_processor,
        use_glm=use_glm,
    )
    return qe, qm, ge, gm


def load_embeddings_dir(d: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qe = np.load(os.path.join(d, "query_embs.npy"))
    qm = np.load(os.path.join(d, "query_meta.npy"), allow_pickle=True)
    ge = np.load(os.path.join(d, "gallery_embs.npy"))
    gm = np.load(os.path.join(d, "gallery_meta.npy"), allow_pickle=True)
    return qe, qm, ge, gm


def main():
    args = parse_args()
    args.out_csv = expand_colab_out_template(args.out_csv)
    if args.checkpoint:
        args.checkpoint = expand_colab_out_template(args.checkpoint)
    if args.gallery_data_root:
        args.gallery_data_root = expand_colab_out_template(args.gallery_data_root)
    if args.query_data_root:
        args.query_data_root = expand_colab_out_template(args.query_data_root)
    if args.embeddings_dir:
        args.embeddings_dir = expand_colab_out_template(args.embeddings_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)

    if args.embeddings_dir:
        qe, qm, ge, gm = load_embeddings_dir(args.embeddings_dir)
        if args.gallery_key_root is None or args.query_key_root is None:
            raise ValueError(
                "With --embeddings_dir, set --gallery_key_root and --query_key_root "
                "so page paths in meta match competition id map keys (see README)."
            )
        gallery_key_root = args.gallery_key_root
        query_key_root = args.query_key_root
    else:
        qe, qm, ge, gm = load_checkpoint_bundle(args)
        gallery_key_root = args.gallery_key_root or args.gallery_data_root
        query_key_root = args.query_key_root or args.query_data_root
        assert gallery_key_root and query_key_root

    if args.mode == "intra_book":
        q_embs = qe
        g_embs = ge
        q_keys = meta_page_keys(qm, query_key_root)
        g_keys = meta_page_keys(gm, gallery_key_root)
        q_prefix, g_prefix = "query_page", "train_page"
    else:
        q_embs, q_keys = aggregate_book_embeddings(qe, qm)
        g_embs = ge
        g_keys = meta_page_keys(gm, gallery_key_root)
        q_prefix, g_prefix = "query_book", "train_page"

    query_ids = resolve_competition_ids(
        q_keys,
        args.query_ids_json,
        allow_synthetic=args.allow_synthetic_ids,
        synthetic_prefix=q_prefix,
        role="query",
    )
    gallery_ids = resolve_competition_ids(
        g_keys,
        args.gallery_ids_json,
        allow_synthetic=args.allow_synthetic_ids,
        synthetic_prefix=g_prefix,
        role="gallery",
    )

    rows = iter_dense_submission_rows(
        q_embs,
        g_embs,
        query_ids,
        gallery_ids,
        query_chunk=args.query_chunk,
    )
    write_anyscript_submission_csv(args.out_csv, rows)
    nq, ng = q_embs.shape[0], g_embs.shape[0]
    print(f"Wrote {args.out_csv} ({args.mode}: {nq} queries x {ng} gallery = {nq * ng} rows)")


if __name__ == "__main__":
    main()
