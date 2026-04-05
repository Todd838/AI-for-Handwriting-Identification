import argparse
import os
from typing import List

import faiss
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
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Pages decoded from disk per forward pass (streaming; lower if GPU OOM).",
    )
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--index_out", type=str, required=True)
    p.add_argument("--meta_out", type=str, required=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --index_out/--meta_out if present (same data_root/filter required).",
    )
    p.add_argument(
        "--save_every_batches",
        type=int,
        default=25,
        help="Persist partial index/meta every N batches (0 disables periodic saves).",
    )
    p.add_argument(
        "--all_pages",
        action="store_true",
        help="Index every page from data_root. Default: only authors with 2+ pages (triplet training filter).",
    )
    p.add_argument("--no_unsloth", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    add_vision_backbone_cli_args(p)
    return p.parse_args()


def _reject_placeholder_paths(pairs) -> None:
    """Colab `!python ... {OUT}/best.pt` passes braces literally — catch before torch.load."""
    for label, p in pairs:
        if p and ("{" in p or "}" in p):
            raise ValueError(
                f"{label} looks like an unexpanded template: {p!r}. "
                "Shell lines starting with `!` do not substitute Python variables. "
                "For --data_root use the word auto or a full path (not {{DATA_ROOT}}). "
                "For checkpoint and outputs use real paths or a Python subprocess with f-strings."
            )


def _save_progress(index, meta, index_out: str, meta_out: str) -> None:
    """Persist partial progress so long Colab runs can resume after disconnect."""
    faiss.write_index(index, index_out)
    np.save(meta_out, np.array(meta, dtype=object))


def main():
    args = parse_args()
    # Must run before _reject: resolve_training_data_root also coerces, but runs later.
    args.data_root = coerce_cli_data_root(args.data_root)
    args.checkpoint = expand_colab_out_template(args.checkpoint)
    args.index_out = expand_colab_out_template(args.index_out)
    args.meta_out = expand_colab_out_template(args.meta_out)
    _reject_placeholder_paths(
        (
            ("--data_root", args.data_root),
            ("--checkpoint", args.checkpoint),
            ("--index_out", args.index_out),
            ("--meta_out", args.meta_out),
        )
    )
    args.data_root = resolve_training_data_root(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.index_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)

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
        all_records: List[PageRecord] = list(records)
    else:
        by_author = group_by_author(records)
        all_records = [r for pages in by_author.values() for r in pages]
    print(
        f"Indexing {len(all_records)} pages "
        f"({'all' if args.all_pages else 'authors with 2+ pages only'})"
    )
    tfm = default_transform(args.image_size)
    n = len(all_records)
    if n == 0:
        raise ValueError("No pages found to index under resolved data_root.")

    index = faiss.IndexFlatIP(args.embed_dim)
    meta: List[tuple] = []
    start = 0
    if args.resume and os.path.isfile(args.index_out) and os.path.isfile(args.meta_out):
        index = faiss.read_index(args.index_out)
        loaded_meta = np.load(args.meta_out, allow_pickle=True).tolist()
        meta = [tuple(x) for x in loaded_meta]
        start = len(meta)
        if index.ntotal != start:
            raise ValueError(
                f"Resume mismatch: index vectors={index.ntotal} but meta rows={start}. "
                "Delete stale outputs or disable --resume."
            )
        if start > n:
            raise ValueError(
                f"Resume mismatch: existing meta rows ({start}) exceed records to index ({n}). "
                "Delete stale outputs or disable --resume."
            )
        if start > 0:
            last = meta[-1]
            expected = all_records[start - 1]
            expected_last = (expected.author_id, expected.book_id, expected.page_path)
            if last != expected_last:
                raise ValueError(
                    "Resume mismatch: last stored metadata row does not match current record order. "
                    "Use the same --data_root/--all_pages setting, or delete stale outputs."
                )
        print(f"Resuming from {start}/{n} pages already indexed.")

    if start >= n:
        print("Index already complete; nothing to do.")
        print(f"Saved FAISS index: {args.index_out}")
        print(f"Saved metadata: {args.meta_out}")
        return

    batch_count = 0
    for start in tqdm(range(start, n, args.batch_size), desc="embedding batches"):
        end = min(n, start + args.batch_size)
        batch_recs = all_records[start:end]
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
        index.add(emb.cpu().numpy().astype("float32"))
        for r in batch_recs:
            meta.append((r.author_id, r.book_id, r.page_path))
        batch_count += 1
        if args.save_every_batches > 0 and (batch_count % args.save_every_batches == 0):
            _save_progress(index, meta, args.index_out, args.meta_out)
            print(f"Checkpointed progress: {len(meta)}/{n} pages")

    _save_progress(index, meta, args.index_out, args.meta_out)
    print(f"Saved FAISS index: {args.index_out}")
    print(f"Saved metadata: {args.meta_out}")


if __name__ == "__main__":
    main()
