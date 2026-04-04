import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_anyscript import build_records, group_by_author
from data_anyscript_vision import TripletPageDataset, default_transform
from modeling_writer import (
    WriterEmbeddingHead,
    add_vision_backbone_cli_args,
    extract_pooled_features,
    get_backbone_hidden_size,
    load_vision_backbone,
    maybe_apply_lora,
    normalize_glm_ocr_hub_id,
    triplet_loss,
    vision_backbone_kwargs_from_args,
)


def resolve_training_data_root(cli_value: str) -> str:
    """
    Resolve dataset path. Use ``auto`` on Colab to search Drive inside this process
    (avoids stale ``DATA_ROOT`` from an old notebook kernel).
    """
    if cli_value != "auto":
        return cli_value
    env = os.environ.get("ANYSCRIPT_DATA_ROOT", "").strip()
    if env and os.path.isdir(env):
        print(f"[train] ANYSCRIPT_DATA_ROOT -> {env!r}")
        return env
    try:
        from data_anyscript import resolve_colab_data_root_any
    except ImportError:
        resolve_colab_data_root_any = None  # type: ignore
    if resolve_colab_data_root_any:
        found = resolve_colab_data_root_any()
        if found:
            print(f"[train] --data_root auto -> {found!r}")
            return found
    raise ValueError(
        "--data_root auto: no triplet-usable tree found on Colab Drive. "
        "Official extract is often .../AnyScriptFiltered/binarized/train (authors live under train/). "
        "Upload/extract the dataset, run inspect_anyscript_layout.py, set ANYSCRIPT_DATA_ROOT to the "
        "path it suggests, or: python /content/ai-hw/scripts/diagnose_data_root.py"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Folder whose children are author IDs, or 'auto' (Colab: search Drive in-process).",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="zai-org/GLM-OCR",
        help="HF vision model id. Public GLM-OCR: zai-org/GLM-OCR (THUDM/glm-ocr often 401 without access).",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps_per_epoch", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_unsloth", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--unfreeze_backbone", action="store_true")
    p.add_argument(
        "--max_wall_time_hours",
        type=float,
        default=None,
        help="Stop training after this many wall-clock hours from start of the epoch loop; saves walltime_stop.pt",
    )
    add_vision_backbone_cli_args(p)
    return p.parse_args()


def collate_fn(batch):
    anchors = torch.stack([x["anchor"] for x in batch], dim=0)
    positives = torch.stack([x["positive"] for x in batch], dim=0)
    negatives = torch.stack([x["negative"] for x in batch], dim=0)
    return anchors, positives, negatives


def main():
    args = parse_args()
    args.data_root = resolve_training_data_root(args.data_root)
    args.model_name = normalize_glm_ocr_hub_id(args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = build_records(args.data_root)
    by_author = group_by_author(records)
    if len(by_author) < 2:
        counts: dict = {}
        for r in records:
            counts[r.author_id] = counts.get(r.author_id, 0) + 1
        n_auth_any = len(counts)
        n_auth_multi = sum(1 for c in counts.values() if c >= 2)
        raise ValueError(
            "Need at least 2 authors with at least 2 pages each for triplet sampling. "
            f"data_root={args.data_root!r}: {len(records)} pages found; "
            f"{n_auth_any} authors with any page; {n_auth_multi} authors with 2+ pages. "
            "Point --data_root at the folder whose direct children are author IDs "
            "(each folder holds page images or book subfolders with images). "
            "On Colab try: --data_root auto  OR  "
            "python /content/ai-hw/scripts/diagnose_data_root.py"
        )

    ds = TripletPageDataset(
        by_author=by_author,
        transform=default_transform(args.image_size),
        steps_per_epoch=args.steps_per_epoch,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    backbone, _, loader_name = load_vision_backbone(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        prefer_unsloth=not args.no_unsloth,
        **vision_backbone_kwargs_from_args(args),
    )
    backbone = backbone.to(device)

    if args.use_lora:
        backbone = maybe_apply_lora(backbone, lora_r=args.lora_r)

    hidden_size = get_backbone_hidden_size(backbone, fallback=1024)
    head = WriterEmbeddingHead(input_dim=hidden_size, embed_dim=args.embed_dim).to(device)

    if not args.unfreeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    params = list(head.parameters()) + [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    best_loss = float("inf")
    history = []
    max_seconds = (
        args.max_wall_time_hours * 3600.0 if args.max_wall_time_hours is not None else None
    )
    t_train_start = time.perf_counter()
    stopped_for_walltime = False

    def save_ckpt(epoch_idx: int, loss_for_ckpt: float, tag: str):
        nonlocal best_loss
        ckpt = {
            "epoch": epoch_idx,
            "loss": loss_for_ckpt,
            "loader": loader_name,
            "model_name": args.model_name,
            "head_state_dict": head.state_dict(),
            "config": vars(args),
        }
        if tag == "epoch":
            torch.save(ckpt, os.path.join(args.output_dir, f"epoch_{epoch_idx}.pt"))
            if loss_for_ckpt < best_loss:
                best_loss = loss_for_ckpt
                torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
        else:
            ckpt["stopped_reason"] = tag
            torch.save(ckpt, os.path.join(args.output_dir, "walltime_stop.pt"))

    for epoch in range(args.epochs):
        backbone.train(args.unfreeze_backbone)
        head.train()
        running = 0.0
        n_steps = 0
        last_loss = 0.0

        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{args.epochs}")
        for anchors, positives, negatives in pbar:
            if max_seconds is not None and (time.perf_counter() - t_train_start) >= max_seconds:
                save_ckpt(epoch + 1, last_loss, "max_wall_time_hours")
                stopped_for_walltime = True
                history.append(
                    {
                        "epoch": epoch + 1,
                        "loss": None,
                        "stopped_early": True,
                        "reason": "max_wall_time_hours",
                        "hours_elapsed": (time.perf_counter() - t_train_start) / 3600.0,
                    }
                )
                break

            anchors = anchors.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True)

            with torch.set_grad_enabled(args.unfreeze_backbone):
                feat_a = extract_pooled_features(backbone, anchors)
                feat_p = extract_pooled_features(backbone, positives)
                feat_n = extract_pooled_features(backbone, negatives)

            emb_a = head(feat_a)
            emb_p = head(feat_p)
            emb_n = head(feat_n)
            loss = triplet_loss(emb_a, emb_p, emb_n, margin=args.margin)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            running += last_loss
            n_steps += 1
            pbar.set_postfix(loss=f"{last_loss:.4f}")

        if stopped_for_walltime:
            break

        epoch_loss = running / max(1, n_steps)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})

        save_ckpt(epoch + 1, epoch_loss, "epoch")

    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if stopped_for_walltime:
        print(
            f"Stopped after {args.max_wall_time_hours} h wall time. "
            f"See walltime_stop.pt and train_history.json"
        )
    print(f"Done. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
