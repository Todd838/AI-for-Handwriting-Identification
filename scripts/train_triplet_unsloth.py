import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_anyscript import build_records, group_by_author
from data_anyscript_vision import TripletPageDataset, default_transform
from modeling_writer import (
    WriterEmbeddingHead,
    extract_pooled_features,
    get_backbone_hidden_size,
    load_vision_backbone,
    maybe_apply_lora,
    triplet_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--model_name", type=str, default="THUDM/glm-ocr")
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
    return p.parse_args()


def collate_fn(batch):
    anchors = torch.stack([x["anchor"] for x in batch], dim=0)
    positives = torch.stack([x["positive"] for x in batch], dim=0)
    negatives = torch.stack([x["negative"] for x in batch], dim=0)
    return anchors, positives, negatives


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = build_records(args.data_root)
    by_author = group_by_author(records)
    if len(by_author) < 2:
        raise ValueError("Need at least 2 authors (with >=2 pages each) in data_root.")

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

    for epoch in range(args.epochs):
        backbone.train(args.unfreeze_backbone)
        head.train()
        running = 0.0

        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{args.epochs}")
        for anchors, positives, negatives in pbar:
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

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running / max(1, len(dl))
        history.append({"epoch": epoch + 1, "loss": epoch_loss})

        ckpt = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "loader": loader_name,
            "model_name": args.model_name,
            "head_state_dict": head.state_dict(),
            "config": vars(args),
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"epoch_{epoch + 1}.pt"))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))

    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
