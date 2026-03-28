import argparse
from typing import Dict, List, Tuple

import faiss
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_path", type=str, required=True)
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--query_embeddings", type=str, required=True)
    p.add_argument("--query_meta", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--target_level", type=str, choices=["author", "book"], default="author")
    return p.parse_args()


def extract_target(meta_row: Tuple[str, str, str], level: str) -> str:
    author_id, book_id, _ = meta_row
    return author_id if level == "author" else f"{author_id}/{book_id}"


def average_precision_binary(relevance: List[int]) -> float:
    hits = 0
    prec_sum = 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            prec_sum += hits / i
    return prec_sum / max(hits, 1)


def main():
    args = parse_args()
    index = faiss.read_index(args.index_path)
    gallery_meta = np.load(args.meta_path, allow_pickle=True)
    query_embs = np.load(args.query_embeddings).astype("float32")
    query_meta = np.load(args.query_meta, allow_pickle=True)

    D, I = index.search(query_embs, args.k)
    del D

    top1 = 0
    top5 = 0
    aps = []

    for qi in range(len(query_embs)):
        qtarget = extract_target(tuple(query_meta[qi]), args.target_level)
        retrieved = [tuple(gallery_meta[idx]) for idx in I[qi]]
        rtargets = [extract_target(x, args.target_level) for x in retrieved]
        rel = [1 if t == qtarget else 0 for t in rtargets]

        top1 += int(rel[0] == 1)
        top5 += int(any(rel[: min(5, len(rel))]))
        aps.append(average_precision_binary(rel))

    n = max(1, len(query_embs))
    metrics: Dict[str, float] = {
        "top1": top1 / n,
        "top5": top5 / n,
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }
    print(metrics)


if __name__ == "__main__":
    main()
