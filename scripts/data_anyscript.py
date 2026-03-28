"""
ICDAR 2026 — Long-Term Handwriting Author Identification (AnyScript Challenge)

Query-by-example retrieval: given a query page or book, rank training items by same author.
Tracks: intra-book (query pages → training pages) and extra-book (query books → training books).
Submission CSV columns: query_document_id, retrieved_document_id, similarity_score

This module expects an *extracted* dataset tree: {data_root}/{author_id}/{book_id}/<page images>.
The official release is often shipped as a .tar.gz; extract it locally, then point build_records()
at that folder (or set ANYSCRIPT_DATA_ROOT).
"""

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# Triplet dataset and image transforms live in data_anyscript_vision.py so this module stays
# importable without PyTorch (e.g. export_anyscript_submission.py --help, id utilities).

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Official submission header (ICDAR 2026 AnyScript). Intra-book: page IDs; extra-book: book IDs.
ANYSCRIPT_SUBMISSION_FIELDS = (
    "query_document_id",
    "retrieved_document_id",
    "similarity_score",
)

# Default .tar.gz path only if set here; prefer env ANYSCRIPT_FILTERED_ARCHIVE (see README).
ANYSCRIPT_FILTERED_ARCHIVE_DEFAULT = ""


def anyscript_filtered_archive_path() -> str:
    """Path to AnyScriptFiltered.tar.gz (ICDAR 2026 AnyScript release)."""
    return os.environ.get("ANYSCRIPT_FILTERED_ARCHIVE") or ANYSCRIPT_FILTERED_ARCHIVE_DEFAULT


def suggested_extract_root(archive_path: Optional[str] = None) -> str:
    """Directory name typically used after extracting the .tar.gz (same basename, no extension)."""
    path = archive_path or anyscript_filtered_archive_path()
    lower = path.lower()
    if lower.endswith(".tar.gz"):
        return path[:-7]
    if lower.endswith(".tgz"):
        return path[:-4]
    if lower.endswith(".tar"):
        return path[:-4]
    return path + "_extracted"


def anyscript_data_root() -> str:
    """
    Root folder for extracted author/book/page images (passed to build_records).
    Env ANYSCRIPT_DATA_ROOT overrides; otherwise uses suggested_extract_root() next to the archive.
    """
    override = os.environ.get("ANYSCRIPT_DATA_ROOT")
    if override:
        return override
    return suggested_extract_root()


def write_anyscript_submission_csv(out_path: str, rows: Iterable[Tuple[str, str, float]]) -> None:
    """
    Write submission CSV with required columns only, in order:
    query_document_id, retrieved_document_id, similarity_score.

    Callers must supply the full query×training score table and official IDs; see README.md
    (section “ICDAR 2026 AnyScript — platform submission”).
    Accepts any iterable (e.g. a generator) so dense matrices can be streamed without RAM for all rows.
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ANYSCRIPT_SUBMISSION_FIELDS))
        writer.writeheader()
        for query_id, retrieved_id, score in rows:
            writer.writerow(
                {
                    "query_document_id": query_id,
                    "retrieved_document_id": retrieved_id,
                    "similarity_score": score,
                }
            )


def page_relative_key(data_root: str, page_path: str) -> str:
    """Stable key for ID maps: path of page relative to data root, forward slashes."""
    root = os.path.abspath(data_root)
    page = os.path.abspath(page_path)
    return os.path.relpath(page, root).replace("\\", "/")


def book_key(author_id: str, book_id: str) -> str:
    """Stable key for extra-book ID maps: author_id/book_id."""
    return f"{author_id}/{book_id}"


def write_page_id_map_template(data_root: str, out_path: str, value: str = "") -> None:
    """Write JSON object mapping page_relative_key -> value; fill values from the challenge README/package."""
    records = build_records(data_root)
    d = {page_relative_key(data_root, r.page_path): value for r in records}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def write_book_id_map_template(data_root: str, out_path: str, value: str = "") -> None:
    """Write JSON object mapping book_key -> value for extra-book submissions."""
    records = build_records(data_root)
    keys = sorted({book_key(r.author_id, r.book_id) for r in records})
    d = {k: value for k in keys}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def load_id_assignment(path: str) -> Union[List[str], Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return {str(k).replace("\\", "/"): v for k, v in data.items()}
    raise ValueError(f"Expected JSON list or object in {path}")


def resolve_competition_ids(
    ordered_keys: Sequence[str],
    ids_json: Optional[str],
    *,
    allow_synthetic: bool,
    synthetic_prefix: str,
    role: str,
) -> List[str]:
    """
    Map each item (page rel path or book_key) to the platform document id.

    ids_json: JSON list (same length and order as ordered_keys) or JSON object mapping key -> id.
    """
    if ids_json:
        assign = load_id_assignment(ids_json)
        if isinstance(assign, list):
            if len(assign) != len(ordered_keys):
                raise ValueError(
                    f"{role}: id list length {len(assign)} does not match {len(ordered_keys)} items"
                )
            return list(assign)
        out: List[str] = []
        for k in ordered_keys:
            kk = k.replace("\\", "/")
            if kk not in assign:
                raise KeyError(f"{role}: missing competition id for key {kk!r}")
            out.append(assign[kk])
        return out
    if allow_synthetic:
        return [f"{synthetic_prefix}_{i:06d}" for i in range(len(ordered_keys))]
    raise ValueError(
        f"{role}: provide --*_ids_json with official IDs, or --allow_synthetic_ids for placeholders"
    )


def iter_dense_submission_rows(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    query_chunk: int = 16,
) -> Iterator[Tuple[str, str, float]]:
    """
    Yield every (query_id, gallery_id, similarity) pair. Embeddings should be L2-normalized
    so the score is cosine similarity (dot product).
    """
    g = np.ascontiguousarray(gallery_embs, dtype=np.float32)
    nq = query_embs.shape[0]
    for qs in range(0, nq, query_chunk):
        qe = min(nq, qs + query_chunk)
        block = np.ascontiguousarray(query_embs[qs:qe], dtype=np.float32) @ g.T
        for i in range(qe - qs):
            qi = qs + i
            qid = query_ids[qi]
            row = block[i]
            for gi, gid in enumerate(gallery_ids):
                yield (qid, gid, float(row[gi]))


@dataclass
class PageRecord:
    author_id: str
    book_id: str
    page_path: str


def build_records(data_root: str) -> List[PageRecord]:
    records: List[PageRecord] = []
    for author_id in sorted(os.listdir(data_root)):
        author_dir = os.path.join(data_root, author_id)
        if not os.path.isdir(author_dir):
            continue
        for book_id in sorted(os.listdir(author_dir)):
            book_dir = os.path.join(author_dir, book_id)
            if not os.path.isdir(book_dir):
                continue
            for fn in sorted(os.listdir(book_dir)):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in IMG_EXTS:
                    continue
                page_path = os.path.join(book_dir, fn)
                records.append(PageRecord(author_id=author_id, book_id=book_id, page_path=page_path))
    return records


def group_by_author(records: List[PageRecord]) -> Dict[str, List[PageRecord]]:
    grouped: Dict[str, List[PageRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.author_id, []).append(rec)
    grouped = {k: v for k, v in grouped.items() if len(v) >= 2}
    return grouped


def flatten_records(by_author: Dict[str, List[PageRecord]]) -> List[PageRecord]:
    all_records: List[PageRecord] = []
    for pages in by_author.values():
        all_records.extend(pages)
    return all_records


def random_query_gallery_split(records: List[PageRecord], query_ratio: float = 0.1) -> Tuple[List[PageRecord], List[PageRecord]]:
    shuffled = records[:]
    random.shuffle(shuffled)
    split = max(1, int(len(shuffled) * query_ratio))
    query = shuffled[:split]
    gallery = shuffled[split:]
    return query, gallery
