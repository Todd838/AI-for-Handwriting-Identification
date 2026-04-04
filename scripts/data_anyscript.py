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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

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

# Local AnyScript filtered archive. Override with env ANYSCRIPT_FILTERED_ARCHIVE.
ANYSCRIPT_FILTERED_ARCHIVE_DEFAULT = r"C:\Users\thisb\Downloads\AnyScriptFiltered.tar.gz"


def anyscript_filtered_archive_path() -> str:
    """Path to AnyScriptFiltered.tar.gz (ICDAR 2026 AnyScript release)."""
    return os.environ.get("ANYSCRIPT_FILTERED_ARCHIVE", ANYSCRIPT_FILTERED_ARCHIVE_DEFAULT)


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
    """
    Expects either:
    - {root}/{author}/{book}/<images> (README layout), or
    - {root}/{author}/<images> (flat release: one implicit book per author).
    """
    records: List[PageRecord] = []
    implicit_book = "__default_book__"
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
        for fn in sorted(os.listdir(author_dir)):
            path = os.path.join(author_dir, fn)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMG_EXTS:
                continue
            records.append(
                PageRecord(author_id=author_id, book_id=implicit_book, page_path=path)
            )
    return records


def looks_like_anyscript_related_path(path: str) -> bool:
    """Heuristic: path is probably this challenge's data, not another Kaggle project."""
    norm = path.replace("\\", "/").lower()
    keys = (
        "anyscript",
        "binarized",
        "icdar",
        "handwrit",
        "writer_id",
        "writer-id",
        "datasets/anyscript",
    )
    return any(k in norm for k in keys)


def colab_anyscript_archive_candidates(drive_my_drive: str = "/content/drive/MyDrive") -> List[str]:
    """Where users often upload AnyScriptFiltered.tar.gz on Colab (try in order)."""
    d = drive_my_drive.rstrip(os.sep)
    return [
        f"{d}/AnyScriptFiltered/AnyScriptFiltered.tar.gz",
        f"{d}/AnyScriptFiltered.tar.gz",
    ]


def colab_drive_data_root_candidates(drive_my_drive: str = "/content/drive/MyDrive") -> List[str]:
    """Ordered paths to try for AnyScript images on Google Colab + Drive."""
    d = drive_my_drive.rstrip(os.sep)
    # Official tarball often creates .../binarized/train (authors are under train/).
    # Try train *before* binarized alone — build_records(binarized) sees only one child "train".
    return [
        f"{d}/AnyScriptFiltered",
        f"{d}/data/datasets/AnyScriptFiltered/binarized/train",
        f"{d}/data/datasets/AnyScriptFiltered/binarized",
        f"{d}/AnyScriptFiltered/binarized/train",
        f"{d}/AnyScriptFiltered/train",
        f"{d}/AnyScriptFiltered/binarized",
        f"{d}/datasets/AnyScriptFiltered/binarized/train",
        f"{d}/datasets/AnyScriptFiltered/binarized",
    ]


def first_triplet_usable_data_root(candidates: Sequence[str]) -> Optional[str]:
    """First directory where at least two authors each have two or more pages (triplet training)."""
    for root in candidates:
        if not root or not os.path.isdir(root):
            continue
        try:
            by_a = group_by_author(build_records(root))
        except OSError:
            continue
        if len(by_a) >= 2:
            return root
    return None


def _find_immediate_subdirs_named(
    start: str,
    name: str,
    *,
    max_depth: int = 14,
    max_visit: int = 5000,
    max_hits: int = 48,
) -> List[str]:
    """BFS under ``start`` for directories whose basename is ``name`` (bounded work)."""
    from collections import deque

    out: List[str] = []
    start = os.path.abspath(start.rstrip(os.sep))
    if not os.path.isdir(start):
        return out
    q: deque = deque([(start, 0)])
    visited = 0
    while q and visited < max_visit and len(out) < max_hits:
        current, depth = q.popleft()
        visited += 1
        if depth > max_depth:
            continue
        try:
            names = os.listdir(current)
        except OSError:
            continue
        for entry in names:
            full = os.path.join(current, entry)
            try:
                is_dir = os.path.isdir(full)
            except OSError:
                continue
            if is_dir:
                if entry == name:
                    out.append(full)
                    if len(out) >= max_hits:
                        return out
                q.append((full, depth + 1))
    return out


def resolve_colab_data_root(my_drive: str = "/content/drive/MyDrive") -> Optional[str]:
    """
    Resolve a folder suitable for triplet training on Colab + Drive.

    Tries fixed paths first, then bounded BFS for ``binarized`` dirs, and ``train`` only if
    the path looks AnyScript-related (avoids picking unrelated image projects on Drive).
    """
    ordered: List[str] = []
    seen: Set[str] = set()

    def add(p: str) -> None:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)

    for p in colab_drive_data_root_candidates(my_drive):
        add(p)
    hit = first_triplet_usable_data_root(ordered)
    if hit:
        return hit

    if os.path.isdir(my_drive):
        for p in _find_immediate_subdirs_named(my_drive, "binarized"):
            add(p)
        for label in ("train", "Train"):
            for p in _find_immediate_subdirs_named(my_drive, label):
                if looks_like_anyscript_related_path(p):
                    add(p)
    hit = first_triplet_usable_data_root(ordered)
    if hit:
        return hit
    return None


def colab_drive_search_bases() -> List[str]:
    """Colab mount points to try when looking for the dataset (My Drive + shared drives)."""
    bases: List[str] = ["/content/drive/MyDrive"]
    legacy = "/content/drive/MyDrive/My Drive"
    if os.path.isdir(legacy):
        bases.append(legacy)
    shared = "/content/drive/Shareddrives"
    if os.path.isdir(shared):
        try:
            for n in sorted(os.listdir(shared))[:16]:
                p = os.path.join(shared, n)
                if os.path.isdir(p):
                    bases.append(p)
        except OSError:
            pass
    return bases


def resolve_colab_data_root_any() -> Optional[str]:
    """Try :func:`resolve_colab_data_root` on each Colab Drive base."""
    for base in colab_drive_search_bases():
        hit = resolve_colab_data_root(base)
        if hit:
            return hit
    return None


_DATA_ROOT_CLI_PLACEHOLDERS = frozenset(
    ("{data_root}", "$data_root", "%data_root%")
)


def coerce_cli_data_root(cli_value: str) -> str:
    """
    Colab ``!python ... --data_root {DATA_ROOT}`` passes the literal braces.
    Treat those tokens as ``auto`` so the same (broken) incantation still works.
    """
    s = cli_value.strip().strip("'\"")
    if s.lower() in _DATA_ROOT_CLI_PLACEHOLDERS:
        print(f"[data] treating --data_root {cli_value!r} as 'auto' (shell did not expand the variable)")
        return "auto"
    return cli_value.strip()


def resolve_training_data_root(cli_value: str) -> str:
    """
    Resolve dataset path. Use ``auto`` on Colab to search Drive inside this process
    (avoids stale ``DATA_ROOT`` from an old notebook kernel).
    """
    cli_value = coerce_cli_data_root(cli_value)
    if cli_value != "auto":
        return cli_value
    env = os.environ.get("ANYSCRIPT_DATA_ROOT", "").strip()
    if env and os.path.isdir(env):
        print(f"[data] ANYSCRIPT_DATA_ROOT -> {env!r}")
        return env
    found = resolve_colab_data_root_any()
    if found:
        print(f"[data] --data_root auto -> {found!r}")
        return found
    raise ValueError(
        "--data_root auto: no triplet-usable tree found on Colab Drive. "
        "Official extract is often .../AnyScriptFiltered/binarized/train (authors live under train/). "
        "Upload/extract the dataset, run inspect_anyscript_layout.py, set ANYSCRIPT_DATA_ROOT to the "
        "path it suggests, or: python scripts/diagnose_data_root.py"
    )


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
