#!/usr/bin/env python3
"""
Find where page images live under a Drive or local folder.

Usage (Colab):
  python .../inspect_anyscript_layout.py
  python .../inspect_anyscript_layout.py /content/drive/MyDrive/AnyScriptFiltered

With no path, picks the first existing folder from the same candidates as training (often
``.../data/datasets/AnyScriptFiltered/binarized/train`` after ``tar`` extract).

Tries the given path and each immediate subdirectory as data_root and reports page counts.
If your count is 0 at the top level, set DATA_ROOT to the subdirectory that shows pages.
"""

import argparse
import os
import sys

# Same package as training (run from repo scripts/ or set PYTHONPATH).
from data_anyscript import (
    build_records,
    colab_drive_data_root_candidates,
    group_by_author,
    looks_like_anyscript_related_path,
)


def report(label: str, root: str) -> int:
    if not os.path.isdir(root):
        print(f"{label}: not a directory -> {root!r}")
        return 0
    recs = build_records(root)
    by_a = group_by_author(recs)
    n_multi = len(by_a)
    print(f"{label}: {root!r}")
    print(f"  pages={len(recs)}, authors_with_2+_pages={n_multi}")
    return len(recs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Folder to inspect; default: first existing path under /content/drive/MyDrive",
    )
    p.add_argument(
        "--max_depth",
        type=int,
        default=2,
        help="How deep to try single-child descent hints (default 2).",
    )
    p.add_argument(
        "--force-unrelated-scan",
        action="store_true",
        help="Scan all My Drive siblings (can suggest wrong datasets e.g. other Kaggle projects).",
    )
    args = p.parse_args()
    print("=== inspect_anyscript_layout ===\n")
    root = args.root
    if root:
        root = os.path.expanduser(root)
    else:
        my_drive = "/content/drive/MyDrive"
        hints = colab_drive_data_root_candidates(my_drive) + [my_drive]
        root = next((h for h in hints if os.path.isdir(h)), None)
        if not root:
            print("ERROR: no dataset folder found under Google Drive.")
            print("Mount Drive in the notebook, extract AnyScriptFiltered.tar.gz, then retry.")
            print("Paths checked (same order as --data_root auto):\n")
            for h in hints:
                print(f"  exists={os.path.isdir(h)!s:5} {h}")
            sys.exit(1)
        print(f"(auto) first existing candidate -> {root!r}\n")

    if not os.path.isdir(root):
        print(f"ERROR: not found or not a directory: {root!r}")
        print("Mount Drive (Colab: /content/drive/MyDrive). After tar extract, data is often under:")
        for h in colab_drive_data_root_candidates("/content/drive/MyDrive") + ["/content/drive/MyDrive"]:
            print(f"  exists={os.path.isdir(h)!s:5} {h}")
        sys.exit(1)

    children = sorted(os.listdir(root))
    print(f"Top-level entries under {root!r} ({len(children)} items):")
    for name in children[:30]:
        pth = os.path.join(root, name)
        kind = "dir" if os.path.isdir(pth) else "file"
        print(f"  [{kind}] {name}")
    if len(children) > 30:
        print(f"  ... and {len(children) - 30} more\n")
    else:
        print()

    best_path = root
    best_n = report("Candidate", root)

    for name in children:
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        n = report(f"Subdir {name!r}", sub)
        if n > best_n:
            best_n = n
            best_path = sub

    # One more level: best subdirectory's children (common: .../train/author_id)
    if best_n == 0 and args.max_depth >= 2:
        for name in children:
            sub = os.path.join(root, name)
            if not os.path.isdir(sub):
                continue
            for name2 in sorted(os.listdir(sub))[:200]:
                sub2 = os.path.join(sub, name2)
                if not os.path.isdir(sub2):
                    continue
                n = report(f"Nested {name}/{name2!r}", sub2)
                if n > best_n:
                    best_n = n
                    best_path = sub2

    tgz_here = [
        os.path.join(root, f)
        for f in children
        if f.endswith(".tar.gz") or f.endswith(".tgz")
    ]

    # Only archives: extract first — do not suggest unrelated Drive folders.
    if best_n == 0 and tgz_here:
        parent = os.path.dirname(root.rstrip(os.sep))
        print("\n*** No image folders here — only archive(s). Extract first, for example:\n")
        for ap in tgz_here:
            print(f'  !tar -xzf "{ap}" -C "{parent}"')
        print(
            "\n(Official tarball usually creates .../data/datasets/AnyScriptFiltered/binarized)\n"
            "Then re-run the dataset cell or this script.\n\n---"
        )
        sys.exit(2)

    # Official tarball often extracts to MyDrive/data/datasets/... (not under AnyScriptFiltered/)
    if best_n == 0 and (parent := os.path.dirname(root.rstrip(os.sep))) and os.path.isdir(parent):
        print(
            f"\nNo pages under {root!r}. Scanning siblings under {parent!r} "
            "(AnyScript-related paths only; use --force-unrelated-scan for full Drive)...\n"
        )
        root_abs = os.path.abspath(root)
        for name in sorted(os.listdir(parent))[:50]:
            sub = os.path.join(parent, name)
            if not os.path.isdir(sub) or os.path.abspath(sub) == root_abs:
                continue
            if not args.force_unrelated_scan and not looks_like_anyscript_related_path(sub):
                continue
            n = report(f"MyDrive sibling {name!r}", sub)
            if n > best_n:
                best_n, best_path = n, sub
            for name2 in sorted(os.listdir(sub))[:40]:
                sub2 = os.path.join(sub, name2)
                if not os.path.isdir(sub2):
                    continue
                if not args.force_unrelated_scan and not looks_like_anyscript_related_path(sub2):
                    continue
                n2 = report(f"  nested {name}/{name2!r}", sub2)
                if n2 > best_n:
                    best_n, best_path = n2, sub2
                for name3 in sorted(os.listdir(sub2))[:30]:
                    sub3 = os.path.join(sub2, name3)
                    if not os.path.isdir(sub3):
                        continue
                    if not args.force_unrelated_scan and not looks_like_anyscript_related_path(sub3):
                        continue
                    n3 = report(f"    {name}/{name2}/{name3!r}", sub3)
                    if n3 > best_n:
                        best_n, best_path = n3, sub3
                    for name4 in sorted(os.listdir(sub3))[:25]:
                        sub4 = os.path.join(sub3, name4)
                        if not os.path.isdir(sub4):
                            continue
                        if not args.force_unrelated_scan and not looks_like_anyscript_related_path(sub4):
                            continue
                        n4 = report(f"      .../{name4!r}", sub4)
                        if n4 > best_n:
                            best_n, best_path = n4, sub4

    print("\n---")
    if best_n == 0:
        print(
            "No pages found. Upload/extract the dataset, or point at the folder whose "
            "children are author IDs (with .png/.jpg under each)."
        )
        sys.exit(2)
    print(f"Suggested DATA_ROOT (most pages here): {best_path!r}")
    by_a = group_by_author(build_records(best_path))
    if len(by_a) < 2:
        print(
            f"WARNING: only {len(by_a)} author(s) with 2+ pages — triplet training needs >= 2."
        )


if __name__ == "__main__":
    main()
