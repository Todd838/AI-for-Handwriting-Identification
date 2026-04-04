#!/usr/bin/env python3
"""Print Colab/Drive layout hints when training finds 0 pages or wrong DATA_ROOT."""

import argparse
import os
import sys

from data_anyscript import (
    build_records,
    colab_drive_search_bases,
    group_by_author,
    resolve_colab_data_root,
    resolve_colab_data_root_any,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Folder to inspect (default: try Colab Drive bases)",
    )
    args = p.parse_args()

    if args.path:
        roots = [os.path.abspath(args.path)]
    else:
        roots = [b for b in colab_drive_search_bases() if os.path.isdir(b)]

    if not roots:
        print("No paths to inspect. Mount Drive on Colab or pass a folder path.")
        sys.exit(1)

    print("=== diagnose_data_root ===\n")
    for r in roots:
        print(f"-- {r!r} exists={os.path.isdir(r)}")
        if not os.path.isdir(r):
            continue
        try:
            kids = sorted(os.listdir(r))[:25]
        except OSError as e:
            print(f"   listdir error: {e}")
            continue
        print(f"   first entries ({len(kids)} shown): {kids}")
        hit = resolve_colab_data_root(r)
        print(f"   resolve_colab_data_root -> {hit!r}")

    print()
    any_hit = resolve_colab_data_root_any()
    print(f"resolve_colab_data_root_any() -> {any_hit!r}")

    if any_hit:
        recs = build_records(any_hit)
        by_a = group_by_author(recs)
        print(f"pages={len(recs)}, authors_2+_pages={len(by_a)}")
        print(f"\nUse: --data_root {any_hit!r}\n  or: --data_root auto")


if __name__ == "__main__":
    main()
