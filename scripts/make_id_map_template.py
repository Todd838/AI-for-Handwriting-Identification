"""Emit JSON id-map templates (keys match export_anyscript_submission); fill values from the official challenge release."""

import argparse
import os

from data_anyscript import write_book_id_map_template, write_page_id_map_template


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--granularity", type=str, choices=["page", "book"], required=True)
    p.add_argument(
        "--placeholder",
        type=str,
        default="",
        help="String to use for every value until you replace with official ids (default empty).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    if args.granularity == "page":
        write_page_id_map_template(args.data_root, args.out_json, value=args.placeholder)
    else:
        write_book_id_map_template(args.data_root, args.out_json, value=args.placeholder)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
