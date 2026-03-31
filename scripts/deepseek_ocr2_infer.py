"""
Run DeepSeek-OCR 2 document OCR via model.infer() (Unsloth / custom code path).
Matches Unsloth tutorial: prompts, base_size / image_size / crop_mode.
Recommended decoding constants live in deepseek_ocr2_config.py (for generate()-based flows).
"""

import argparse
import os

from deepseek_ocr2 import run_deepseek_infer
from deepseek_ocr2_config import (
    DEFAULT_INFER_BASE_SIZE,
    DEFAULT_INFER_CROP_MODE,
    DEFAULT_INFER_IMAGE_SIZE,
    INFER_PRESETS,
    PROMPT_CHOICES,
    PROMPT_REC_TEMPLATE,
    generation_kwargs_for_config,
)
from modeling_writer import (
    add_vision_backbone_cli_args,
    load_vision_backbone,
    vision_backbone_kwargs_from_args,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        type=str,
        default="unsloth/DeepSeek-OCR-2",
        help="HF repo id or local directory with weights",
    )
    p.add_argument("--image_file", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Full prompt string; overrides --prompt_style",
    )
    p.add_argument(
        "--prompt_style",
        type=str,
        default="free",
        choices=list(PROMPT_CHOICES.keys()) + ["rec"],
        help="Preset from DeepSeek docs (rec: use --ref_text)",
    )
    p.add_argument("--ref_text", type=str, default="xxxx", help="For prompt_style=rec")
    p.add_argument("--infer_preset", type=str, default="unsloth_default", choices=list(INFER_PRESETS.keys()))
    p.add_argument("--base_size", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--no_crop_mode", action="store_true")
    p.add_argument("--no_save_results", action="store_true")
    p.add_argument("--test_compress", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_unsloth", action="store_true")
    p.add_argument("--print_generation_defaults", action="store_true")
    add_vision_backbone_cli_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    if args.print_generation_defaults:
        print("Recommended decoding (for generate(), not infer):", generation_kwargs_for_config())
    preset = INFER_PRESETS[args.infer_preset]
    base_size = args.base_size if args.base_size is not None else preset["base_size"]
    image_size = args.image_size if args.image_size is not None else preset["image_size"]
    crop_mode = not args.no_crop_mode and preset.get("crop_mode", True)

    if args.prompt:
        prompt = args.prompt
    elif args.prompt_style == "rec":
        prompt = PROMPT_REC_TEMPLATE.replace("xxxx", args.ref_text)
    else:
        prompt = PROMPT_CHOICES[args.prompt_style]

    vb = vision_backbone_kwargs_from_args(args)
    vb["backbone"] = "deepseek_ocr2"
    model, tokenizer, tag = load_vision_backbone(
        model_name=args.model_path,
        load_in_4bit=args.load_in_4bit,
        prefer_unsloth=not args.no_unsloth,
        **vb,
    )
    print(f"Loaded {tag}")
    res = run_deepseek_infer(
        model,
        tokenizer,
        prompt=prompt,
        image_file=args.image_file,
        output_path=args.output_path,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=not args.no_save_results,
        test_compress=args.test_compress,
    )
    print("infer done:", res)


if __name__ == "__main__":
    main()
