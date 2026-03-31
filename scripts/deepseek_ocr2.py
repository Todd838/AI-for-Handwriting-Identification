"""
DeepSeek-OCR 2: snapshot download, Unsloth / Transformers loading, and model.infer() wrapper.
See deepseek_ocr2_config.py for prompts and recommended decoding constants.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from deepseek_ocr2_config import DEFAULT_SNAPSHOT_REPO, RECOMMENDED_DECODING


def snapshot_deepseek_weights(local_dir: str, repo_id: str = DEFAULT_SNAPSHOT_REPO) -> str:
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id, local_dir=local_dir)
    return local_dir


def _optional_unsloth_kwargs() -> Dict[str, Any]:
    """Do not import unsloth here — importing the package requires a GPU on many builds."""
    from transformers import AutoModel

    return {
        "auto_model": AutoModel,
        "trust_remote_code": True,
        "unsloth_force_compile": True,
        "use_gradient_checkpointing": "unsloth",
    }


def load_deepseek_ocr2_unsloth(
    model_name_or_path: str,
    *,
    load_in_4bit: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek Unsloth path requires CUDA")
    os.environ.setdefault("UNSLOTH_WARN_UNINITIALIZED", "0")
    from unsloth import FastVisionModel

    extra = _optional_unsloth_kwargs()
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name_or_path,
        load_in_4bit=load_in_4bit,
        **extra,
    )
    return model, tokenizer


def load_deepseek_ocr2_transformers(
    model_name_or_path: str,
    *,
    use_flash_attention: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.nn.Module, Any]:
    from transformers import AutoModel, AutoTokenizer

    tok_kw = {"trust_remote_code": True}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kw)
    model_kw: Dict[str, Any] = {"trust_remote_code": True, "use_safetensors": True}
    if use_flash_attention:
        model_kw["_attn_implementation"] = "flash_attention_2"
    model = AutoModel.from_pretrained(model_name_or_path, **model_kw)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        model = model.to(dtype or torch.bfloat16)
    else:
        model = model.float()
    return model, tokenizer


def load_deepseek_ocr2(
    model_name_or_path: str,
    *,
    load_in_4bit: bool = False,
    prefer_unsloth: bool = True,
    use_flash_attention: bool = False,
) -> Tuple[torch.nn.Module, Any, str]:
    if prefer_unsloth and torch.cuda.is_available():
        try:
            m, t = load_deepseek_ocr2_unsloth(model_name_or_path, load_in_4bit=load_in_4bit)
            return m, t, "deepseek_ocr2_unsloth"
        except Exception:
            pass
    m, t = load_deepseek_ocr2_transformers(
        model_name_or_path, use_flash_attention=use_flash_attention
    )
    return m, t, "deepseek_ocr2_transformers"


def run_deepseek_infer(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    image_file: str,
    output_path: str,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    save_results: bool = True,
    test_compress: bool = False,
    extra_infer_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    if not hasattr(model, "infer"):
        raise RuntimeError(
            "This model has no .infer(); use a DeepSeek-OCR 2 build with infer(tokenizer, ...)."
        )
    kw: Dict[str, Any] = {
        "prompt": prompt,
        "image_file": image_file,
        "output_path": output_path,
        "base_size": base_size,
        "image_size": image_size,
        "crop_mode": crop_mode,
        "save_results": save_results,
        "test_compress": test_compress,
    }
    if extra_infer_kwargs:
        kw.update(extra_infer_kwargs)
    return model.infer(tokenizer, **kw)


def generation_kwargs_for_config() -> Dict[str, Any]:
    """Pass into model.generate / chat templates when not using infer()."""
    return dict(RECOMMENDED_DECODING)
