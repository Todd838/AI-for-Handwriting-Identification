from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WriterEmbeddingHead(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    return torch.relu(pos_dist - neg_dist + margin).mean()


# Public HF weights; THUDM/glm-ocr returns 401 for most users (gated / legacy id).
_PUBLIC_GLM_OCR_HUB = "zai-org/GLM-OCR"
_LEGACY_GLM_OCR_HUB = "thudm/glm-ocr"


def normalize_glm_ocr_hub_id(model_name: str) -> str:
    """Map legacy THUDM/glm-ocr to zai-org/GLM-OCR so old Colab commands still work."""
    if not model_name:
        return model_name
    key = model_name.strip().lower().replace("_", "-")
    if key == _LEGACY_GLM_OCR_HUB:
        print(
            f"[vision] remapping legacy hub id {model_name!r} -> {_PUBLIC_GLM_OCR_HUB!r} "
            "(THUDM/glm-ocr requires HF access; zai-org/GLM-OCR is public)"
        )
        return _PUBLIC_GLM_OCR_HUB
    return model_name.strip()


def _unsloth_runtime_supported() -> bool:
    """Unsloth initializes only with a CUDA GPU on standard Windows/Linux wheels."""
    return bool(torch.cuda.is_available())


def _try_unsloth_vision(model_name: str, load_in_4bit: bool):
    if not _unsloth_runtime_supported():
        raise RuntimeError("Unsloth requires a CUDA GPU; use --no_unsloth or install PyTorch with CUDA.")
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )
    return model, processor


def _fallback_transformers_vision(model_name: str):
    """HF fallback. GLM-OCR uses AutoModelForImageTextToText (not AutoModel / AutoImageProcessor)."""
    mn = model_name.lower()
    if "glm" in mn and "ocr" in mn:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        return model, processor

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def _resolve_backbone_kind(backbone: str, model_name: str) -> str:
    if backbone != "auto":
        return backbone
    mn = model_name.lower()
    if "deepseek" in mn and "ocr" in mn:
        return "deepseek_ocr2"
    return "glm_style"


def load_vision_backbone(
    model_name: str,
    load_in_4bit: bool = True,
    prefer_unsloth: bool = True,
    backbone: str = "auto",
    deepseek_local_dir: Optional[str] = None,
    deepseek_snapshot_repo: str = "unsloth/DeepSeek-OCR-2",
    deepseek_download: bool = False,
    deepseek_flash_attention: bool = False,
) -> Tuple[nn.Module, object, str]:
    model_name = normalize_glm_ocr_hub_id(model_name)
    kind = _resolve_backbone_kind(backbone, model_name)

    if kind == "deepseek_ocr2":
        from deepseek_ocr2 import load_deepseek_ocr2, snapshot_deepseek_weights

        if deepseek_download:
            if not deepseek_local_dir:
                raise ValueError("--deepseek_download requires --deepseek_local_dir")
            snapshot_deepseek_weights(deepseek_local_dir, deepseek_snapshot_repo)
            path = deepseek_local_dir
        else:
            path = deepseek_local_dir or model_name
        model, tokenizer, tag = load_deepseek_ocr2(
            path,
            load_in_4bit=load_in_4bit,
            prefer_unsloth=prefer_unsloth,
            use_flash_attention=deepseek_flash_attention,
        )
        return model, tokenizer, tag

    if prefer_unsloth and _unsloth_runtime_supported():
        try:
            model, processor = _try_unsloth_vision(model_name=model_name, load_in_4bit=load_in_4bit)
            return model, processor, "unsloth"
        except Exception:
            pass
    model, processor = _fallback_transformers_vision(model_name=model_name)
    return model, processor, "transformers"


def add_vision_backbone_cli_args(p: ArgumentParser) -> None:
    p.add_argument(
        "--backbone",
        type=str,
        default="auto",
        choices=["auto", "glm_style", "deepseek_ocr2"],
        help="auto: deepseek_ocr2 if model_name looks like DeepSeek-OCR; else GLM-style loading",
    )
    p.add_argument(
        "--deepseek_local_dir",
        type=str,
        default=None,
        help="Local dir for DeepSeek-OCR 2 weights (HF id or path); used with --deepseek_download or as sole path",
    )
    p.add_argument(
        "--deepseek_snapshot_repo",
        type=str,
        default="unsloth/DeepSeek-OCR-2",
        help="HF repo id for snapshot_download when --deepseek_download is set",
    )
    p.add_argument(
        "--deepseek_download",
        action="store_true",
        help="Run snapshot_download into --deepseek_local_dir before load",
    )
    p.add_argument(
        "--deepseek_flash_attention",
        action="store_true",
        help="DeepSeek Transformers fallback only: use flash_attention_2 (Linux/CUDA + flash-attn)",
    )


def vision_backbone_kwargs_from_args(args: Any) -> Dict[str, Any]:
    return {
        "backbone": getattr(args, "backbone", "auto"),
        "deepseek_local_dir": getattr(args, "deepseek_local_dir", None),
        "deepseek_snapshot_repo": getattr(args, "deepseek_snapshot_repo", "unsloth/DeepSeek-OCR-2"),
        "deepseek_download": getattr(args, "deepseek_download", False),
        "deepseek_flash_attention": getattr(args, "deepseek_flash_attention", False),
    }


def get_backbone_hidden_size(model: nn.Module, fallback: int = 1024) -> int:
    cfg = getattr(model, "config", None)
    hidden_size = getattr(cfg, "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size
    vision_config = getattr(cfg, "vision_config", None)
    if vision_config is not None:
        # GLM-OCR vision merger output dim (matches get_image_features / LM input width)
        ohs = getattr(vision_config, "out_hidden_size", None)
        if isinstance(ohs, int):
            return int(ohs)
        vh = getattr(vision_config, "hidden_size", None)
        if isinstance(vh, int):
            return int(vh)
    return fallback


def maybe_apply_lora(model: nn.Module, lora_r: int = 16) -> nn.Module:
    if not _unsloth_runtime_supported():
        return model
    try:
        from unsloth import FastVisionModel

        return FastVisionModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=2 * lora_r,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=True,
        )
    except Exception:
        return model


def vision_uses_glm_image_processor(model: nn.Module) -> bool:
    """GlmOcr* needs HF processor pixel_values + image_grid_thw (not raw torchvision tensors)."""
    core = model.module if hasattr(model, "module") else model
    return callable(getattr(core, "get_image_features", None))


def glm_vision_inputs_from_pils(processor: Any, pil_images: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run GLM-OCR / Glm46V inputs for ``get_image_features`` (patch grid + pixel values).

    ``Glm46VProcessor.__call__`` expects ``text`` with ``<|image|>`` per image; for vision-only
    embedding we use ``processor.image_processor`` when present.
    """
    if processor is None:
        raise ValueError("GLM-OCR requires the AutoProcessor returned with the vision backbone.")
    image_proc = getattr(processor, "image_processor", None)
    if image_proc is not None:
        inputs = image_proc(images=pil_images, return_tensors="pt")
    else:
        tok = getattr(processor, "image_token", "<|image|>")
        text = [tok] * len(pil_images)
        inputs = processor(images=pil_images, text=text, return_tensors="pt")
    try:
        pv = inputs["pixel_values"]
        grid = inputs["image_grid_thw"]
    except Exception as e:
        raise KeyError(
            f"Processor output must include pixel_values and image_grid_thw; got {type(inputs)} keys "
            f"{list(inputs.keys()) if hasattr(inputs, 'keys') else inputs}"
        ) from e
    return pv, grid


def _vision_forward(core: nn.Module, pixel_values: torch.Tensor):
    if hasattr(core, "vision_encoder"):
        return core.vision_encoder(pixel_values)
    if hasattr(core, "vision_model"):
        return core.vision_model(pixel_values)
    if hasattr(core, "model"):
        inner = core.model
        if hasattr(inner, "vision_encoder"):
            return inner.vision_encoder(pixel_values)
        if hasattr(inner, "vision_model"):
            return inner.vision_model(pixel_values)
    return core(pixel_values=pixel_values)


def _pooled_features_from_glm_image_features(model: nn.Module, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
    core = model.module if hasattr(model, "module") else model
    out = core.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        return_dict=True,
    )
    chunks = out.pooler_output
    if isinstance(chunks, (tuple, list)):
        return torch.stack([c.float().mean(dim=0) for c in chunks], dim=0)
    if chunks.ndim == 2:
        return chunks.mean(dim=0, keepdim=True)
    raise RuntimeError(f"Unexpected GLM pooler_output layout: {type(chunks)} {getattr(chunks, 'shape', None)}")


def extract_pooled_features(
    model: nn.Module,
    pixel_values: torch.Tensor,
    *,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pooled visual features [B, D] for triplet / FAISS training.
    GLM-OCR: pass processor outputs as ``pixel_values`` + ``image_grid_thw``.
    Other backbones: ``pixel_values`` only (torchvision-style tensors).
    """
    core = model.module if hasattr(model, "module") else model

    if image_grid_thw is not None:
        return _pooled_features_from_glm_image_features(model, pixel_values, image_grid_thw)

    if callable(getattr(core, "get_image_features", None)):
        raise ValueError(
            "This backbone (e.g. GLM-OCR) requires HF processor inputs: call "
            "glm_vision_inputs_from_pils(processor, images) and pass image_grid_thw=... "
            "into extract_pooled_features."
        )

    out = _vision_forward(core, pixel_values)

    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
        return hidden.mean(dim=1)
    if isinstance(out, tuple) and len(out) > 0:
        hidden = out[0]
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        if hidden.ndim == 2:
            return hidden
    raise RuntimeError("Could not extract pooled features from the backbone output.")


@torch.no_grad()
def encode_batch(
    model: nn.Module,
    head: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    *,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    grid_dev = image_grid_thw.to(device) if image_grid_thw is not None else None
    feats = extract_pooled_features(model, images.to(device), image_grid_thw=grid_dev)
    return head(feats)
