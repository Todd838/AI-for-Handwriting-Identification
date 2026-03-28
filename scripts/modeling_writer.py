from typing import Optional, Tuple

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


def _try_unsloth_vision(model_name: str, load_in_4bit: bool):
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )
    return model, processor


def _fallback_transformers_vision(model_name: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def load_vision_backbone(
    model_name: str,
    load_in_4bit: bool = True,
    prefer_unsloth: bool = True,
) -> Tuple[nn.Module, object, str]:
    if prefer_unsloth:
        try:
            model, processor = _try_unsloth_vision(model_name=model_name, load_in_4bit=load_in_4bit)
            return model, processor, "unsloth"
        except Exception:
            pass
    model, processor = _fallback_transformers_vision(model_name=model_name)
    return model, processor, "transformers"


def get_backbone_hidden_size(model: nn.Module, fallback: int = 1024) -> int:
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "hidden_size"):
        return int(vision_config.hidden_size)
    return fallback


def maybe_apply_lora(model: nn.Module, lora_r: int = 16) -> nn.Module:
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


def extract_pooled_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Returns a pooled visual feature tensor [B, D] from either:
    - a model exposing `vision_encoder(...)`
    - a generic HF model returning last_hidden_state
    """
    if hasattr(model, "vision_encoder"):
        out = model.vision_encoder(pixel_values)
    else:
        out = model(pixel_values=pixel_values)

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
) -> torch.Tensor:
    feats = extract_pooled_features(model, images.to(device))
    return head(feats)
