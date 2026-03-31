"""
Recommended decoding / prompt presets from DeepSeek-OCR 2 + Unsloth docs.
Writer-retrieval training ignores prompts; use deepseek_ocr2_infer.py for OCR generation.
"""

# Generation / decoding (when using model.generate or APIs that accept these)
RECOMMENDED_DECODING = {
    "temperature": 0.0,
    "max_new_tokens": 8192,
    "ngram_size": 30,
    "window_size": 90,
}

# Unsloth infer() defaults (dynamic resolution path)
DEFAULT_INFER_BASE_SIZE = 1024
DEFAULT_INFER_IMAGE_SIZE = 640
DEFAULT_INFER_CROP_MODE = True

# Alternative HF snippet uses image_size 768
INFER_PRESETS = {
    "unsloth_default": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "hf_transformers": {"base_size": 1024, "image_size": 768, "crop_mode": True},
}

# Prompt templates (prepend <image> where required)
PROMPT_DOCUMENT_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown."
PROMPT_OCR = "<image>\n<|grounding|>OCR this image."
PROMPT_FREE_OCR = "<image>\nFree OCR."
PROMPT_FIGURE = "<image>\nParse the figure."
PROMPT_GENERAL = "<image>\nDescribe this image in detail."
PROMPT_REC_TEMPLATE = "<image>\nLocate <|ref|>xxxx<|/ref|> in the image."  # replace xxxx

PROMPT_CHOICES = {
    "document": PROMPT_DOCUMENT_MARKDOWN,
    "ocr": PROMPT_OCR,
    "free": PROMPT_FREE_OCR,
    "figure": PROMPT_FIGURE,
    "general": PROMPT_GENERAL,
    "markdown": PROMPT_DOCUMENT_MARKDOWN,
}

DEFAULT_SNAPSHOT_REPO = "unsloth/DeepSeek-OCR-2"
