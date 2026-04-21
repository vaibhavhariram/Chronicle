"""Hugging Face model loader and inference."""

import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COMPILE = os.environ.get("COMPILE", "").lower() in ("1", "true", "yes")

# Global cache
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None

# Cache dir: HF_HUB_CACHE env or workspace-relative .cache/huggingface
def _cache_dir() -> Optional[str]:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    # Use workspace cache if we're in a project (avoids ~/.cache permission issues)
    cwd = Path.cwd()
    if (cwd / "chronicle_runtime").exists():
        cache = cwd / ".cache" / "huggingface" / "hub"
        cache.mkdir(parents=True, exist_ok=True)
        return str(cache)
    return None


def _get_device() -> str:
    """Select device: cuda if available, else cpu. Override via DEVICE env."""
    env_device = os.environ.get("DEVICE", "").lower()
    if env_device in ("cuda", "cpu"):
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_deterministic():
    """Set deterministic settings for reproducible generation."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model, cached globally."""
    global _tokenizer, _model

    model_name = model_name or os.environ.get("MODEL_NAME", "gpt2")
    device = device or _get_device()

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    _set_deterministic()

    cache_dir = _cache_dir()
    _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Mistral (and other models) may lack a pad token; left-padding requires one.
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
    _tokenizer.padding_side = "left"

    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=torch_dtype
    )
    _model.to(device)
    _model.eval()
    if COMPILE:
        _model = torch.compile(_model, mode="reduce-overhead")

    return _tokenizer, _model


def generate(
    prompt: str,
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> str:
    """Generate text from prompt using the loaded model."""
    tokenizer, model = load_model()
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
