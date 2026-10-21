"""VRAM helpers: the 4 GB budget forces strictly sequential model residency."""
from __future__ import annotations

import gc
import logging

log = logging.getLogger(__name__)


def free_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def log_vram(tag: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            log.info("[VRAM] %-28s used %.2f / %.2f GB (device-wide, incl. Unreal)",
                     tag, (total - free) / 1e9, total / 1e9)
    except ImportError:  # pragma: no cover
        pass


def bnb_4bit_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
