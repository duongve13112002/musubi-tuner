"""DeepSpeed integration utilities shared across all training scripts."""

import logging
from types import SimpleNamespace
from typing import Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def is_deepspeed_active(accelerator) -> bool:
    """Return True when DeepSpeed is configured (any ZeRO stage)."""
    from accelerate.utils import DistributedType

    return accelerator.distributed_type == DistributedType.DEEPSPEED


def is_deepspeed_zero3(accelerator) -> bool:
    """Return True only when DeepSpeed ZeRO Stage 3 is active.

    ZeRO Stage 3 shards model parameters across GPUs, which makes the standard
    model.state_dict() return empty/incomplete tensors. Always use
    accelerator.get_state_dict(model) when this returns True.
    """
    from accelerate.utils import DistributedType

    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    return (
        accelerator.distributed_type == DistributedType.DEEPSPEED
        and plugin is not None
        and getattr(plugin, "zero_stage", 0) == 3
    )


def patch_model_for_deepspeed(model: nn.Module) -> None:
    """Set model.config.hidden_size so DeepSpeed 'auto' bucket values resolve correctly.

    DeepSpeed uses model.config.hidden_size (HuggingFace convention) to compute
    appropriate values for 'auto' entries in the DeepSpeed JSON config. The models
    in this project are plain nn.Module and don't follow that convention, but they
    expose an equivalent dimension under different attribute names. This function
    creates a minimal model.config namespace so 'auto' values work transparently.

    Attribute name mapping per model family:
      - FLUX / FLUX.2 / HunyuanVideo : model.hidden_size
      - Qwen Image                    : model.inner_dim
      - Kandinsky5                    : model.model_dim
    """
    # Already has a proper config.hidden_size — nothing to do
    existing_config = getattr(model, "config", None)
    if existing_config is not None and hasattr(existing_config, "hidden_size"):
        return

    # Try each attribute name used across the project's model families
    hidden_size: Optional[int] = None
    for attr in ("hidden_size", "inner_dim", "model_dim"):
        val = getattr(model, attr, None)
        if isinstance(val, int) and val > 0:
            hidden_size = val
            break

    if hidden_size is not None:
        model.config = SimpleNamespace(hidden_size=hidden_size)
        logger.debug(
            f"Patched {type(model).__name__}.config.hidden_size = {hidden_size} for DeepSpeed 'auto' compatibility"
        )
    else:
        logger.warning(
            f"{type(model).__name__} has no recognised hidden-size attribute "
            "(tried: hidden_size, inner_dim, model_dim). "
            "Do not use 'auto' values in your DeepSpeed config for this model; "
            "specify explicit bucket sizes instead."
        )


def check_block_swap_deepspeed_zero3(blocks_to_swap: int, accelerator) -> None:
    """Raise ValueError if block swap and ZeRO Stage 3 are both active.

    Block swap manually moves transformer blocks between CPU and GPU outside
    PyTorch's knowledge. ZeRO Stage 3 also partitions the same parameters across
    GPUs. The two mechanisms conflict and cannot coexist.
    """
    if blocks_to_swap > 0 and is_deepspeed_zero3(accelerator):
        raise ValueError(
            "--blocks_to_swap is incompatible with DeepSpeed ZeRO Stage 3. "
            "Both try to manage the same model parameters simultaneously. "
            "Choose one: use --blocks_to_swap with DDP (default), "
            "or use DeepSpeed ZeRO Stage 1 or 2 without block swap."
        )


def gather_state_dict_for_save(accelerator, prepared_model, should_save: bool = True):
    """Gather a ZeRO3-safe state dict from all processes before saving.

    This must be called on ALL processes (it is a collective operation), but
    only the result on the main process is needed for writing to disk. When
    ZeRO Stage 3 is NOT active the function returns None immediately and the
    caller should fall back to model.state_dict() as usual.

    Args:
        accelerator: The Accelerator instance.
        prepared_model: The model returned by accelerator.prepare() (not unwrapped).
        should_save: Whether a save is actually about to happen. Passing False
            lets all processes skip the gather efficiently when the epoch/step
            condition is not met.

    Returns:
        dict or None: Full state dict on all processes when ZeRO3 is active and
        should_save is True; None otherwise.
    """
    if not is_deepspeed_zero3(accelerator) or not should_save:
        return None
    # Collective gather — every process participates
    return accelerator.get_state_dict(prepared_model)
