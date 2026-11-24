# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precomputed multimodal embeddings passed through ``multi_modal_embeddings``.

Each entry of ``multi_modal_embeddings[modality]`` is either a bare tensor of
shape ``[rows, width]`` or a dict holding the tensor under ``"mm_embeddings"``
plus the geometry and timing metadata that the raw-media path would have
derived from the image or video (grid, timestamps, ...). Models validate the
metadata they need; this module only normalizes the container.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch

EXTERNAL_EMBEDDING_MODALITIES = ("image", "video")

_ITEM_KEYS = frozenset(
    {
        "mm_embeddings",
        "image_grid_thw",
        "video_grid_thw",
        "timestamps",
        "fps",
        "frames_indices",
        "second_per_grid_ts",
    }
)


@dataclass(frozen=True)
class ExternalEmbeddingItem:
    """One precomputed image or video embedding with its source metadata."""

    modality: str
    embedding: torch.Tensor
    """Embedding rows in prompt order, shape ``[rows, width]``."""
    grid_thw: Optional[tuple[int, int, int]] = None
    """Pre-merge patch grid ``(t, h, w)`` as the vision processor reports it."""
    timestamps: Optional[tuple[float, ...]] = None
    """Seconds per temporal patch, for models that print frame timestamps."""
    fps: Optional[float] = None
    """Frames per second. Qwen3-VL reads it with ``frames_indices`` as the
    source video's rate; Qwen2.5-VL reads it alone as the sampled rate."""
    frames_indices: Optional[tuple[int, ...]] = None
    """Source frame index of each sampled frame (Qwen3-VL)."""
    second_per_grid_ts: Optional[float] = None
    """Seconds covered by one temporal patch, for models that scale MRoPE time."""

    @property
    def num_rows(self) -> int:
        """Number of embedding rows, which is the number of prompt placeholders."""
        return int(self.embedding.shape[0])


def _as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Convert a list, tuple or tensor of integers to a tuple of Python ints."""
    if isinstance(value, torch.Tensor):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            raise ValueError(f"{name} must hold integers, got {value.dtype}")
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list, tuple or tensor of integers")
    result = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(f"{name} must hold integers, got {type(entry).__name__}")
        result.append(int(entry))
    return tuple(result)


def _as_float_tuple(value: Any, name: str) -> tuple[float, ...]:
    """Convert a list, tuple or tensor of numbers to a tuple of finite floats."""
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list, tuple or tensor of numbers")
    result = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(f"{name} must hold numbers, got {type(entry).__name__}")
        entry = float(entry)
        if not math.isfinite(entry):
            raise ValueError(f"{name} must hold finite numbers")
        result.append(entry)
    return tuple(result)


def _as_positive_float(value: Any, name: str) -> float:
    """Convert a scalar to a finite positive float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _check_embedding_tensor(embedding: Any, name: str) -> torch.Tensor:
    """Check the dense floating rank-2 tensor contract of an attached embedding."""
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(embedding).__name__}")
    if embedding.layout != torch.strided:
        raise ValueError(f"{name} must be a dense tensor, got layout {embedding.layout}")
    if not embedding.dtype.is_floating_point:
        raise ValueError(f"{name} must be a floating-point tensor, got {embedding.dtype}")
    if embedding.dim() != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(embedding.shape)}")
    if embedding.shape[0] <= 0 or embedding.shape[1] <= 0:
        raise ValueError(f"{name} must be non-empty, got shape {tuple(embedding.shape)}")
    return embedding


def _parse_item(modality: str, index: int, entry: Any) -> ExternalEmbeddingItem:
    """Normalize one user entry to an ExternalEmbeddingItem."""
    name = f"{modality} embedding {index}"
    if isinstance(entry, torch.Tensor):
        return ExternalEmbeddingItem(modality, _check_embedding_tensor(entry, name))
    if not isinstance(entry, dict):
        raise TypeError(f"{name} must be a tensor or a dict, got {type(entry).__name__}")
    unknown = set(entry) - _ITEM_KEYS
    if unknown:
        raise ValueError(f"{name} has unknown keys {sorted(unknown)}")
    if "mm_embeddings" not in entry:
        raise ValueError(f"{name} is missing 'mm_embeddings'")
    grid_key = f"{modality}_grid_thw"
    other_grid_key = "video_grid_thw" if modality == "image" else "image_grid_thw"
    if other_grid_key in entry:
        raise ValueError(f"{name} must use '{grid_key}', not '{other_grid_key}'")
    if modality == "image":
        for key in ("timestamps", "fps", "frames_indices", "second_per_grid_ts"):
            if key in entry:
                raise ValueError(f"{name}: '{key}' only applies to video")

    grid = None
    if entry.get(grid_key) is not None:
        grid = _as_int_tuple(entry[grid_key], grid_key)
        if len(grid) != 3:
            raise ValueError(f"{name}: {grid_key} must have 3 entries (t, h, w), got {grid}")
        if any(dim <= 0 for dim in grid):
            raise ValueError(f"{name}: {grid_key} entries must be positive, got {grid}")

    def optional(key: str, convert: Callable[[Any, str], Any]) -> Any:
        return convert(entry[key], key) if entry.get(key) is not None else None

    return ExternalEmbeddingItem(
        modality,
        _check_embedding_tensor(entry["mm_embeddings"], name),
        grid_thw=grid,
        timestamps=optional("timestamps", _as_float_tuple),
        fps=optional("fps", _as_positive_float),
        frames_indices=optional("frames_indices", _as_int_tuple),
        second_per_grid_ts=optional("second_per_grid_ts", _as_positive_float),
    )


def parse_external_embeddings(
    multimodal_embedding: Any,
    supported_modalities: Sequence[str] = EXTERNAL_EMBEDDING_MODALITIES,
) -> dict[str, list[ExternalEmbeddingItem]]:
    """Normalize ``multi_modal_embeddings`` to per-modality item lists.

    Args:
        multimodal_embedding: Mapping of modality to a tensor, a dict entry, or
            a list of them, in the order of that modality's prompt placeholders.
        supported_modalities: Modalities the caller accepts.

    Returns:
        Modality to items, in the order given.
    """
    if not isinstance(multimodal_embedding, dict):
        raise ValueError("multimodal_embedding must be a dictionary")
    unsupported = set(multimodal_embedding) - set(supported_modalities)
    if unsupported:
        raise ValueError(
            f"Unsupported modalities {sorted(unsupported)} for external multimodal "
            f"embeddings; supported: {list(supported_modalities)}"
        )
    items: dict[str, list[ExternalEmbeddingItem]] = {}
    for modality, entries in multimodal_embedding.items():
        if isinstance(entries, (torch.Tensor, dict)):
            entries = [entries]
        if not isinstance(entries, (list, tuple)):
            raise TypeError(f"multi_modal_embeddings['{modality}'] must be a list")
        items[modality] = [_parse_item(modality, i, entry) for i, entry in enumerate(entries)]
    if not any(items.values()):
        raise ValueError("At least one multimodal embedding is required")
    return items
