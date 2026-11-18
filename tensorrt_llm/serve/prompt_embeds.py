# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prompt_embeds`` on the OpenAI routes: payload decoding and chat placement.

Completions take ``prompt_embeds`` as one or more base64 ``torch.save`` payloads,
each a whole prompt of ``[L, hidden_size]`` rows. Chat takes a content part
``{"type": "prompt_embeds", "prompt_embeds": {"data": <base64>}}``: the part
renders as a unique marker, and after the chat template runs, each marker becomes
a run of placeholder ids whose positions take the part's rows.

Tensor payloads from clients are untrusted, so the server accepts them only when
started with ``--enable_prompt_embeds``. Decoding is bounded by
``TRTLLM_PROMPT_EMBEDS_MAX_BYTES`` per request and runs off the event loop.
"""

import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from tensorrt_llm.inputs.utils import load_base64_tensor

PROMPT_EMBEDS_PART_TYPE = "prompt_embeds"

# Key under which a parsed chat message carries its prompt_embeds parts. The
# chat route pops it before rendering.
PROMPT_EMBEDS_MESSAGE_KEY = "prompt_embeds_parts"

PROMPT_EMBEDS_MAX_BYTES_ENV = "TRTLLM_PROMPT_EMBEDS_MAX_BYTES"
DEFAULT_PROMPT_EMBEDS_MAX_BYTES = 256 * 1024 * 1024

DISABLED_MESSAGE = "prompt_embeds requires the server to be started with --enable_prompt_embeds."

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Token id at embedding positions; matches build_prompt_embeds' default.
PLACEHOLDER_TOKEN_ID = 0


@dataclass(frozen=True)
class ChatPromptEmbedsPart:
    """One ``prompt_embeds`` content part of a chat message.

    Attributes:
        marker: Unique text the part renders as in the chat template.
        data: The part's base64 ``torch.save`` payload, still encoded.
    """

    marker: str
    data: str


def prompt_embeds_max_bytes() -> int:
    """Return the per-request limit on decoded ``prompt_embeds`` bytes."""
    value = os.environ.get(PROMPT_EMBEDS_MAX_BYTES_ENV)
    if value is None:
        return DEFAULT_PROMPT_EMBEDS_MAX_BYTES
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"{PROMPT_EMBEDS_MAX_BYTES_ENV} must be positive, got {value!r}")
    return limit


def new_chat_part(part: Dict[str, Any]) -> ChatPromptEmbedsPart:
    """Validate a ``prompt_embeds`` content part without decoding it."""
    body = part.get(PROMPT_EMBEDS_PART_TYPE)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, str) or not data:
        raise ValueError(
            "A prompt_embeds content part needs prompt_embeds.data: a base64 "
            "torch.save of a [num_tokens, hidden_size] tensor."
        )
    return ChatPromptEmbedsPart(marker=f"<|prompt_embeds:{uuid.uuid4().hex}|>", data=data)


def decode_prompt_embeds(
    payloads: Sequence[str],
    *,
    max_bytes: int,
    hidden_size: Optional[int] = None,
) -> List[torch.Tensor]:
    """Decode a request's ``prompt_embeds`` payloads under one byte budget.

    Blocking; run it off the event loop.

    Args:
        payloads: Base64 ``torch.save`` payloads, each a ``[rows, hidden]`` tensor.
        max_bytes: Limit on the decoded bytes of all payloads together.
        hidden_size: Expected row width, when the server knows it.

    Returns:
        The tensors, in payload order.
    """
    encoded_limit = 4 * ((max_bytes + 2) // 3)
    if sum(len(p) if isinstance(p, str) else 0 for p in payloads) > encoded_limit:
        raise ValueError(
            f"prompt_embeds payloads exceed the {max_bytes}-byte request limit "
            f"({PROMPT_EMBEDS_MAX_BYTES_ENV})."
        )
    tensors = []
    remaining = max_bytes
    for index, payload in enumerate(payloads):
        tensor = load_base64_tensor(payload, max_bytes=remaining)
        remaining -= tensor.untyped_storage().nbytes()
        if tensor.dim() != 2:
            raise ValueError(
                f"prompt_embeds[{index}] must have shape [num_tokens, hidden_size], "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"prompt_embeds[{index}] dtype must be float16, bfloat16 or float32, "
                f"got {tensor.dtype}."
            )
        if tensor.shape[0] == 0:
            raise ValueError(f"prompt_embeds[{index}] has no rows.")
        if hidden_size is not None and tensor.shape[1] != hidden_size:
            raise ValueError(
                f"prompt_embeds[{index}] has hidden size {tensor.shape[1]}; the "
                f"model expects {hidden_size}."
            )
        tensors.append(tensor)
    return tensors


def model_hidden_size(model_config: Any) -> Optional[int]:
    """Return the text model's hidden size from a Hugging Face config, if known."""
    if model_config is None:
        return None
    get_text_config = getattr(model_config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else model_config
    hidden_size = getattr(text_config, "hidden_size", None)
    return hidden_size if isinstance(hidden_size, int) else None


def pop_chat_parts(conversation: List[Dict[str, Any]]) -> List[ChatPromptEmbedsPart]:
    """Remove and return the ``prompt_embeds`` parts of parsed chat messages."""
    parts: List[ChatPromptEmbedsPart] = []
    for message in conversation:
        parts.extend(message.pop(PROMPT_EMBEDS_MESSAGE_KEY, None) or ())
    return parts


def splice_chat_prompt_embeds(
    text: str,
    parts: Sequence[ChatPromptEmbedsPart],
    rows: Sequence[torch.Tensor],
    tokenize: Callable[[str, bool], List[int]],
    add_special_tokens: bool,
) -> Dict[str, Any]:
    """Turn a rendered chat prompt with part markers into an embeddings prompt.

    The text between markers is tokenized segment by segment (special tokens
    only for the first segment, when requested); each marker becomes one
    placeholder id per row of its part.

    Args:
        text: The rendered chat prompt.
        parts: The request's parts; each marker must occur exactly once.
        rows: Decoded rows of each part, in ``parts`` order.
        tokenize: ``tokenize(segment, add_special_tokens)`` with the engine's
            tokenizer.
        add_special_tokens: Whether the prompt gets the tokenizer's special
            tokens (e.g. BOS).

    Returns:
        A ``TokensPrompt`` dict with ``prompt_embeds`` and ``prompt_embeds_mask``.
    """
    dtypes = {r.dtype for r in rows}
    if len(dtypes) > 1:
        raise ValueError(
            f"All prompt_embeds parts of a request must share one dtype, got "
            f"{sorted(str(d) for d in dtypes)}."
        )
    placed = []
    for part, part_rows in zip(parts, rows):
        offset = text.find(part.marker)
        if offset < 0:
            raise ValueError("The chat template dropped a message holding a prompt_embeds part.")
        if text.find(part.marker, offset + 1) >= 0:
            raise ValueError("The chat template repeated a message holding a prompt_embeds part.")
        placed.append((offset, part.marker, part_rows))
    placed.sort(key=lambda item: item[0])

    token_ids: List[int] = []
    mask: List[bool] = []

    def add_text(segment: str) -> None:
        """Append the ids of one text segment."""
        first = not token_ids
        if not segment and not (first and add_special_tokens):
            return
        ids = tokenize(segment, add_special_tokens and first)
        token_ids.extend(ids)
        mask.extend([False] * len(ids))

    cursor = 0
    for offset, marker, part_rows in placed:
        add_text(text[cursor:offset])
        token_ids.extend([PLACEHOLDER_TOKEN_ID] * part_rows.shape[0])
        mask.extend([True] * part_rows.shape[0])
        cursor = offset + len(marker)
    add_text(text[cursor:])
    return {
        "prompt_token_ids": token_ids,
        "prompt_embeds": torch.cat([r for _, _, r in placed], dim=0),
        "prompt_embeds_mask": mask,
    }
