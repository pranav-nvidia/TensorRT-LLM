# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded decoding of base64 ``torch.save`` tensor payloads from clients."""

import asyncio
import base64
import io
import pickle
import zipfile

import pytest
import torch

from tensorrt_llm.inputs.utils import async_load_base64_tensor, load_base64_tensor

pytestmark = pytest.mark.cpu_only

_LIMIT = 1 << 20


def _encode(value: object, **save_kwargs: object) -> str:
    """Return base64 ``torch.save`` bytes of ``value``."""
    buf = io.BytesIO()
    torch.save(value, buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _recompress(payload: str) -> str:
    """Rewrite a torch.save archive with every member deflated."""
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            dst.writestr(info.filename, src.read(info))
    return base64.b64encode(out.getvalue()).decode("ascii")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_round_trip(dtype: torch.dtype) -> None:
    """A saved tensor decodes to equal values on CPU."""
    tensor = torch.randn(5, 8).to(dtype)
    decoded = load_base64_tensor(_encode(tensor), max_bytes=_LIMIT)
    assert decoded.device.type == "cpu"
    assert decoded.dtype == dtype
    assert torch.equal(decoded, tensor)


def test_async_round_trip() -> None:
    """The async wrapper returns the same tensor."""
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    decoded = asyncio.run(async_load_base64_tensor(_encode(tensor), max_bytes=_LIMIT))
    assert torch.equal(decoded, tensor)


def test_encoded_length_checked_before_decode() -> None:
    """An over-long string is rejected without decoding it."""
    with pytest.raises(ValueError, match="base64 characters"):
        load_base64_tensor("A" * 5000, max_bytes=1024)


def test_decoded_size_over_limit() -> None:
    """A payload over the byte limit is rejected."""
    payload = _encode(torch.zeros(64, 64))
    with pytest.raises(ValueError, match="limit"):
        load_base64_tensor(payload, max_bytes=4096)


@pytest.mark.parametrize("payload", ["not base64!", "QUJD\nREVG"])
def test_invalid_base64(payload: str) -> None:
    """Non-alphabet characters and line breaks are rejected."""
    with pytest.raises(ValueError, match="not valid base64"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_non_string_payload() -> None:
    """Bytes are not accepted in place of base64 text."""
    with pytest.raises(TypeError, match="base64 string"):
        load_base64_tensor(b"QUJD", max_bytes=_LIMIT)


def test_legacy_format_rejected() -> None:
    """The pre-zip torch.save format declares storage sizes and is refused."""
    payload = _encode(torch.zeros(2, 2), _use_new_zipfile_serialization=False)
    with pytest.raises(ValueError, match="zip archive"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_compressed_members_rejected() -> None:
    """A deflated archive could expand past the limit and is refused."""
    payload = _recompress(_encode(torch.zeros(16, 16)))
    with pytest.raises(ValueError, match="uncompressed"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_arbitrary_pickle_rejected() -> None:
    """weights_only loading refuses objects outside the tensor allowlist."""
    payload = _encode({"cls": pickle.Pickler})
    with pytest.raises(ValueError, match="could not be loaded"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_non_tensor_rejected() -> None:
    """A container of tensors is not a single tensor."""
    with pytest.raises(TypeError, match="single tensor"):
        load_base64_tensor(_encode([torch.zeros(2)]), max_bytes=_LIMIT)


def test_sparse_rejected() -> None:
    """Sparse layouts are refused."""
    sparse = torch.zeros(4, 4).to_sparse()
    with pytest.raises((TypeError, ValueError)):
        load_base64_tensor(_encode(sparse), max_bytes=_LIMIT)
