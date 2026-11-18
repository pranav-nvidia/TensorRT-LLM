# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded decoding of base64 ``torch.save`` tensor payloads from clients."""

import asyncio
import base64
import collections
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
    """Globals outside one tensor's rebuild are refused before loading."""
    payload = _encode({"cls": pickle.Pickler})
    with pytest.raises(ValueError, match="unsupported pickle opcode|references"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


class _BigBytearray:
    """Pickles as ``bytearray(n)``: allowed by weights_only, allocates on load."""

    def __reduce__(self) -> tuple:
        return (bytearray, (1 << 30,))


def _replace_pickle(payload: str, pickled: bytes) -> str:
    """Swap the data.pkl record of a torch.save archive."""
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            data = pickled if info.filename.endswith("/data.pkl") else src.read(info)
            dst.writestr(info.filename, data)
    return base64.b64encode(out.getvalue()).decode("ascii")


def test_allocating_reducer_rejected() -> None:
    """A tiny pickle that would allocate 1 GiB while unpickling is refused unread."""
    payload = _replace_pickle(_encode(torch.zeros(2)), pickle.dumps(_BigBytearray(), protocol=2))
    with pytest.raises(ValueError, match="bytearray"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_expanded_view_over_logical_limit() -> None:
    """A view describing more elements than it stores is bounded by its logical size."""
    view = torch.zeros(1, 1024).expand(4096, 1024)
    with pytest.raises(ValueError, match="describes"):
        load_base64_tensor(_encode(view), max_bytes=_LIMIT)


def test_integer_tensor_rejected() -> None:
    """Only float16, bfloat16 and float32 storages are accepted."""
    with pytest.raises(ValueError, match="unsupported dtype"):
        load_base64_tensor(_encode(torch.zeros(2, 2, dtype=torch.int32)), max_bytes=_LIMIT)


def test_non_tensor_rejected() -> None:
    """A container of tensors is not a single tensor's pickle."""
    with pytest.raises(ValueError, match="unsupported pickle opcode"):
        load_base64_tensor(_encode([torch.zeros(2)]), max_bytes=_LIMIT)


def test_sparse_rejected() -> None:
    """Sparse layouts are refused."""
    sparse = torch.zeros(4, 4).to_sparse()
    with pytest.raises((TypeError, ValueError)):
        load_base64_tensor(_encode(sparse), max_bytes=_LIMIT)


def test_repeated_allowed_reducer_rejected_before_load(monkeypatch) -> None:
    """Copies of one memoized dict through OrderedDict amplify memory; refuse unread."""
    shared = {i: i for i in range(16)}
    copies = [collections.OrderedDict(shared) for _ in range(8)]
    payload = _replace_pickle(_encode(torch.zeros(2)), pickle.dumps(copies, protocol=2))

    def no_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch.load must not run on a rejected payload")

    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(ValueError, match="unsupported pickle opcode"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(4, 3)[::2],
        torch.zeros(3, 2).t(),
        torch.zeros(2, requires_grad=True),
        torch.zeros(1000, 64, dtype=torch.bfloat16),
    ],
)
def test_ordinary_saved_tensors_accepted(tensor: torch.Tensor) -> None:
    """Views, transposes, grad flags and large shapes still pass the pickle check."""
    decoded = load_base64_tensor(_encode(tensor), max_bytes=_LIMIT)
    assert torch.equal(decoded, tensor.detach())
