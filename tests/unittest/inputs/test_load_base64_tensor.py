# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded decoding of base64 ``torch.save`` tensor payloads from clients."""

import asyncio
import base64
import collections
import io
import pickle
import struct
import zipfile
import zlib

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


def test_repeated_allowed_reducer_rejected() -> None:
    """Copies of one memoized dict through OrderedDict are not one tensor's pickle."""
    shared = {i: i for i in range(16)}
    copies = [collections.OrderedDict(shared) for _ in range(8)]
    payload = _replace_pickle(_encode(torch.zeros(2)), pickle.dumps(copies, protocol=2))
    with pytest.raises(ValueError, match="unsupported pickle opcode"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_payload_is_never_unpickled(monkeypatch) -> None:
    """The tensor is built from checked fields; client bytes never reach torch.load."""

    def no_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch.load must not run on client bytes")

    monkeypatch.setattr(torch, "load", no_load)
    tensor = torch.randn(3, 4)
    assert torch.equal(load_base64_tensor(_encode(tensor), max_bytes=_LIMIT), tensor)


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_pickle_protocols_accepted(protocol: int) -> None:
    """torch.save with a newer pickle protocol (STACK_GLOBAL, framing) still decodes."""
    tensor = torch.randn(2, 3)
    payload = _encode(tensor, pickle_protocol=protocol)
    assert torch.equal(load_base64_tensor(payload, max_bytes=_LIMIT), tensor)


def _zip_member(name: str, data: bytes, offset: int, method: int = 0, size: int = 0) -> tuple:
    """Local and central directory headers of one record (``size`` for deflated data)."""
    raw = name.encode()
    crc = zlib.crc32(data if not method else bytes(size)) & 0xFFFFFFFF
    usize = size or len(data)
    local = struct.pack(
        "<IHHHHHIIIHH", 0x04034B50, 20, 0, method, 0, 0, crc, len(data), usize, len(raw), 0
    )
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50,
        20,
        20,
        0,
        method,
        0,
        0,
        crc,
        len(data),
        usize,
        len(raw),
        0,
        0,
        0,
        0,
        0,
        offset,
    )
    return local + raw + data, central + raw


def _benign_members() -> list:
    """The records torch.save writes for ``zeros(2, 2)``."""
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(_encode(torch.zeros(2, 2)))))
    return [(info.filename, src.read(info)) for info in src.infolist()]


def test_legacy_stream_before_zip_ignored() -> None:
    """A legacy torch.save stream at byte 0 ahead of a valid zip is never read."""
    legacy = io.BytesIO()
    torch.save(torch.arange(4), legacy, _use_new_zipfile_serialization=False)
    body, central = bytearray(legacy.getvalue()), bytearray()
    for name, data in _benign_members():
        # The version record claims offset 0, where torch.load would see the legacy magic.
        local, entry = _zip_member(name, data, 0 if name.endswith("/version") else len(body))
        body += local
        central += entry
    count = len(_benign_members())
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, count, count, len(central), len(body), 0)
    payload = base64.b64encode(bytes(body + central + eocd)).decode()
    decoded = load_base64_tensor(payload, max_bytes=_LIMIT)
    assert decoded.dtype == torch.float32 and torch.equal(decoded, torch.zeros(2, 2))


def test_hidden_central_directory_ignored() -> None:
    """A second central directory, reachable only through stored offsets, is never read.

    The EOCD overstates the central directory size, so zipfile reads the benign one
    shifted back while miniz would read one hidden in the last entry's comment,
    pointing at a deflated 16 MiB storage. The decoder reads only zipfile's view.
    """
    hidden_numel = 4 << 20
    pickled = _rebuild_pickle(hidden_numel // 2, (2, 1), hidden_numel, wrap=False)
    deflate = zlib.compressobj(9, zlib.DEFLATED, -15)
    compressed = deflate.compress(bytes(4 * hidden_numel)) + deflate.flush()
    body, visible, hidden = bytearray(), [], b""
    for name, data in _benign_members():
        visible.append((name, data, len(body)))
        body += _zip_member(name, data, 0)[0]
    for name, data, method, size in [
        ("archive/data.pkl", pickled, 0, 0),
        ("archive/version", b"3\n", 0, 0),
        ("archive/data/0", compressed, 8, 4 * hidden_numel),
    ]:
        local, entry = _zip_member(name, data, len(body), method, size)
        hidden += entry
        body += local
    entries = [_zip_member(name, data, 0)[1] for name, data, _ in visible]
    shift = sum(len(e) for e in entries)
    central = b""
    for i, (name, data, offset) in enumerate(visible):
        entry = bytearray(_zip_member(name, data, offset + shift)[1])
        if i == len(visible) - 1:
            entry[32:34] = struct.pack("<H", len(hidden))
            entry += hidden
        central += bytes(entry)
    eocd = struct.pack(
        "<IHHHHIIH", 0x06054B50, 0, 0, 3, 3, len(central), len(body) + shift, shift
    ) + bytes(shift)
    payload = base64.b64encode(bytes(body) + central + eocd).decode()
    decoded = load_base64_tensor(payload, max_bytes=_LIMIT)
    assert torch.equal(decoded, torch.zeros(2, 2))


def _rebuild_pickle(rows: int, stride: tuple, storage_numel: int, wrap: bool) -> bytes:
    """A protocol-2 ``_rebuild_tensor_v2`` pickle of a ``(rows, 2)`` float tensor.

    With ``wrap``, the tensor is passed to ``OrderedDict``, which iterates its rows
    while unpickling; its hooks are then an empty tuple, keeping the opcode counts
    of an ordinary ``torch.save`` pickle.
    """

    def tup(*items: int) -> bytes:
        return b"(" + b"".join(b"J" + i.to_bytes(4, "little") for i in items) + b"t"

    storage = (
        b"((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000"
        b"X\x03\x00\x00\x00cpu" + b"J" + storage_numel.to_bytes(4, "little") + b"tQ"
    )
    hooks = b")" if wrap else b"ccollections\nOrderedDict\n)R"
    rebuild = (
        b"ctorch._utils\n_rebuild_tensor_v2\n"
        + storage
        + b"K\x00"
        + tup(rows, 2)
        + tup(*stride)
        + b"\x89"
        + hooks
        + b"tR"
    )
    if wrap:
        return b"\x80\x02ccollections\nOrderedDict\n" + rebuild + b"\x85R."
    return b"\x80\x02" + rebuild + b"."


def test_reducer_wrapping_tensor_rejected() -> None:
    """``OrderedDict(tensor)`` would unbind 1e8 zero-stride rows; refuse unread."""
    pickled = _rebuild_pickle(100_000_000, (0, 0), 1, wrap=True)
    payload = _replace_pickle(_encode(torch.zeros(1)), pickled)
    with pytest.raises(ValueError, match="rebuild exactly one tensor"):
        load_base64_tensor(payload, max_bytes=4096)


def test_hand_built_pickle_accepted() -> None:
    """The hand-built rebuild pickle loads when it is well formed."""
    payload = _replace_pickle(_encode(torch.zeros(8)), _rebuild_pickle(4, (2, 1), 8, wrap=False))
    assert load_base64_tensor(payload, max_bytes=_LIMIT).shape == (4, 2)


def test_view_past_storage_rejected() -> None:
    """A size and stride reaching past the stored elements are refused unread."""
    payload = _replace_pickle(_encode(torch.zeros(8)), _rebuild_pickle(5, (2, 1), 8, wrap=False))
    with pytest.raises(ValueError, match="past its storage"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def _add_records(payload: str, names: list) -> str:
    """Append empty records to a torch.save archive."""
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            dst.writestr(info.filename, src.read(info))
        for name in names:
            dst.writestr(name, b"")
    return base64.b64encode(out.getvalue()).decode("ascii")


@pytest.mark.parametrize(
    "names, match",
    [
        ([f"junk/{i}" for i in range(20_000)], "too many records"),
        (["archive/notes"], "only the records"),
        (["archive/data/1"], "only the records"),
    ],
)
def test_extra_archive_records_rejected(names: list, match: str) -> None:
    """Records torch.save does not write, or too many of them, are refused unread."""
    payload = _add_records(_encode(torch.zeros(2)), names)
    with pytest.raises(ValueError, match=match):
        load_base64_tensor(payload, max_bytes=8 << 20)


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
