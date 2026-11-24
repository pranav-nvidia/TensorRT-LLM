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

from tensorrt_llm.inputs import utils
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


def test_encoded_length_checked_before_decode(monkeypatch) -> None:
    """An over-long string is rejected without decoding it."""

    def no_decode(*args, **kwargs) -> bytes:
        raise AssertionError("an over-long payload was decoded")

    monkeypatch.setattr(utils, "_b64decode_strict", no_decode)
    with pytest.raises(ValueError, match="base64 characters"):
        load_base64_tensor("A" * 5000, max_bytes=1024)


def test_decoded_size_over_limit() -> None:
    """Decoded bytes over the limit are rejected even when the text length passes."""
    with pytest.raises(ValueError, match="is 3 bytes, over the 2-byte limit"):
        load_base64_tensor("AAAA", max_bytes=2)


@pytest.mark.parametrize("payload", ["not base64!", "QUJD\nRE!G", "QUJD\u00e9"])
def test_invalid_base64(payload: str) -> None:
    """Non-alphabet characters are rejected, with or without line breaks."""
    with pytest.raises(ValueError, match="not valid base64"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_base64_decodes_without_strict_mode(monkeypatch) -> None:
    """Python 3.10 has no a2b_base64 strict_mode; validated b64decode is used instead."""
    calls = []
    b64decode = base64.b64decode

    def spy(text: str, validate: bool = False) -> bytes:
        """Record the call and decode as ``base64.b64decode`` does."""
        calls.append(validate)
        return b64decode(text, validate=validate)

    monkeypatch.setattr(utils.sys, "version_info", (3, 10, 0))
    monkeypatch.setattr(utils.base64, "b64decode", spy)
    tensor = torch.randn(4, 4)
    assert torch.equal(load_base64_tensor(_encode(tensor), max_bytes=_LIMIT), tensor)
    with pytest.raises(ValueError, match="not valid base64"):
        load_base64_tensor("not base64!", max_bytes=_LIMIT)
    assert calls and all(calls)


def test_line_wrapped_base64_near_limit_accepted() -> None:
    """Line breaks do not count toward the encoded-length limit."""
    tensor = torch.randn(1000)
    payload = _encode(tensor)
    wrapped = "\r\n".join(payload[i : i + 76] for i in range(0, len(payload), 76))
    limit = len(base64.b64decode(payload))
    assert len(wrapped) > 4 * ((limit + 2) // 3)
    assert torch.equal(load_base64_tensor(wrapped, max_bytes=limit), tensor)


def test_line_wrapped_base64_accepted() -> None:
    """MIME-style line-wrapped base64 decodes like the unwrapped text."""
    tensor = torch.randn(16, 8)
    payload = _encode(tensor)
    wrapped = "\r\n".join(payload[i : i + 76] for i in range(0, len(payload), 76))
    assert torch.equal(load_base64_tensor(wrapped, max_bytes=_LIMIT), tensor)


def test_non_string_payload() -> None:
    """Bytes are not accepted in place of base64 text."""
    with pytest.raises(TypeError, match="base64 string"):
        load_base64_tensor(b"QUJD", max_bytes=_LIMIT)


def test_legacy_format_rejected() -> None:
    """The pre-zip torch.save format declares storage sizes and is refused."""
    payload = _encode(torch.zeros(2, 2), _use_new_zipfile_serialization=False)
    with pytest.raises(ValueError, match="zip"):
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


@pytest.mark.parametrize("view_size", [(2, 8), (4096, 1024)])
def test_view_repeating_storage_rejected(view_size: tuple) -> None:
    """A view describing more elements than it stores is refused at any size."""
    view = torch.zeros(1, view_size[1]).expand(*view_size)
    with pytest.raises(ValueError, match="repeats storage|describes"):
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


def test_hidden_central_directory_rejected() -> None:
    """A second central directory, reachable only through stored offsets, is refused.

    The EOCD overstates the central directory size and pads with a comment, so
    zipfile reads the benign directory shifted back while miniz would read one
    hidden in the last entry's comment, pointing at a deflated 16 MiB storage.
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
    with pytest.raises(ValueError, match="no comment"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


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
    # Four rows of two at stride 3 cover 8 elements but end at element 10.
    payload = _replace_pickle(_encode(torch.zeros(8)), _rebuild_pickle(4, (3, 1), 8, wrap=False))
    with pytest.raises(ValueError, match="past its storage"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def _view_pickle(offset: int, size: tuple, stride: tuple, storage_numel: int) -> bytes:
    """A protocol-2 ``_rebuild_tensor_v2`` pickle of a float view with arbitrary ints."""

    def num(value: int) -> bytes:
        length = (value.bit_length() + 8) // 8
        return b"\x8a" + bytes([length]) + value.to_bytes(length, "little", signed=True)

    return (
        b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n((X\x07\x00\x00\x00storage"
        b"ctorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpu"
        + num(storage_numel)
        + b"tQ"
        + num(offset)
        + b"("
        + b"".join(num(v) for v in size)
        + b"t("
        + b"".join(num(v) for v in stride)
        + b"t\x89ccollections\nOrderedDict\n)RtR."
    )


@pytest.mark.parametrize(
    "offset, size, stride",
    [
        (0, (0, 2**64), (1, 1)),
        (0, (1, 4), (2**64, 1)),
        (2**64, (0,), (1,)),
    ],
)
def test_out_of_range_view_values_rejected(offset: int, size: tuple, stride: tuple) -> None:
    """Values the extent check skips (zero-sized or size-1 dims) are still bounded."""
    payload = _replace_pickle(_encode(torch.zeros(4)), _view_pickle(offset, size, stride, 4))
    with pytest.raises(ValueError, match="size, stride or offset"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_zero_sized_view_with_huge_dims_rejected() -> None:
    """Dims that each fit but multiply past the limit are refused despite a zero dim."""
    size = (1_000_000,) * 4 + (0,)
    payload = _replace_pickle(_encode(torch.zeros(4)), _view_pickle(0, size, (1,) * 5, 4))
    with pytest.raises(ValueError, match="multiply"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_base64_decoded_in_slices(monkeypatch) -> None:
    """Each base64 call sees a bounded slice, so a worker thread yields the GIL between them."""
    lengths = []
    a2b_base64 = utils.binascii.a2b_base64

    def spy(text: str, **kwargs: bool) -> bytes:
        """Record the slice length and decode as ``binascii.a2b_base64`` does."""
        lengths.append(len(text))
        return a2b_base64(text, **kwargs)

    monkeypatch.setattr(utils, "_BASE64_SLICE_CHARS", 64)
    monkeypatch.setattr(utils.binascii, "a2b_base64", spy)
    tensor = torch.randn(100)
    payload = _encode(tensor)
    wrapped = "\n".join(payload[i : i + 76] for i in range(0, len(payload), 76))
    for text in (payload, wrapped):
        assert torch.equal(load_base64_tensor(text, max_bytes=_LIMIT), tensor)
    # A slice may carry up to three characters of an incomplete quartet.
    assert len(lengths) > 2 * len(payload) // 64 and max(lengths) <= 64 + 3


def test_padding_before_the_end_rejected(monkeypatch) -> None:
    """Padding that ends a slice but not the payload is refused, as whole-text decoding does."""
    monkeypatch.setattr(utils, "_BASE64_SLICE_CHARS", 64)
    raw = base64.b64decode(_encode(torch.randn(100)))
    # 47 bytes encode to 64 characters ending in one "=", exactly one slice.
    head = base64.b64encode(raw[:47]).decode()
    assert len(head) == 64 and head.endswith("=")
    payload = head + base64.b64encode(raw[47:]).decode()
    with pytest.raises(ValueError, match="not valid base64"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


@pytest.mark.parametrize("tail", ["", "\n", "\r\n" * 40])
def test_sliced_decoding_matches_whole_text(monkeypatch, tail: str) -> None:
    """Sliced decoding agrees with whole-text decoding, padding at a slice end included."""
    monkeypatch.setattr(utils, "_BASE64_SLICE_CHARS", 64)
    for size in range(40, 52):
        raw = bytes(range(size))
        text = base64.b64encode(raw).decode()
        wrapped = "\n".join(text[i : i + 16] for i in range(0, len(text), 16))
        assert utils._b64decode_strict(text + tail, strip_whitespace=True) == raw
        assert utils._b64decode_strict(wrapped + tail, strip_whitespace=True) == raw
        if not tail:
            assert utils._b64decode_strict(text) == raw
        if text.endswith("="):
            with pytest.raises(ValueError):
                utils._b64decode_strict(text + tail + "QQ==", strip_whitespace=True)


def test_oversized_pickle_record_rejected_unread(monkeypatch) -> None:
    """A data.pkl record over the pickle limit is refused from its size, before it is read."""
    pickled = pickle.dumps(torch.zeros(1).tolist(), protocol=2)[:-1] + b"N" * 8192 + b"."
    payload = _replace_pickle(_encode(torch.zeros(2)), pickled)
    read = zipfile.ZipFile.read
    reads = []

    def spy(self: zipfile.ZipFile, name: str, *args: object) -> bytes:
        """Record which records are read."""
        reads.append(name)
        return read(self, name, *args)

    monkeypatch.setattr(zipfile.ZipFile, "read", spy)
    with pytest.raises(ValueError, match="pickle is"):
        load_base64_tensor(payload, max_bytes=_LIMIT)
    assert not any(name.endswith("data.pkl") for name in reads)


def _patch_record_sizes(payload: str, record: str, compress_size: int, file_size: int) -> str:
    """Overwrite one record's declared stored and uncompressed sizes in both its headers."""
    raw = bytearray(base64.b64decode(payload))
    name = f"archive/{record}".encode()
    info = zipfile.ZipFile(io.BytesIO(bytes(raw))).getinfo(name.decode())
    struct.pack_into("<II", raw, info.header_offset + 18, compress_size, file_size)
    entry = raw.find(b"PK\x01\x02")
    while entry != -1:
        name_len = struct.unpack_from("<H", raw, entry + 28)[0]
        if raw[entry + 46 : entry + 46 + name_len] == name:
            struct.pack_into("<II", raw, entry + 20, compress_size, file_size)
        entry = raw.find(b"PK\x01\x02", entry + 4)
    return base64.b64encode(bytes(raw)).decode()


def _spy_reads(monkeypatch) -> list:
    """Record the names of archive records the loader reads."""
    read = zipfile.ZipFile.read
    reads = []

    def spy(self: zipfile.ZipFile, name: str, *args: object) -> bytes:
        reads.append(name)
        return read(self, name, *args)

    monkeypatch.setattr(zipfile.ZipFile, "read", spy)
    return reads


@pytest.mark.parametrize("record", ["data.pkl", "byteorder"])
def test_record_sizes_must_agree(monkeypatch, record: str) -> None:
    """A stored record whose two declared sizes differ is refused before any record is read."""
    payload = _encode(torch.zeros(4096))
    info = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))).getinfo(f"archive/{record}")
    # zipfile would read compress_size bytes, past the small declared file_size.
    forged = _patch_record_sizes(payload, record, info.file_size + 4096, info.file_size)
    reads = _spy_reads(monkeypatch)
    with pytest.raises(ValueError, match="uncompressed"):
        load_base64_tensor(forged, max_bytes=_LIMIT)
    assert reads == []


@pytest.mark.parametrize("value", [b"little\n", b"bigend"])
def test_byteorder_record_must_be_little(monkeypatch, value: bytes) -> None:
    """A longer byteorder record is refused unread; a six-byte one must say little."""
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(_encode(torch.zeros(4)))))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            data = value if info.filename.endswith("/byteorder") else src.read(info)
            dst.writestr(info.filename, data)
    reads = _spy_reads(monkeypatch)
    with pytest.raises(ValueError, match="little-endian"):
        load_base64_tensor(base64.b64encode(out.getvalue()).decode(), max_bytes=_LIMIT)
    assert any(name.endswith("byteorder") for name in reads) == (len(value) == 6)


def test_empty_tensor_loads_on_cpu() -> None:
    """A tensor with no storage loads on CPU even when the default device is elsewhere."""
    payload = _encode(torch.zeros(0, 4))
    with torch.device("meta"):
        tensor = load_base64_tensor(payload, max_bytes=_LIMIT)
    assert tensor.device.type == "cpu" and tensor.shape == (0, 4)


def _patch_storage_record(payload: str, data: bytes, size: int, crc: int) -> str:
    """Overwrite the data/0 record's bytes and its declared stored size and CRC."""
    raw = bytearray(base64.b64decode(payload))
    info = zipfile.ZipFile(io.BytesIO(bytes(raw))).getinfo("archive/data/0")
    name_len, extra_len = struct.unpack_from("<HH", raw, info.header_offset + 26)
    start = info.header_offset + 30 + name_len + extra_len
    raw[start : start + len(data)] = data
    struct.pack_into("<II", raw, info.header_offset + 14, crc, size)
    entry = raw.find(b"PK\x01\x02")
    while entry != -1:
        name_len = struct.unpack_from("<H", raw, entry + 28)[0]
        if raw[entry + 46 : entry + 46 + name_len] == b"archive/data/0":
            struct.pack_into("<II", raw, entry + 16, crc, size)
        entry = raw.find(b"PK\x01\x02", entry + 4)
    return base64.b64encode(bytes(raw)).decode()


def test_truncated_storage_record_rejected() -> None:
    """A stored record shorter than its declared file size is refused from its sizes."""
    stored = struct.pack("<4f", 1, 2, 3, 4)
    payload = _patch_storage_record(_encode(torch.zeros(4)), stored, 4, zlib.crc32(stored[:4]))
    with pytest.raises(ValueError, match="uncompressed"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_storage_crc_mismatch_rejected() -> None:
    """Storage bytes that do not match the record's CRC are refused."""
    stored = struct.pack("<4f", 1, 2, 3, 4)
    payload = _patch_storage_record(_encode(torch.zeros(4)), stored, 16, zlib.crc32(bytes(16)))
    with pytest.raises(ValueError, match="CRC"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_zip64_record_away_from_locator_rejected() -> None:
    """A locator pointing anywhere but right before itself could name a decoy record."""
    raw = bytearray(base64.b64decode(_encode(torch.zeros(2))))
    locator = len(raw) - 22 - 20
    assert raw[locator : locator + 4] == b"PK\x06\x07"
    (reloff,) = struct.unpack_from("<Q", raw, locator + 8)
    struct.pack_into("<Q", raw, locator + 8, reloff - 8)
    with pytest.raises(ValueError, match="directly precede its locator"):
        load_base64_tensor(base64.b64encode(bytes(raw)).decode(), max_bytes=_LIMIT)


def test_declared_storage_past_archive_rejected() -> None:
    """A small archive declaring a large storage is refused before allocating it."""
    numel = 1 << 20
    payload = _replace_pickle(_encode(torch.zeros(4)), _view_pickle(0, (4,), (1,), numel))
    raw = bytearray(base64.b64decode(payload))
    info = zipfile.ZipFile(io.BytesIO(bytes(raw))).getinfo("archive/data/0")
    struct.pack_into("<II", raw, info.header_offset + 18, numel * 4, numel * 4)
    entry = raw.find(b"PK\x01\x02")
    while entry != -1:
        name_len = struct.unpack_from("<H", raw, entry + 28)[0]
        if raw[entry + 46 : entry + 46 + name_len] == b"archive/data/0":
            struct.pack_into("<II", raw, entry + 20, numel * 4, numel * 4)
        entry = raw.find(b"PK\x01\x02", entry + 4)
    with pytest.raises(ValueError, match="does not match its tensor"):
        load_base64_tensor(base64.b64encode(bytes(raw)).decode(), max_bytes=8 * numel)


def test_input_shorter_than_zip64_locator_rejected() -> None:
    """Locator bytes are not read from before the start of a short input."""
    eocd = struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 0, 0x4B50, 0x0706, 0, 0)
    with pytest.raises(ValueError):
        load_base64_tensor(base64.b64encode(bytes(8) + eocd).decode(), max_bytes=_LIMIT)


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
        ([f"junk/{i}" for i in range(20_000)], "central directory is over"),
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
        torch.arange(12.0).reshape(4, 3)[::2],
        torch.arange(12.0).reshape(4, 3)[1:, 1:],
        torch.arange(6.0).reshape(3, 2).t(),
        torch.arange(10.0)[3:7],
        torch.arange(4.0, requires_grad=True),
        torch.randn(1000, 64).to(torch.bfloat16),
    ],
)
def test_ordinary_saved_tensors_accepted(tensor: torch.Tensor) -> None:
    """Views, transposes, grad flags and large shapes still pass the pickle check."""
    decoded = load_base64_tensor(_encode(tensor), max_bytes=_LIMIT)
    assert torch.equal(decoded, tensor.detach())
