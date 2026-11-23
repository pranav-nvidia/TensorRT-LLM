# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precomputed image / video embeddings for the Qwen-VL family (CPU)."""

import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5VLInputProcessorBase
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase
from tensorrt_llm.inputs.external_embeddings import ExternalEmbeddingItem
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

_VISION_START = 151652
_VISION_END = 151653
_IMAGE_PAD = 151655
_VIDEO_PAD = 151656
_SPECIAL_IDS = {
    "<|vision_start|>": _VISION_START,
    "<|vision_end|>": _VISION_END,
    "<|image_pad|>": _IMAGE_PAD,
    "<|video_pad|>": _VIDEO_PAD,
}
_HIDDEN = 8


class _FakeTokenizer:
    """Maps special tokens to their ids, each timestamp to one id, other text to 7."""

    _PIECE_RE = re.compile(r"<\|[a-z_]+\|>|<[0-9.]+ seconds>|[^<]+")

    def __call__(self, text: str, return_tensors: str = None) -> SimpleNamespace:
        """Tokenize ``text`` into a [1, n] id tensor."""
        ids = [self.piece_id(piece) for piece in self._PIECE_RE.findall(text)]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids: list) -> str:
        """Inverse of the special-token mapping, enough for token-id prompts."""
        inverse = {v: k for k, v in _SPECIAL_IDS.items()}
        return "".join(inverse.get(i, "x") for i in ids)

    @staticmethod
    def piece_id(piece: str) -> int:
        """Id of one piece."""
        if piece in _SPECIAL_IDS:
            return _SPECIAL_IDS[piece]
        match = re.fullmatch(r"<([0-9.]+) seconds>", piece)
        if match:
            return 1000 + round(float(match.group(1)) * 10)
        return 7


def _make_processor(cls: type, deepstack_levels: int = 0, tokens_per_second=None):
    """Bare processor with a minimal Qwen config and the fake tokenizer."""
    processor = object.__new__(cls)
    vision_config = SimpleNamespace(
        spatial_merge_size=2,
        temporal_patch_size=2,
        deepstack_visual_indexes=list(range(deepstack_levels)),
    )
    if tokens_per_second is not None:
        vision_config.tokens_per_second = tokens_per_second
    processor._config = SimpleNamespace(
        image_token_id=_IMAGE_PAD,
        video_token_id=_VIDEO_PAD,
        vision_start_token_id=_VISION_START,
        vision_end_token_id=_VISION_END,
        text_config=SimpleNamespace(
            hidden_size=_HIDDEN, vocab_size=200000, max_position_embeddings=128000
        ),
        vision_config=vision_config,
    )
    processor._processor = SimpleNamespace()
    processor._tokenizer = _FakeTokenizer()
    return processor


def _attach(processor, prompt, embeddings, **extra):
    """Run the public attach hook and wrap the result as executor params."""
    tokens, extra_inputs = processor.attach_multimodal_embeddings(
        {"prompt": prompt, **extra}, embeddings, SamplingParams()
    )
    return tokens, MultimodalParams(**extra_inputs)


def _video(t, h, w, width=_HIDDEN, seed=0, **meta):
    rows = t * (h // 2) * (w // 2)
    generator = torch.Generator().manual_seed(seed)
    return {
        "embedding": torch.randn(rows, width, generator=generator),
        "video_grid_thw": [t, h, w],
        **meta,
    }


def test_qwen3_video_prompt_has_per_frame_timestamps_and_runs():
    processor = _make_processor(Qwen3VLInputProcessorBase)
    video = _video(2, 4, 2, timestamps=[0.25, 1.25])
    tokens, params = _attach(
        processor, "a<|vision_start|><|video_pad|><|vision_end|>b", {"video": [video]}
    )

    frame = [_VISION_START, _VIDEO_PAD, _VIDEO_PAD, _VISION_END]
    # HF rounds timestamps with "{:.1f}".
    assert tokens == [7, 1002, *frame, 1012, *frame, 7]
    mm_input = params.multimodal_input
    assert mm_input.multimodal_positions == [3]
    assert mm_input.multimodal_lengths == [4]
    assert mm_input.multimodal_run_positions == [3, 8]
    assert mm_input.multimodal_run_lengths == [2, 2]
    assert params.get_item_layout() == ((("video", 0),), (4,))
    assert params.multimodal_data["multimodal_embedding"][0] is video["embedding"]
    # Qwen3 MRoPE: each frame is a (1, h=2, w=1) block placed after its timestamp
    # and vision-start text tokens; text resumes at the block maximum + 1.
    expected = [
        [0, 1, 2, 3, 3, 5, 6, 7, 8, 8, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 3, 5, 6, 7, 8, 8, 10, 11],
    ]
    mrope = params.multimodal_data["mrope_config"]
    assert mrope["mrope_position_ids"][:, 0, :].tolist() == expected
    assert mrope["mrope_position_deltas"].tolist() == [[0]]


def test_qwen3_timestamps_from_fps_and_frame_indices_match_hf():
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    processor = _make_processor(Qwen3VLInputProcessorBase)
    indices = [0, 7, 15, 22, 30]  # odd count: HF pads with the last index
    hf = Qwen3VLProcessor._calculate_timestamps(None, list(indices), 29.97, 2)
    item = ExternalEmbeddingItem(
        "video", torch.zeros(3, _HIDDEN), (3, 2, 2), fps=29.97, frames_indices=tuple(indices)
    )
    assert processor._resolve_external_video_metadata(item)["timestamps"] == hf


def test_mixed_image_and_video_follow_prompt_order():
    processor = _make_processor(Qwen3VLInputProcessorBase, deepstack_levels=1)
    width = 2 * _HIDDEN
    image = {"embedding": torch.ones(2, width), "image_grid_thw": [1, 2, 4]}
    video = _video(1, 2, 2, width=width, timestamps=[0.0])
    prompt = (
        "<|vision_start|><|video_pad|><|vision_end|>x<|vision_start|><|image_pad|><|vision_end|>"
    )
    tokens, params = _attach(processor, prompt, {"image": [image], "video": [video]})

    assert tokens == [
        1000,
        _VISION_START,
        _VIDEO_PAD,
        _VISION_END,
        7,
        _VISION_START,
        _IMAGE_PAD,
        _IMAGE_PAD,
        _VISION_END,
    ]
    assert params.get_item_layout() == ((("video", 0), ("image", 0)), (1, 2))
    assert params.multimodal_input.multimodal_positions == [2, 6]
    embeddings = params.multimodal_data["multimodal_embedding"]
    assert embeddings[0] is video["embedding"] and embeddings[1] is image["embedding"]


def test_mixed_bare_and_framed_video_placeholders_are_rejected():
    processor = _make_processor(Qwen3VLInputProcessorBase)
    videos = [_video(1, 2, 2, timestamps=[0.0]), _video(1, 2, 2, timestamps=[1.0])]
    with pytest.raises(ValueError, match="framed"):
        _attach(
            processor,
            "<|video_pad|><|vision_start|><|video_pad|><|vision_end|>",
            {"video": videos},
        )


def test_non_square_image_grid_drives_mrope():
    processor = _make_processor(Qwen3VLInputProcessorBase)
    image = {"embedding": torch.ones(2, _HIDDEN), "image_grid_thw": [1, 2, 4]}
    tokens, params = _attach(
        processor, "<|vision_start|><|image_pad|><|vision_end|>", {"image": [image]}
    )
    mrope = params.multimodal_data["mrope_config"]["mrope_position_ids"]
    # One row, two columns: height index constant, width index increments.
    assert mrope[:, 0, 1:3].tolist() == [[1, 1], [1, 1], [1, 2]]


def test_qwen3_5_config_takes_hidden_size_rows():
    """Qwen3.5 vision configs have no deepstack levels, so rows are hidden-size wide."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    config = Qwen3_5Config(
        text_config={"hidden_size": _HIDDEN, "vocab_size": 200000},
        vision_config={"out_hidden_size": _HIDDEN},
        image_token_id=_IMAGE_PAD,
        video_token_id=_VIDEO_PAD,
        vision_start_token_id=_VISION_START,
        vision_end_token_id=_VISION_END,
    )
    assert getattr(config.vision_config, "deepstack_visual_indexes", None) is None
    processor = _make_processor(Qwen3VLInputProcessorBase)
    processor._config = config
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    image = {"embedding": torch.ones(2, _HIDDEN), "image_grid_thw": [1, 2, 4]}
    tokens, _ = _attach(processor, prompt, {"image": [image]})
    assert tokens.count(_IMAGE_PAD) == 2
    wide = {"embedding": torch.ones(2, 2 * _HIDDEN), "image_grid_thw": [1, 2, 4]}
    with pytest.raises(ValueError, match="width"):
        _attach(processor, prompt, {"image": [wide]})


def test_bare_image_tensor_still_accepted_and_warns():
    processor = _make_processor(Qwen3VLInputProcessorBase)
    with patch("tensorrt_llm._torch.models.modeling_qwen2vl.logger.warning_once") as warn:
        tokens, params = _attach(processor, "<|image_pad|>", {"image": torch.ones(4, _HIDDEN)})
    warn.assert_called_once()
    assert tokens == [_IMAGE_PAD] * 4
    assert params.multimodal_input is not None


def test_qwen2_5_video_scales_time_by_second_per_grid():
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    video = _video(2, 2, 2, fps=1.0)  # second_per_grid_ts = temporal_patch_size / fps = 2
    tokens, params = _attach(
        processor, "<|vision_start|><|video_pad|><|vision_end|>", {"video": [video]}
    )
    assert tokens == [_VISION_START, _VIDEO_PAD, _VIDEO_PAD, _VISION_END]
    mrope = params.multimodal_data["mrope_config"]["mrope_position_ids"]
    # Temporal index advances by second_per_grid * tokens_per_second = 4 per patch.
    assert mrope[0, 0, 1:3].tolist() == [1, 5]


@pytest.mark.parametrize(
    "timing, match",
    [
        ({}, "exactly one of"),
        ({"fps": 2.0, "second_per_grid_ts": 1.0}, "exactly one of"),
        ({"timestamps": [0.0, 1.0]}, "not 'timestamps'"),
    ],
)
def test_qwen2_5_video_requires_one_time_scale(timing, match):
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    with pytest.raises(ValueError, match=match):
        _attach(processor, "<|video_pad|>", {"video": [_video(2, 2, 2, **timing)]})


def _attach_framed_video(processor, video: dict, max_tokens: int) -> tuple:
    """Attach one framed video with a given generation length."""
    return processor.attach_multimodal_embeddings(
        {"prompt": "<|vision_start|><|video_pad|><|vision_end|>"},
        {"video": [video]},
        SamplingParams(max_tokens=max_tokens),
    )


@pytest.mark.parametrize(
    "timing, match",
    [({"second_per_grid_ts": 1e5}, "positions exceed"), ({"fps": 1e-300}, "overflows float32")],
)
def test_qwen2_5_video_positions_past_rotary_table_are_rejected(timing, match):
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    with pytest.raises(ValueError, match=match):
        _attach_framed_video(processor, _video(8, 2, 2, **timing), max_tokens=1)


@pytest.mark.parametrize("max_tokens", [32, None])
def test_qwen2_5_video_decode_positions_stay_in_rotary_table(max_tokens):
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    # The last video position is 1 + 7 * 9141 * 2 = 127975, below 128000.
    video = _video(8, 2, 2, second_per_grid_ts=9141.0)
    _attach_framed_video(processor, video, max_tokens=1)
    with pytest.raises(ValueError, match="set max_tokens to at most 23$"):
        _attach_framed_video(processor, video, max_tokens=max_tokens)


def test_qwen2_5_decode_bound_clamps_max_tokens_to_context():
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    # Positions trail the token count, so any max_tokens fits once clamped.
    video = _video(2, 8, 8, second_per_grid_ts=1.0)
    _attach_framed_video(processor, video, max_tokens=10**9)


@pytest.mark.parametrize("timestamps", [[0.0, 1e308], [-1.0, 0.0]])
def test_qwen3_out_of_range_timestamps_are_rejected(timestamps):
    processor = _make_processor(Qwen3VLInputProcessorBase)
    with pytest.raises(ValueError, match=r"\[0, 1e6\] seconds"):
        _attach_framed_video(processor, _video(2, 4, 2, timestamps=timestamps), max_tokens=1)


def test_qwen2_5_bare_video_placeholder_is_rejected():
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    with pytest.raises(ValueError, match="framed"):
        _attach(processor, "<|video_pad|>", {"video": [_video(2, 2, 2, fps=2.0)]})


@pytest.mark.parametrize(
    "embeddings, prompt, match",
    [
        ({"video": [_video(2, 4, 2)]}, "<|video_pad|>", "timestamps"),
        ({"video": [_video(2, 4, 2, timestamps=[0.0])]}, "<|video_pad|>", "expected 2 timestamps"),
        (
            {"video": [{"embedding": torch.ones(4, _HIDDEN), "video_grid_thw": [2, 4, 3]}]},
            "<|video_pad|>",
            "multiples of the spatial merge size",
        ),
        (
            {"video": [{"embedding": torch.ones(3, _HIDDEN), "video_grid_thw": [2, 4, 2]}]},
            "<|video_pad|>",
            "has 3 rows but its grid",
        ),
        ({"video": [{"embedding": torch.ones(4, _HIDDEN)}]}, "<|video_pad|>", "video_grid_thw"),
        ({"image": [torch.ones(4, _HIDDEN + 1)]}, "<|image_pad|>", "width"),
        ({"image": [torch.ones(4, _HIDDEN)]}, "no placeholder", "0 image placeholders"),
        ({"image": [torch.ones(4, _HIDDEN)]}, "<|image_pad|><|image_pad|>", "more image"),
        ({"image": [torch.ones(4, _HIDDEN, dtype=torch.int32)]}, "<|image_pad|>", "floating"),
        ({"audio": [torch.ones(4, _HIDDEN)]}, "<|image_pad|>", "Unsupported modalities"),
        (
            {"image": [{"embedding": torch.ones(4, _HIDDEN), "timestamps": [0.0]}]},
            "<|image_pad|>",
            "only applies to video",
        ),
    ],
)
def test_invalid_embeddings_are_rejected(embeddings, prompt, match):
    processor = _make_processor(Qwen3VLInputProcessorBase)
    with pytest.raises((ValueError, TypeError), match=match):
        _attach(processor, prompt, embeddings)


def _hashes(processor, prompt, embeddings, **extra):
    """Prompt tokens and per-item cache keys of one attached request."""
    tokens, params = _attach(processor, prompt, embeddings, **extra)
    return tokens, params.multimodal_input.multimodal_hashes


def test_position_metadata_joins_the_cache_key():
    """Same rows and prompt tokens but different MRoPE inputs never share KV."""
    rows = torch.ones(4, _HIDDEN)
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    qwen3 = _make_processor(Qwen3VLInputProcessorBase)
    square = _hashes(qwen3, image, {"image": [{"embedding": rows, "image_grid_thw": [1, 4, 4]}]})
    wide = _hashes(qwen3, image, {"image": [{"embedding": rows, "image_grid_thw": [1, 2, 8]}]})
    assert square[0] == wide[0]
    assert square[1] != wide[1]

    video = "<|vision_start|><|video_pad|><|vision_end|>"
    qwen2_5 = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)

    def clip(**timing):
        return {"video": [{"embedding": rows, "video_grid_thw": [2, 2, 4], **timing}]}

    slow = _hashes(qwen2_5, video, clip(fps=1.0))
    fast = _hashes(qwen2_5, video, clip(fps=2.0))
    assert slow[0] == fast[0]
    assert slow[1] != fast[1]
    # fps resolves to second_per_grid_ts = temporal_patch_size / fps.
    assert _hashes(qwen2_5, video, clip(second_per_grid_ts=2.0)) == slow

    # Qwen3 prints frame timestamps into the prompt, so they change the tokens.
    early = _hashes(qwen3, video, clip(timestamps=[0.0, 1.0]))
    late = _hashes(qwen3, video, clip(timestamps=[0.0, 2.0]))
    assert early[0] != late[0]


def test_uuid_joins_the_cache_key_and_absent_modalities_are_ignored():
    processor = _make_processor(Qwen3VLInputProcessorBase)
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    image = {"image": [{"embedding": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}]}
    _, plain = _hashes(processor, prompt, image)
    _, tagged = _hashes(processor, prompt, image, multi_modal_uuids={"image": ["img-1"]})
    assert tagged != plain
    # As for raw media, a UUID key with no items of that modality is ignored.
    _, other = _hashes(processor, prompt, image, multi_modal_uuids={"video": ["clip-1"]})
    assert other == plain


def test_two_embeddings_with_same_prompt_get_different_cache_keys():
    processor = _make_processor(Qwen3VLInputProcessorBase)
    prompt = "<|vision_start|><|image_pad|><|vision_end|>describe"
    first = {"embedding": torch.zeros(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}
    second = {"embedding": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}
    tokens_a, params_a = _attach(processor, prompt, {"image": [first]})
    tokens_b, params_b = _attach(processor, prompt, {"image": [second]})
    assert tokens_a == tokens_b
    assert (
        params_a.multimodal_input.multimodal_hashes != params_b.multimodal_input.multimodal_hashes
    )


def test_llm_preprocess_carries_hashes_and_uuids_from_token_ids():
    processor = _make_processor(Qwen3VLInputProcessorBase)
    llm = object.__new__(BaseLLM)
    llm.input_processor = processor
    llm._hf_model_config = SimpleNamespace(is_encoder_decoder=False)
    prompt_ids = [7, _VISION_START, _IMAGE_PAD, _VISION_END, 7]
    inputs = {
        "prompt_token_ids": prompt_ids,
        "multi_modal_embeddings": {
            "image": [{"embedding": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}]
        },
        "multi_modal_uuids": {"image": ["img-1"]},
    }
    tokens, _, params, _ = llm._preprocess(inputs, SamplingParams(), None)

    assert tokens == [7, _VISION_START, *([_IMAGE_PAD] * 4), _VISION_END, 7]
    mm_input = params.multimodal_input
    assert mm_input is not None
    assert mm_input.multimodal_uuids == ["img-1"]
    _, from_text = _hashes(
        processor,
        "x<|vision_start|><|image_pad|><|vision_end|>x",
        inputs["multi_modal_embeddings"],
        multi_modal_uuids={"image": ["img-1"]},
    )
    assert mm_input.multimodal_hashes == from_text
    assert "mrope_config" in params.multimodal_data
    assert "multimodal_embed_mask_cumsum" in params.multimodal_data
