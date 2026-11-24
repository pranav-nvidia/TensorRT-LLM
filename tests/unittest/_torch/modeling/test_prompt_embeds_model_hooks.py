# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-family prompt_embeds handling: models that read token ids after the
input embedding treat an embedding row like one of their image positions."""

from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from tensorrt_llm._torch.models._arch_index import MODEL_ARCH_TO_MODULE
from tensorrt_llm._torch.models.modeling_exaone4_5 import Exaone4_5_ForConditionalGeneration
from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4AssistantForCausalLM, Gemma4ForCausalLM
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3VLForConditionalGeneration
from tensorrt_llm._torch.models.modeling_speculative import (
    Eagle3ForCausalLM,
    MistralLarge3EagleForCausalLM,
)
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModelForCausalLM,
    get_registered_model_class,
    prompt_embeds_unsupported_reason,
)

pytestmark = pytest.mark.cpu_only

IDS = torch.tensor([11, 0, 0, 14, 0], dtype=torch.int32)
ROWS = torch.tensor([1, 2, 4])


def _gemma4(ple_dim: int, pad_token_id: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(hidden_size_per_layer_input=ple_dim),
        config=SimpleNamespace(pad_token_id=pad_token_id),
    )


def test_plain_default() -> None:
    plain = SimpleNamespace()
    assert prompt_embeds_unsupported_reason(plain)  # Not opted in.
    assert DecoderModelForCausalLM.prompt_embeds_model_inputs(plain, IDS, ROWS) == {}


@pytest.mark.parametrize(
    "ple_dim, pad_token_id, ple_ids",
    [(32, 7, [11, 7, 7, 14, 7]), (0, None, None), (32, None, "refused")],
    ids=["ple_pads_embedding_rows", "no_ple", "ple_without_pad"],
)
def test_gemma4(
    ple_dim: int, pad_token_id: int | None, ple_ids: list[int] | Literal["refused"] | None
) -> None:
    model = _gemma4(ple_dim, pad_token_id)
    reason = Gemma4ForCausalLM.prompt_embeds_config_unsupported_reason(model)
    if ple_ids == "refused":
        assert reason
        return
    assert reason is None
    extra = Gemma4ForCausalLM.prompt_embeds_model_inputs(model, IDS, ROWS)
    assert (extra["ple_input_ids"].tolist() if extra else None) == ple_ids
    # Plain rows keep their ids, so plain requests batched alongside are unaffected.
    assert IDS.tolist() == [11, 0, 0, 14, 0]


@pytest.mark.parametrize(
    "cls",
    [
        Eagle3ForCausalLM,
        MistralLarge3EagleForCausalLM,
        Gemma4AssistantForCausalLM,
        MiniMaxM3VLForConditionalGeneration,
        Exaone4_5_ForConditionalGeneration,
        # Any modeling_v2 target (their libraries build only for their own SMs).
        type(
            "ModelingV2Target",
            (DecoderModelForCausalLM,),
            {
                "__module__": "tensorrt_llm._torch.modeling_v2.models.family.target.modeling",
                "supports_prompt_embeds": True,
            },
        ),
    ],
)
def test_refused_families(cls: type) -> None:
    assert prompt_embeds_unsupported_reason(object.__new__(cls))


# Registered architectures whose forward and id-reading layers were checked with
# prompt_embeds. A new subclass of one of them inherits the opt-in and must be
# checked and listed here, or opt out.
VERIFIED_ARCHITECTURES = {
    "AfmoeForCausalLM",
    "Apertus1p5ForConditionalGeneration",
    "ApertusForCausalLM",
    "Cohere2ForCausalLM",
    "Cosmos3EdgeForCausalLM",
    "Cosmos3EdgeForConditionalGeneration",
    "Cosmos3ForConditionalGeneration",
    "DeciLMForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Exaone4ForCausalLM",
    "ExaoneMoeForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma4ForCausalLM",
    "Glm4MoeForCausalLM",
    "Glm5NextForCausalLM",
    "GlmMoeDsaForCausalLM",
    "GptOssForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "KimiLinearForCausalLM",
    "LagunaForCausalLM",
    "Llama4ForConditionalGeneration",
    "LlamaForCausalLM",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM3SparseForCausalLM",
    "MistralForCausalLM",
    "MistralLarge3ForCausalLM",
    "MixtralForCausalLM",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "NemotronHPuzzleForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen2VLForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
    "SeedOssForCausalLM",
    "Starcoder2ForCausalLM",
    "Step3p5ForCausalLM",
}


def test_opted_in_architectures_are_verified() -> None:
    opted_in = {
        arch
        for arch in MODEL_ARCH_TO_MODULE
        if getattr(get_registered_model_class(arch), "supports_prompt_embeds", False)
    }
    assert opted_in == VERIFIED_ARCHITECTURES
