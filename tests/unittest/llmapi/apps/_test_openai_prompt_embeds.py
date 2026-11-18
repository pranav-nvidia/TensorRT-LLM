# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-serve ``prompt_embeds``: embeddings of a prompt's tokens generate what the tokens do."""

import base64
import io
import os
from typing import Any, Dict, List

import openai
import pytest
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

MAX_TOKENS = 16


@pytest.fixture(scope="module")
def model_path() -> str:
    """Qwen3-0.6B checkpoint directory."""
    return get_model_path("Qwen3/Qwen3-0.6B")


@pytest.fixture(scope="module")
def server(model_path: str) -> RemoteOpenAIServer:
    """A server with prompt_embeds enabled."""
    with RemoteOpenAIServer(model_path, ["--enable_prompt_embeds"]) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    """OpenAI client for the server."""
    return server.get_client()


@pytest.fixture(scope="module")
def tokenizer(model_path: str) -> Any:
    """The model's tokenizer."""
    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="module")
def embed_table(model_path: str) -> torch.Tensor:
    """The model's input embedding table."""
    with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt") as f:
        return f.get_tensor("model.embed_tokens.weight")


def _b64(tensor: torch.Tensor) -> str:
    """Encode a tensor as a prompt_embeds payload."""
    buf = io.BytesIO()
    torch.save(tensor.contiguous(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _greedy() -> Dict[str, Any]:
    """Sampling arguments for a deterministic comparison."""
    return {"max_tokens": MAX_TOKENS, "temperature": 0.0}


def _complete(client: openai.OpenAI, model: str, **kwargs: Any) -> Any:
    """Call /v1/completions, passing prompt_embeds through extra_body."""
    extra = {}
    if "prompt_embeds" in kwargs:
        extra["prompt_embeds"] = kwargs.pop("prompt_embeds")
        kwargs.setdefault("prompt", None)
    return client.completions.create(model=model, extra_body=extra, **_greedy(), **kwargs)


def test_completion_matches_token_prompt(
    client: openai.OpenAI, model_path: str, tokenizer: Any, embed_table: torch.Tensor
) -> None:
    """A whole-prompt payload of token embeddings reproduces the token prompt."""
    ids: List[int] = tokenizer.encode("The capital of France is", add_special_tokens=False)
    by_ids = client.completions.create(model=model_path, prompt=ids, **_greedy())
    by_embeds = _complete(client, model_path, prompt_embeds=_b64(embed_table[ids]))
    assert by_embeds.choices[0].text == by_ids.choices[0].text
    assert by_embeds.usage.prompt_tokens == len(ids)


def test_completion_batch_and_stream(
    client: openai.OpenAI, model_path: str, tokenizer: Any, embed_table: torch.Tensor
) -> None:
    """Several payloads in one request, streamed, each reproduce their token prompt."""
    texts = ["Water boils at a temperature of", "The largest planet in the solar system is"]
    id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    expected = [
        client.completions.create(model=model_path, prompt=ids, **_greedy()).choices[0].text
        for ids in id_lists
    ]
    stream = client.completions.create(
        model=model_path,
        prompt=None,
        stream=True,
        extra_body={"prompt_embeds": [_b64(embed_table[ids]) for ids in id_lists]},
        **_greedy(),
    )
    got = ["", ""]
    for chunk in stream:
        for choice in chunk.choices:
            got[choice.index] += choice.text
    assert got == expected


def test_chat_part_matches_text(
    client: openai.OpenAI, model_path: str, tokenizer: Any, embed_table: torch.Tensor
) -> None:
    """A chat part holding a message's token embeddings reproduces the text message."""
    question = "What is the capital of Japan? Answer in one word."
    ids = tokenizer.encode(question, add_special_tokens=False)
    by_text = client.chat.completions.create(
        model=model_path, messages=[{"role": "user", "content": question}], **_greedy()
    )
    part = {"type": "prompt_embeds", "prompt_embeds": {"data": _b64(embed_table[ids])}}
    by_embeds = client.chat.completions.create(
        model=model_path, messages=[{"role": "user", "content": [part]}], **_greedy()
    )
    assert by_embeds.choices[0].message.content == by_text.choices[0].message.content
    assert by_embeds.usage.prompt_tokens == by_text.usage.prompt_tokens


def test_completion_wrong_hidden_size(client: openai.OpenAI, model_path: str) -> None:
    """A payload of the wrong width is a client error."""
    with pytest.raises(openai.BadRequestError, match="hidden size"):
        _complete(client, model_path, prompt_embeds=_b64(torch.zeros(4, 7, dtype=torch.bfloat16)))
