# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prompt_embeds`` on /v1/completions and as a chat content part (CPU, fake generator)."""

import base64
import io
from typing import Any, Dict, Iterator, List

import pytest
import torch
from test_common.serve.app_factory import AggConfig, ServeTestEnv, isolated_serve_env

from tensorrt_llm.serve.prompt_embeds import (
    DISABLED_MESSAGE,
    PROMPT_EMBEDS_MAX_BYTES_ENV,
    ChatPromptEmbedsPart,
    model_hidden_size,
    splice_chat_prompt_embeds,
)
from tensorrt_llm.serve.render.chat import render_chat
from tensorrt_llm.serve.render.prompt_types import UnsupportedRenderError
from tensorrt_llm.serve.render.resources import RenderResources

# The server's input-processing pool thread outlives a test by design.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

# Divisible by the default attention head count, so the fake config loads.
HIDDEN = 64


def _b64(tensor: torch.Tensor) -> str:
    """Encode a tensor the way clients send prompt_embeds."""
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _part(tensor: torch.Tensor) -> Dict[str, Any]:
    """A chat content part carrying ``tensor``."""
    return {"type": "prompt_embeds", "prompt_embeds": {"data": _b64(tensor)}}


@pytest.fixture
def serve_env(tmp_path) -> Iterator[ServeTestEnv]:
    """An isolated serve environment."""
    with isolated_serve_env(tmp_path) as env:
        yield env


def _client(env: ServeTestEnv, enabled: bool = True):
    """Return (server, client) for a fake LLM server."""
    server = env.build_openai_server(AggConfig(enable_prompt_embeds=enabled, hidden_size=HIDDEN))
    return server, env.client(server.app)


def _completion(client, **body: Any):
    """POST /v1/completions."""
    return client.post("/v1/completions", json={"model": "fake-model", "max_tokens": 1, **body})


def _chat(client, messages: List[Dict[str, Any]], **body: Any):
    """POST /v1/chat/completions."""
    return client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "max_tokens": 1, "messages": messages, **body},
    )


def test_completion_rejected_when_disabled(serve_env: ServeTestEnv) -> None:
    """Without --enable_prompt_embeds the server refuses tensor payloads."""
    server, client = _client(serve_env, enabled=False)
    with client:
        response = _completion(client, prompt_embeds=_b64(torch.zeros(3, HIDDEN)))
    assert response.status_code == 400
    assert DISABLED_MESSAGE in response.json()["message"]
    assert server.generator.generate_calls == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_completion_single_payload(serve_env: ServeTestEnv, dtype: torch.dtype) -> None:
    """One payload becomes one whole-prompt embeddings request."""
    rows = torch.randn(3, HIDDEN).to(dtype)
    server, client = _client(serve_env)
    with client:
        response = _completion(client, prompt_embeds=_b64(rows))
    assert response.status_code == 200, response.text
    (call,) = server.generator.generate_calls
    assert set(call.inputs) == {"prompt_embeds"}
    assert call.inputs["prompt_embeds"].dtype == dtype
    assert torch.equal(call.inputs["prompt_embeds"], rows)
    assert response.json()["usage"]["prompt_tokens"] == 3


def test_completion_payloads_then_text_prompts(serve_env: ServeTestEnv) -> None:
    """Embedding prompts are served before text prompts, in order."""
    first, second = torch.randn(2, HIDDEN), torch.randn(4, HIDDEN)
    server, client = _client(serve_env)
    with client:
        response = _completion(client, prompt_embeds=[_b64(first), _b64(second)], prompt="hi")
    assert response.status_code == 200, response.text
    calls = server.generator.generate_calls
    assert len(calls) == 3
    assert torch.equal(calls[0].inputs["prompt_embeds"], first)
    assert torch.equal(calls[1].inputs["prompt_embeds"], second)
    assert "prompt_embeds" not in calls[2].inputs
    assert [c["index"] for c in response.json()["choices"]] == [0, 1, 2]


@pytest.mark.parametrize(
    ("tensor", "match"),
    [
        (torch.zeros(3, HIDDEN + 1), "hidden size"),
        (torch.zeros(3, HIDDEN, 1), "shape"),
        (torch.zeros(3, HIDDEN, dtype=torch.int32), "dtype"),
        (torch.zeros(0, HIDDEN), "no rows"),
    ],
)
def test_completion_bad_tensor(serve_env: ServeTestEnv, tensor: torch.Tensor, match: str) -> None:
    """Shape, dtype and width are checked before the request is submitted."""
    server, client = _client(serve_env)
    with client:
        assert model_hidden_size(server.model_config) == HIDDEN
        response = _completion(client, prompt_embeds=_b64(tensor))
    assert response.status_code == 400
    assert match in response.json()["message"]
    assert server.generator.generate_calls == []


def test_completion_malformed_payload(serve_env: ServeTestEnv) -> None:
    """A payload that is not base64 torch.save is a client error."""
    server, client = _client(serve_env)
    with client:
        response = _completion(client, prompt_embeds="not base64!")
    assert response.status_code == 400
    assert server.generator.generate_calls == []


def test_completion_request_byte_limit(serve_env: ServeTestEnv, monkeypatch) -> None:
    """Payloads of one request share the byte limit."""
    monkeypatch.setenv(PROMPT_EMBEDS_MAX_BYTES_ENV, "10000")
    payload = _b64(torch.zeros(16, HIDDEN))  # ~5 KB each
    server, client = _client(serve_env)
    with client:
        assert _completion(client, prompt_embeds=payload).status_code == 200
        response = _completion(client, prompt_embeds=[payload, payload, payload])
    assert response.status_code == 400
    assert "limit" in response.json()["message"]
    assert len(server.generator.generate_calls) == 1


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt_embeds": []},
        {"prompt_embeds": "AAAA", "echo": True},
        {"prompt_embeds": "AAAA", "truncate_prompt_tokens": 2},
    ],
)
def test_completion_request_validation(serve_env: ServeTestEnv, body: Dict[str, Any]) -> None:
    """Missing prompts and options that need prompt text are rejected."""
    server, client = _client(serve_env)
    with client:
        response = _completion(client, **body)
    assert 400 <= response.status_code < 500
    assert server.generator.generate_calls == []


def test_chat_rejected_when_disabled(serve_env: ServeTestEnv) -> None:
    """Chat parts need --enable_prompt_embeds too."""
    server, client = _client(serve_env, enabled=False)
    with client:
        response = _chat(client, [{"role": "user", "content": [_part(torch.zeros(2, HIDDEN))]}])
    assert response.status_code == 400
    assert DISABLED_MESSAGE in response.json()["message"]
    assert server.generator.generate_calls == []


def test_chat_part_placement(serve_env: ServeTestEnv) -> None:
    """Each part becomes a run of masked placeholder positions where it renders."""
    first, second = torch.randn(2, HIDDEN), torch.randn(3, HIDDEN)
    server, client = _client(serve_env)
    messages = [
        {"role": "system", "content": [_part(first)]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                _part(second),
                {"type": "text", "text": "ok"},
            ],
        },
    ]
    with client:
        response = _chat(client, messages)
    assert response.status_code == 200, response.text
    (call,) = server.generator.generate_calls
    inputs = call.inputs
    assert set(inputs) == {"prompt_token_ids", "prompt_embeds", "prompt_embeds_mask"}

    def enc(text: str) -> List[int]:
        return server.tokenizer.encode(text, add_special_tokens=False)

    # The fake template renders <|role|>content per message; text parts join with "\n".
    expected_ids = (
        enc("<|system|>") + [0] * 2 + enc("<|user|>hi\n") + [0] * 3 + enc("\nok<|assistant|>")
    )
    expected_mask = [False] * len(enc("<|system|>")) + [True] * 2
    expected_mask += [False] * len(enc("<|user|>hi\n")) + [True] * 3
    expected_mask += [False] * len(enc("\nok<|assistant|>"))
    assert inputs["prompt_token_ids"] == expected_ids
    assert inputs["prompt_embeds_mask"] == expected_mask
    assert torch.equal(inputs["prompt_embeds"], torch.cat([first, second]))
    assert server.generator.preprocess_calls == []


@pytest.mark.parametrize(
    ("extra", "content_extra", "match"),
    [
        ({"prompt_token_ids": [1, 2]}, [], "prompt_token_ids"),
        ({"truncate_prompt_tokens": 4}, [], "truncate_prompt_tokens"),
    ],
)
def test_chat_unsupported_combinations(
    serve_env: ServeTestEnv, extra: Dict[str, Any], content_extra: List[Any], match: str
) -> None:
    """Combinations whose positions the server cannot map are rejected."""
    server, client = _client(serve_env)
    content = [_part(torch.zeros(2, HIDDEN))] + content_extra
    with client:
        response = _chat(client, [{"role": "user", "content": content}], **extra)
    assert response.status_code == 400
    assert match in response.json()["message"]
    assert server.generator.generate_calls == []


def test_chat_part_without_data(serve_env: ServeTestEnv) -> None:
    """A part must carry prompt_embeds.data."""
    server, client = _client(serve_env)
    with client:
        response = _chat(
            client, [{"role": "user", "content": [{"type": "prompt_embeds", "prompt_embeds": {}}]}]
        )
    assert response.status_code == 400
    assert server.generator.generate_calls == []


def test_render_chat_refuses_parts(serve_env: ServeTestEnv) -> None:
    """Callers outside the chat route cannot render prompt_embeds parts."""
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    server, client = _client(serve_env)
    with client:
        request = ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": [_part(torch.zeros(2, HIDDEN))]}],
        )
        with pytest.raises(UnsupportedRenderError, match="prompt_embeds"):
            render_chat(request, RenderResources.from_server(server))


def _identity_tokenize(text: str, add_special_tokens: bool) -> List[int]:
    """One id per character; 99 stands for a BOS token."""
    return ([99] if add_special_tokens else []) + [ord(c) for c in text]


def test_splice_special_tokens_only_on_first_segment() -> None:
    """BOS is added once, before the first segment, even when it is empty."""
    part = ChatPromptEmbedsPart(marker="<M>", data="")
    out = splice_chat_prompt_embeds(
        "<M>ab", [part], [torch.zeros(2, 4)], _identity_tokenize, add_special_tokens=True
    )
    assert out["prompt_token_ids"] == [99, 0, 0, ord("a"), ord("b")]
    assert out["prompt_embeds_mask"] == [False, True, True, False, False]


@pytest.mark.parametrize(
    ("text", "match"),
    [("no marker here", "dropped"), ("<M>x<M>", "repeated")],
)
def test_splice_marker_must_render_once(text: str, match: str) -> None:
    """A template that drops or repeats a part's message is an error."""
    part = ChatPromptEmbedsPart(marker="<M>", data="")
    with pytest.raises(ValueError, match=match):
        splice_chat_prompt_embeds(text, [part], [torch.zeros(1, 4)], _identity_tokenize, False)


def test_splice_rejects_mixed_dtypes() -> None:
    """All parts of one request share a dtype."""
    parts = [ChatPromptEmbedsPart("<A>", ""), ChatPromptEmbedsPart("<B>", "")]
    rows = [torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.float16)]
    with pytest.raises(ValueError, match="dtype"):
        splice_chat_prompt_embeds("<A><B>", parts, rows, _identity_tokenize, False)
