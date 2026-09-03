from __future__ import annotations

from typing import Any

import pytest

from deep_research_rl.agent.qwen import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_REVISION,
    QwenGenerationSettings,
    QwenInferenceError,
    _finish_reason,
    _resolve_device,
    _resolve_dtype,
)


class _Unavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class _CudaUnavailable(_Unavailable):
    @staticmethod
    def is_bf16_supported() -> bool:
        return False


class _Backends:
    mps = _Unavailable()


class FakeTorch:
    cuda = _CudaUnavailable()
    backends = _Backends()
    float16 = "float16"
    float32 = "float32"
    bfloat16 = "bfloat16"


def test_approved_qwen_checkpoint_is_an_immutable_default() -> None:
    assert DEFAULT_QWEN_MODEL == "Qwen/Qwen3-4B-Instruct-2507"
    assert DEFAULT_QWEN_REVISION == "cdbee75f17c01a7cc42f958dc650907174af0554"


def test_generation_settings_reject_unbounded_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="bounds must be positive"):
        QwenGenerationSettings(max_new_tokens=0)
    with pytest.raises(ValueError, match="top_p"):
        QwenGenerationSettings(top_p=0.0)


def test_device_and_dtype_resolution_are_explicit() -> None:
    torch_module: Any = FakeTorch()
    assert _resolve_device(torch_module, "auto") == "cpu"
    assert _resolve_dtype(torch_module, "auto", "cpu") == "float32"
    assert _resolve_dtype(torch_module, "float16", "cpu") == "float16"
    with pytest.raises(QwenInferenceError, match="CUDA"):
        _resolve_device(torch_module, "cuda")


def test_finish_reason_prefers_eos_then_length() -> None:
    assert _finish_reason((1, 2), eos_token_id=2, max_new_tokens=2) == "eos"
    assert _finish_reason((1, 3), eos_token_id=2, max_new_tokens=2) == "length"
    assert _finish_reason((1,), eos_token_id=[2, 4], max_new_tokens=2) == "stop"
