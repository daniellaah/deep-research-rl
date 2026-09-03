"""Revision-pinned Qwen inference adapter with exact generated-token log probabilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from deep_research_rl.agent.contracts import FinishReason, PolicyOutput
from deep_research_rl.agent.prompting import PROMPT_FORMAT_VERSION, build_policy_messages
from deep_research_rl.core.models import AgentState

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


class QwenDependencyError(RuntimeError):
    """Raised when the optional local model runtime is unavailable."""


class QwenInferenceError(RuntimeError):
    """Raised when model input or output violates the rollout contract."""


@dataclass(frozen=True, slots=True)
class QwenGenerationSettings:
    """Bounded generation settings recorded by the rollout command."""

    max_prompt_tokens: int = 8192
    max_new_tokens: int = 96
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_prompt_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("prompt and response token bounds must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must not be negative")
        if self.seed < 0:
            raise ValueError("seed must not be negative")


class QwenPolicyAdapter:
    """Generate strict baseline actions through a local Transformers Qwen checkpoint."""

    prompt_format = PROMPT_FORMAT_VERSION

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        model_name: str,
        model_revision: str,
        device: str,
        settings: QwenGenerationSettings,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.settings = settings
        self._generation_index = 0

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str = DEFAULT_QWEN_MODEL,
        model_revision: str = DEFAULT_QWEN_REVISION,
        device: str = "auto",
        dtype: str = "auto",
        settings: QwenGenerationSettings | None = None,
        local_files_only: bool = False,
    ) -> QwenPolicyAdapter:
        """Load model and tokenizer from exactly one immutable Hugging Face revision."""

        if local_files_only:
            # Some Transformers/Hub version pairs still perform repository metadata probes while
            # resolving sharded checkpoints. Force the Hub's process-wide offline mode before its
            # first import so this option is a hard no-network guarantee.
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on optional installation
            raise QwenDependencyError(
                'Qwen rollout requires the optional dependencies: pip install -e ".[rollout]"'
            ) from error

        resolved_device = _resolve_device(torch, device)
        resolved_dtype = _resolve_dtype(torch, dtype, resolved_device)
        tokenizer_factory: Any = AutoTokenizer
        model_factory: Any = AutoModelForCausalLM
        tokenizer = tokenizer_factory.from_pretrained(
            model_name,
            revision=model_revision,
            local_files_only=local_files_only,
        )
        model = model_factory.from_pretrained(
            model_name,
            revision=model_revision,
            dtype=resolved_dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        loaded_revision = getattr(model.config, "_commit_hash", None)
        if loaded_revision is not None and loaded_revision != model_revision:
            raise QwenInferenceError(
                f"loaded model revision {loaded_revision} differs from requested {model_revision}"
            )
        model.to(resolved_device)
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            model_name=model_name,
            model_revision=model_revision,
            device=resolved_device,
            settings=settings or QwenGenerationSettings(),
        )

    def generate(self, state: AgentState, *, max_searches: int) -> PolicyOutput:
        """Generate one response and retain every generated token's sampling log probability."""

        messages = list(build_policy_messages(state, max_searches=max_searches))
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        prompt_length = int(input_ids.shape[-1])
        if prompt_length > self.settings.max_prompt_tokens:
            raise QwenInferenceError(
                f"prompt has {prompt_length} tokens, exceeding bound "
                f"{self.settings.max_prompt_tokens}; append-only context was not truncated"
            )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        seed = self.settings.seed + self._generation_index
        self._generation_index += 1
        self._torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            self._torch.cuda.manual_seed_all(seed)

        generation_kwargs: dict[str, object] = {
            "attention_mask": attention_mask,
            "do_sample": self.settings.do_sample,
            "max_new_tokens": self.settings.max_new_tokens,
            "output_scores": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "return_dict_in_generate": True,
        }
        if self.settings.do_sample:
            generation_kwargs.update(
                temperature=self.settings.temperature,
                top_k=self.settings.top_k,
                top_p=self.settings.top_p,
            )

        with self._torch.inference_mode():
            generated = self._model.generate(input_ids=input_ids, **generation_kwargs)

        scores = tuple(generated.scores)
        sequence = generated.sequences[0]
        response_tensor = sequence[prompt_length : prompt_length + len(scores)]
        if int(response_tensor.shape[-1]) != len(scores):
            raise QwenInferenceError("generated token IDs and score tensors have different lengths")
        response_ids = tuple(int(value) for value in response_tensor.detach().cpu().tolist())
        response_logprobs = tuple(
            float(score[0].float().log_softmax(dim=-1)[token_id].detach().cpu().item())
            for score, token_id in zip(scores, response_ids, strict=True)
        )
        prompt_ids = tuple(int(value) for value in input_ids[0].detach().cpu().tolist())
        raw_response = self._tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        finish_reason = _finish_reason(
            response_ids,
            eos_token_id=self._model.generation_config.eos_token_id,
            max_new_tokens=self.settings.max_new_tokens,
        )
        return PolicyOutput.from_generation(
            raw_response=raw_response,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            finish_reason=finish_reason,
        )


def _resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise QwenInferenceError("CUDA was requested but is not available")
        if requested == "mps" and not torch_module.backends.mps.is_available():
            raise QwenInferenceError("MPS was requested but is not available")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch_module: Any, requested: str, device: str) -> Any:
    dtypes = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
    }
    if requested != "auto":
        if requested not in dtypes:
            raise ValueError("dtype must be one of: auto, float16, float32, bfloat16")
        return dtypes[requested]
    if device == "mps":
        return torch_module.float16
    if device == "cuda":
        return (
            torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16
        )
    return torch_module.float32


def _finish_reason(
    response_ids: tuple[int, ...],
    *,
    eos_token_id: int | list[int] | None,
    max_new_tokens: int,
) -> FinishReason:
    if eos_token_id is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = set(eos_token_id)
    if response_ids and response_ids[-1] in eos_ids:
        return "eos"
    if len(response_ids) >= max_new_tokens:
        return "length"
    return "stop"
