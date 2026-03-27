"""Calibration utilities for collecting KV samples from a vLLM model."""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from .common import CalibrationData
from .pca import PCACalibrator
from .vllm_backend import extract_request_spans, resolve_attention_layers


DEFAULT_WARMUP_PROMPTS = (
    "Explain how KV cache compression improves LLM serving efficiency.",
    "Describe the transformer attention mechanism in mathematical detail.",
    "Summarize the role of PCA in feature decorrelation and compression.",
    "Write a short overview of GPU memory bottlenecks in autoregressive decoding.",
    "Compare grouped-query attention with standard multi-head attention.",
)


def _parse_layer_idx(prefix: str, fallback: int) -> int:
    matches = re.findall(r"\d+", prefix)
    if not matches:
        return fallback
    return int(matches[-1])


@dataclass
class CalibrationPatchedLayer:
    """Original forward method for one patched attention layer."""

    impl: Any
    original_forward: Any


class CalibrationForwardPatch:
    """Capture KV tensors from a vLLM attention forward call."""

    def __init__(self, collector: "VLLMCalibrationCollector", layer_idx: int, original: Any) -> None:
        self.collector = collector
        self.layer_idx = layer_idx
        self.original = original

    def __call__(
        self,
        impl: Any,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if key.dim() == 3 and value.dim() == 3:
            spans = extract_request_spans(attn_metadata, int(key.shape[0]), device=key.device)
            self.collector.capture(self.layer_idx, key, value, spans)
        return self.original(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)


@dataclass
class VLLMCalibrationCollector:
    """Collect KV samples from vLLM warmup traffic and compute PCA bases."""

    head_group_size: int = 1
    rope_theta: float = 10000.0
    samples_collected: int = 0
    _pca: PCACalibrator = field(init=False)
    _patches: list[CalibrationPatchedLayer] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._pca = PCACalibrator(head_group_size=self.head_group_size, rope_theta=self.rope_theta)

    def capture(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, spans: Sequence[Any]) -> None:
        """Capture one attention step from a patched vLLM forward."""

        for span in spans:
            key_tokens = key[span.start : span.end].detach().to("cpu", dtype=torch.float32)
            value_tokens = value[span.start : span.end].detach().to("cpu", dtype=torch.float32)
            positions = span.positions.detach().to("cpu", dtype=torch.long)
            self._pca.collect(layer_idx, "keys", key_tokens, positions)
            self._pca.collect(layer_idx, "values", value_tokens)

    def install(self, model: Any) -> None:
        """Patch all vLLM attention layers on the provided model/runner."""

        if self._patches:
            return
        for fallback_idx, (prefix, layer) in enumerate(resolve_attention_layers(model)):
            layer_idx = _parse_layer_idx(prefix, fallback_idx)
            impl = layer.impl
            original_forward = impl.forward
            impl.forward = types.MethodType(
                CalibrationForwardPatch(self, layer_idx, original_forward),
                impl,
            )
            self._patches.append(CalibrationPatchedLayer(impl=impl, original_forward=original_forward))

    def uninstall(self) -> None:
        """Restore original vLLM attention forwards."""

        for patch in self._patches:
            patch.impl.forward = patch.original_forward
        self._patches.clear()

    def run_warmup(
        self,
        llm: Any,
        prompts: Sequence[str],
        *,
        sampling_params: Any | None = None,
        max_prompts: int | None = None,
    ) -> None:
        """Run warmup prompts through a vLLM LLM instance while patched."""

        warmup_prompts = list(prompts[:max_prompts]) if max_prompts is not None else list(prompts)
        if not warmup_prompts:
            return
        self.install(llm)
        try:
            params = sampling_params
            if params is None:
                try:
                    from vllm import SamplingParams
                except ImportError as exc:  # pragma: no cover - vLLM is optional in unit tests.
                    raise ImportError("vLLM is required to run calibration warmup.") from exc
                params = SamplingParams(max_tokens=1, temperature=0.0)
            llm.generate(warmup_prompts, params)
            self.samples_collected += len(warmup_prompts)
        finally:
            self.uninstall()

    def compute_calibration(self, bit_budget_ratio: float = 0.25) -> CalibrationData:
        """Compute the calibration artifact consumed by the KVTC vLLM backend."""

        return self._pca.compute(bit_budget_ratio=bit_budget_ratio)

    @staticmethod
    def save(path: str | Path, calibration_data: CalibrationData) -> None:
        """Persist a calibration artifact to disk."""

        torch.save(calibration_data, Path(path))

    @staticmethod
    def load(path: str | Path) -> CalibrationData:
        """Load a calibration artifact from disk."""

        return torch.load(Path(path), weights_only=False)


def calibrate_vllm_model(
    llm: Any,
    prompts: Sequence[str] = DEFAULT_WARMUP_PROMPTS,
    *,
    bit_budget_ratio: float = 0.25,
    head_group_size: int = 1,
    rope_theta: float = 10000.0,
    sampling_params: Any | None = None,
    max_prompts: int | None = None,
    save_path: str | Path | None = None,
) -> CalibrationData:
    """Warm a vLLM model, collect KV samples, and compute calibration data."""

    collector = VLLMCalibrationCollector(head_group_size=head_group_size, rope_theta=rope_theta)
    collector.run_warmup(llm, prompts, sampling_params=sampling_params, max_prompts=max_prompts)
    calibration = collector.compute_calibration(bit_budget_ratio=bit_budget_ratio)
    if save_path is not None:
        collector.save(save_path, calibration)
    return calibration


__all__ = [
    "DEFAULT_WARMUP_PROMPTS",
    "VLLMCalibrationCollector",
    "calibrate_vllm_model",
]
