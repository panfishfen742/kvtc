"""Optional real-model evaluation for KVTC."""

from __future__ import annotations

import os

import pytest
import torch

from src.calibrate import KVTCCalibrator
from src.pipeline import KVTCCompressor


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TEST") != "1",
    reason="Set RUN_REAL_MODEL_TEST=1 to run the TinyLlama integration test.",
)


def test_tinyllama_compression_metrics() -> None:
    transformers = pytest.importorskip("transformers")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    texts = [
        "KV cache compression can unlock longer contexts.",
        "Dynamic programming assigns bits where variance matters.",
        "Principal components make entropy coding more effective.",
        "Attention sinks should be preserved exactly.",
        "Sliding windows remain uncompressed in KVTC.",
        "TinyLlama offers a small real-model integration target.",
        "Synthetic tests alone are not enough for honest evaluation.",
        "Entropy coding should exploit low-entropy quantized streams.",
        "RoPE must be undone before calibrating key tensors.",
        "This is an independent implementation of the paper.",
    ]
    calibrator = KVTCCalibrator(head_group_size=1)
    calibrator.collect_samples(model, tokenizer, texts, max_samples=10)
    calibration = calibrator.compute_calibration()
    compressor = KVTCCompressor(calibration)
    encoded = tokenizer(texts[0], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded, use_cache=True)
    keys, values = [], []
    past = outputs.past_key_values
    if hasattr(past, 'layers'):
        for layer in past.layers:
            keys.append(layer.keys[0].transpose(0, 1))
            values.append(layer.values[0].transpose(0, 1))
    elif hasattr(past, 'key_cache'):
        for i in range(len(past.key_cache)):
            keys.append(past.key_cache[i][0].transpose(0, 1))
            values.append(past.value_cache[i][0].transpose(0, 1))
    else:
        for layer_keys, layer_values in past:
            keys.append(layer_keys[0].transpose(0, 1))
            values.append(layer_values[0].transpose(0, 1))
    kv_cache = {"keys": torch.stack(keys), "values": torch.stack(values)}
    positions = torch.arange(encoded["input_ids"].shape[1], dtype=torch.long)
    compressed = compressor.compress(kv_cache, positions)
    restored = compressor.decompress(compressed)
    cosine = torch.nn.functional.cosine_similarity(
        kv_cache["keys"].reshape(1, -1),
        restored["keys"].reshape(1, -1),
    ).item()
    mse = torch.mean((kv_cache["keys"] - restored["keys"]) ** 2).item()
    print(f"cosine_similarity={cosine:.4f}, compression_ratio={compressed.metadata.compression_ratio:.2f}, mse={mse:.6f}")
    assert cosine > 0.9
