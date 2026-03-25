"""Benchmark KVTC on Mistral-7B-Instruct-v0.3 (4-bit quantized).
Tests multiple bit budget ratios to show compression vs quality tradeoff.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from src.pca import PCACalibrator
from src.pipeline import KVTCCompressor

def extract_kv(past_kv):
    if hasattr(past_kv, 'layers'):
        return [(l.keys, l.values) for l in past_kv.layers]
    if hasattr(past_kv, 'key_cache'):
        return [(past_kv.key_cache[i], past_kv.value_cache[i]) for i in range(len(past_kv.key_cache))]
    return [(item[0], item[1]) for item in past_kv]

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading {model_name} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    # Detect architecture
    with torch.no_grad():
        test = tokenizer("test", return_tensors="pt").to(device)
        kv = extract_kv(model(**test, use_cache=True).past_key_values)
        head_dim = kv[0][0].shape[-1]
        n_kv_heads = kv[0][0].shape[1]
        n_layers = len(kv)
    print(f"Config: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    # Calibration texts (longer to get good PCA bases)
    cal_texts = [
        "The quick brown fox jumps over the lazy dog in a beautiful meadow on a sunny day.",
        "Machine learning models require careful optimization, tuning, and evaluation on diverse datasets.",
        "KV cache compression can unlock longer contexts for LLM inference in production systems.",
        "Dynamic programming finds optimal bit allocations by minimizing reconstruction error.",
        "Principal component analysis decorrelates feature dimensions to expose low-rank structure.",
        "Entropy coding exploits statistical redundancy in quantized data for lossless compression.",
        "Attention mechanisms allow transformer models to focus on relevant tokens in the sequence.",
        "GPU memory management is critical for serving large language models at scale efficiently.",
        "Neural network architectures continue to evolve with innovations in efficiency and capability.",
        "The transformer architecture introduced in 2017 has revolutionized natural language processing.",
    ]

    # Test prompts (must be >132 tokens to have middle region)
    test_prompts = [
        "Explain the concept of attention mechanisms in neural networks. Cover the history starting from Bahdanau attention in 2014, then move to the transformer architecture introduced in 2017. Discuss multi-head attention, self-attention, cross-attention, and how they differ. Explain the key, query, and value projections in detail. Discuss positional encodings and why they are needed. Cover the softmax operation and how it creates attention weights. Explain masked attention for autoregressive generation. Discuss grouped query attention and multi-query attention optimizations. Cover flash attention and memory efficient attention implementations.",
        "Write a comprehensive guide to building a modern web application from scratch. Start with choosing a technology stack including frontend framework, backend language, database, and deployment platform. Discuss React versus Vue versus Angular for the frontend. Compare Node.js, Python Flask, Python FastAPI, Go, and Rust for the backend. Explain PostgreSQL versus MongoDB versus Redis for different use cases. Cover authentication and authorization patterns including OAuth, JWT tokens, and session management. Discuss API design principles including REST versus GraphQL. Cover database schema design, migrations, and ORM usage. Explain containerization with Docker and orchestration with Kubernetes.",
        "Describe the complete history of artificial intelligence from its inception to the present day. Start with Alan Turing and the Turing test in 1950. Discuss the Dartmouth conference in 1956 where AI was formally founded as a field. Cover the early optimism period and the development of programs like ELIZA and SHRDLU. Explain the first AI winter in the 1970s and the reasons behind the funding cuts. Discuss expert systems and their rise in the 1980s. Cover the second AI winter in the late 1980s and early 1990s. Explain the emergence of machine learning approaches including support vector machines, random forests, and boosting methods.",
    ]

    # Test multiple bit budget ratios
    bit_budgets = [
        {"name": "High Quality (25%)", "ratio": 0.25},
        {"name": "Balanced (12%)", "ratio": 0.12},
        {"name": "Aggressive (6%)", "ratio": 0.06},
    ]

    for bb in bit_budgets:
        print(f"\n{'='*65}")
        print(f"  BIT BUDGET: {bb['name']} (ratio={bb['ratio']})")
        print(f"{'='*65}")

        # Calibrate
        calibrator = PCACalibrator(head_group_size=1, rope_theta=10000.0)
        t0 = time.time()
        for text in cal_texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, use_cache=True)
            positions = torch.arange(inputs["input_ids"].shape[1], device="cpu")
            for li, (k, v) in enumerate(extract_kv(out.past_key_values)):
                calibrator.collect(li, "keys", k[0].transpose(0, 1).cpu().float(), positions)
                calibrator.collect(li, "values", v[0].transpose(0, 1).cpu().float())
        calibration = calibrator.compute(bit_budget_ratio=bb["ratio"])
        cal_time = time.time() - t0
        print(f"  Calibration: {cal_time:.1f}s, {len(calibration.entries)} PCA entries")

        compressor = KVTCCompressor(calibration)
        all_key_cos = []
        all_val_cos = []
        all_ratios = []

        for pi, prompt in enumerate(test_prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model(**inputs, use_cache=True)
            kv_layers = extract_kv(out.past_key_values)
            positions = torch.arange(seq_len, device="cpu")

            # Build [layers, seq, heads, dim] tensor
            all_keys = []
            all_values = []
            for k, v in kv_layers:
                all_keys.append(k[0].transpose(0, 1).cpu().float())
                all_values.append(v[0].transpose(0, 1).cpu().float())
            kv_cache = {"keys": torch.stack(all_keys), "values": torch.stack(all_values)}

            # Compress/decompress
            t0 = time.time()
            compressed = compressor.compress(kv_cache, positions)
            compress_ms = (time.time() - t0) * 1000

            t0 = time.time()
            restored = compressor.decompress(compressed)
            decompress_ms = (time.time() - t0) * 1000

            # Metrics
            key_cos = F.cosine_similarity(kv_cache["keys"].reshape(1,-1), restored["keys"].reshape(1,-1)).item()
            val_cos = F.cosine_similarity(kv_cache["values"].reshape(1,-1), restored["values"].reshape(1,-1)).item()
            cr = compressed.metadata.compression_ratio
            mid = compressed.metadata.middle_len
            sink = compressed.metadata.sink_len
            win = compressed.metadata.window_len

            all_key_cos.append(key_cos)
            all_val_cos.append(val_cos)
            all_ratios.append(cr)

            print(f"\n  Prompt {pi+1} ({seq_len} tokens, middle={mid}, sinks={sink}, window={win}):")
            print(f"    Key cosine:   {key_cos:.6f}")
            print(f"    Value cosine: {val_cos:.6f}")
            print(f"    Compression:  {cr:.1f}x")
            print(f"    Compress:     {compress_ms:.0f}ms | Decompress: {decompress_ms:.0f}ms")

        avg_kcos = sum(all_key_cos) / len(all_key_cos)
        avg_vcos = sum(all_val_cos) / len(all_val_cos)
        avg_cr = sum(all_ratios) / len(all_ratios)
        print(f"\n  AVERAGE: key={avg_kcos:.6f}, val={avg_vcos:.6f}, compression={avg_cr:.1f}x")

    print("\nDone!")

if __name__ == "__main__":
    main()
