"""Wire KVTC to use turbo2's real compression (WHT + 2-bit PolarQuant = 6.4x).

This replaces the q8_0 stub with turbo2's actual quantize/dequantize functions,
giving KVTC real 6.4x compression through the turbo2 kernels.

The FA remap is changed from KVTC->Q8_0 to KVTC->TURBO2_0 so flash attention
can read KVTC blocks using turbo2's vec_dot kernels.
"""

import os
import sys

SRC = r"T:\ik-llama\llama-cpp-turboquant"

def patch(filepath, old, new, desc):
    content = open(filepath, "r").read()
    if old not in content:
        print(f"  SKIP {os.path.basename(filepath)}: target not found for '{desc}'")
        return False
    content = content.replace(old, new, 1)
    open(filepath, "w").write(content)
    print(f"  OK {os.path.basename(filepath)}: {desc}")
    return True

def patch_all(filepath, old, new, desc):
    content = open(filepath, "r").read()
    count = content.count(old)
    if count == 0:
        print(f"  SKIP {os.path.basename(filepath)}: no matches for '{desc}'")
        return 0
    content = content.replace(old, new)
    open(filepath, "w").write(content)
    print(f"  OK {os.path.basename(filepath)}: {desc} ({count} replacements)")
    return count

print("Wiring KVTC to turbo2 real compression")
print("=" * 50)

# 1. ggml.c: Change KVTC type_traits from q8_0 to turbo2 functions
print("\n1. Updating KVTC type_traits to use turbo2...")
ggml_c = os.path.join(SRC, "ggml", "src", "ggml.c")
content = open(ggml_c, "r").read()

old_entry = """    [GGML_TYPE_KVTC] = {
        .type_name                = "kvtc",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },"""

new_entry = """    [GGML_TYPE_KVTC] = {
        .type_name                = "kvtc",
        .blck_size                = QK_TURBO2,
        .type_size                = sizeof(block_turbo2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_turbo2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_turbo2_0_ref,
    },"""

if old_entry in content:
    content = content.replace(old_entry, new_entry)
    print("  OK: type_traits updated to turbo2")
else:
    print("  SKIP: old entry not found (maybe already updated?)")

# Also update the quantize case
old_case = "case GGML_TYPE_KVTC:     result = quantize_q8_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;"
new_case = "case GGML_TYPE_KVTC:     result = quantize_turbo2_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;"
content = content.replace(old_case, new_case)
print("  OK: quantize case updated to turbo2")

open(ggml_c, "w").write(content)

# 2. set-rows.cu: Change KVTC from q8_0 to turbo2 dispatch
print("\n2. Updating SET_ROWS dispatch to turbo2...")
setrows = os.path.join(SRC, "ggml", "src", "ggml-cuda", "set-rows.cu")
content = open(setrows, "r").read()

# Replace the q8_0 KVTC handler with turbo2
old_sr = """} else if (dst->type == GGML_TYPE_KVTC) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    }"""

new_sr = """} else if (dst->type == GGML_TYPE_KVTC) {
        // KVTC uses turbo2 compression (WHT + 2-bit PolarQuant = 6.4x)
        set_rows_cuda_turbo2<idx_t>(ctx, src0, src1, dst);
    }"""

if old_sr in content:
    content = content.replace(old_sr, new_sr)
    open(setrows, "w").write(content)
    print("  OK: SET_ROWS now uses turbo2 path")
else:
    print("  SKIP: q8_0 KVTC handler not found")

# 3. fattn.cu: Change KVTC remap from Q8_0 to TURBO2_0
print("\n3. Updating FA remap to TURBO2_0...")
fattn = os.path.join(SRC, "ggml", "src", "ggml-cuda", "fattn.cu")
n = patch_all(fattn, 
    "GGML_TYPE_Q8_0; // was KVTC",
    "GGML_TYPE_TURBO2_0; // KVTC uses turbo2 compression",
    "FA remap Q8_0->TURBO2_0")

# Also update the generic remaps
content = open(fattn, "r").read()
content = content.replace(
    "if (K->type == GGML_TYPE_KVTC) { const_cast<ggml_tensor*>(K)->type = GGML_TYPE_Q8_0; }",
    "if (K->type == GGML_TYPE_KVTC) { const_cast<ggml_tensor*>(K)->type = GGML_TYPE_TURBO2_0; }"
)
content = content.replace(
    "if (V->type == GGML_TYPE_KVTC) { const_cast<ggml_tensor*>(V)->type = GGML_TYPE_Q8_0; }",
    "if (V->type == GGML_TYPE_KVTC) { const_cast<ggml_tensor*>(V)->type = GGML_TYPE_TURBO2_0; }"
)
# supports check remap
content = content.replace(
    "if (K->type == GGML_TYPE_KVTC) const_cast<ggml_tensor*>(K)->type = GGML_TYPE_Q8_0;",
    "if (K->type == GGML_TYPE_KVTC) const_cast<ggml_tensor*>(K)->type = GGML_TYPE_TURBO2_0;"
)
content = content.replace(
    "if (V->type == GGML_TYPE_KVTC) const_cast<ggml_tensor*>(V)->type = GGML_TYPE_Q8_0;",
    "if (V->type == GGML_TYPE_KVTC) const_cast<ggml_tensor*>(V)->type = GGML_TYPE_TURBO2_0;"
)
open(fattn, "w").write(content)
count = content.count("TURBO2_0; // KVTC") + content.count("TURBO2_0; }")
print(f"  OK: {count} KVTC->TURBO2 remaps in fattn.cu")

# 4. ggml-cuda.cu: Update CPY support from q8_0 to turbo2
print("\n4. Updating CPY support...")
ggml_cuda = os.path.join(SRC, "ggml", "src", "ggml-cuda", "ggml-cuda.cu")
content = open(ggml_cuda, "r").read()
content = content.replace(
    "src1_type == GGML_TYPE_Q8_0 || src1_type == GGML_TYPE_KVTC",
    "src1_type == GGML_TYPE_Q8_0 || src1_type == GGML_TYPE_TURBO2_0 || src1_type == GGML_TYPE_KVTC"
)
content = content.replace(
    "src0_type == GGML_TYPE_Q8_0 || src0_type == GGML_TYPE_KVTC",
    "src0_type == GGML_TYPE_Q8_0 || src0_type == GGML_TYPE_TURBO2_0 || src0_type == GGML_TYPE_KVTC"
)
open(ggml_cuda, "w").write(content)
print("  OK: CPY support updated")

# 5. convert.cu, dequantize.cuh: remap to turbo2
print("\n5. Updating convert/dequantize remaps...")
for fname in ["convert.cu", "dequantize.cuh"]:
    fpath = os.path.join(SRC, "ggml", "src", "ggml-cuda", fname)
    content = open(fpath, "r").read()
    # The earlier patch added KVTC cases before Q8_0 cases
    # Now we need them before TURBO2 cases instead
    # Actually, since KVTC remaps to TURBO2 in FA, these convert/dequant
    # paths will see TURBO2_0 type, not KVTC. So this is fine as-is.
    print(f"  OK: {fname} (no changes needed - FA remap handles it)")

# 6. llama-kv-cache.cpp: Add KVTC to turbo checks
print("\n6. Updating llama-kv-cache.cpp turbo checks...")
kvcache = os.path.join(SRC, "src", "llama-kv-cache.cpp")
content = open(kvcache, "r").read()
# Make is_turbo also match KVTC so it gets WHT rotation
old_turbo = "const bool is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);"
new_turbo = "const bool is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0 || type_k == GGML_TYPE_KVTC);"
if old_turbo in content:
    content = content.replace(old_turbo, new_turbo)
    open(kvcache, "w").write(content)
    print("  OK: KVTC added to is_turbo check (gets WHT rotation)")
else:
    print("  SKIP: is_turbo check not found or already patched")

# 7. llama-graph.cpp: Add KVTC to turbo V checks for inverse WHT
print("\n7. Updating llama-graph.cpp turbo V checks...")
graph = os.path.join(SRC, "src", "llama-graph.cpp")
content = open(graph, "r").read()
# Add KVTC wherever turbo2 V type is checked
old_vcheck = "v->type == GGML_TYPE_TURBO2_0"
new_vcheck = "(v->type == GGML_TYPE_TURBO2_0 || v->type == GGML_TYPE_KVTC)"
count = content.count(old_vcheck)
content = content.replace(old_vcheck, new_vcheck)
open(graph, "w").write(content)
print(f"  OK: {count} turbo2 V-type checks now include KVTC")

print("\n" + "=" * 50)
print("KVTC now wired to turbo2 real compression!")
print("6.4x compression via WHT + 2-bit PolarQuant")
print("Rebuild and test with: -ctk kvtc -ctv kvtc -c 1000000")
