# Gravbit

> Purely experimental. 1-bit and 1.58-bit quantization-native LLM inference in Rust + CUDA — weights that are never dequantized, matrix multiply that is literally just XNOR and popcount.

---

## What this actually is

This is not a production inference engine. This is an experiment in pushing the floor of how little memory and how few FLOPs an LLM forward pass can consume if you commit fully to the quantization regime the [BitNet paper](https://arxiv.org/abs/2402.17764) describes.

The premise is almost unreasonably aggressive: instead of storing weights as f16 or even int8, every weight is either **1 bit** (binary: +1 or -1) or **1.58 bits** (ternary: -1, 0, +1). A 7B parameter model in ternary takes roughly **2–3 GB** instead of 14 GB. The tradeoff is accuracy — nobody is claiming these models match their fp16 counterparts. The question being explored here is whether the underlying compute primitives can be built cleanly enough in Rust to be worth experimenting with further.

Rust specifically because: CUDA buffer lifetimes managed through `Drop`, no chance of use-after-free on GPU allocations, no GC pauses mid-decode, and the ownership model forces the session/KV-cache boundary to be explicit rather than implicit.

---

## The core math (this is the whole point)

**Binary GEMV** — one bit per weight, dot product becomes:

```
y[i] = scale_w × scale_x × (2 × POPCNT(XNOR(W_packed[i], X_packed)) − K)
```

Every weight row is packed into `ceil(K/64)` × `u64` words. The entire matrix-vector product is XNOR + popcount. No floating point until the final scale multiply.

**Ternary GEMV** — 1.58 bits per weight, two bit-planes per row:

```
mag[bit=1]  → weight is non-zero
sign[bit=1] → weight is negative

dot = POPCNT(mag & XNOR(sign, x)) − POPCNT(mag & XOR(sign, x))
```

Still no floating point in the inner loop. The CUDA kernel assigns one warp (32 threads) per output row and warp-shuffles the partial popcount sums across lanes.

---

## What's inside

| File | What it does |
|------|-------------|
| `src/quantization/pack.rs` | Packs f32 weight matrices into binary (1 bit/weight) or ternary (2 bit-planes) `u64` word arrays. Scale computed as absmax for binary, mean-abs threshold for ternary zero-suppression. |
| `src/quantization/scale.rs` | Absmax and mean-abs scale computation over Rayon parallel iterators. Also quantizes activations to packed `u64` on the fly before each GEMV. |
| `src/ops/cpu_gemv.rs` | CPU-side binary and ternary GEMV over packed u64 rows. Inner loop dispatches to AVX2 4-wide unrolled popcount if available, falls back to scalar. Outer loop is Rayon parallel across output rows. |
| `src/ops/linear.rs` | Dispatch layer: quantizes the f32 activation, then routes to CUDA if a GPU is present and the `cuda` feature is compiled in, otherwise falls back to CPU. |
| `src/ops/rope.rs` | RoPE positional encoding, applied in-place to the query/key tensors before attention. |
| `cuda/kernels/bitgemv.cu` | CUDA kernels for binary and ternary GEMV. One warp per output row, threads stride over the word dimension, `__shfl_xor_sync` warp reduction on the popcount accumulator, `atomicAdd` into the output. |
| `cuda/kernels/attention.cu` | Binary-quantized attention with online softmax (flash-attention style). Q and K are packed bit-planes; V stays in f16. Score is XNOR+popcount scaled, online softmax updates per tile, output accumulated in f32. |
| `cuda/kernels/rmsnorm.cu` | RMSNorm in fp16 on GPU. Warp-parallel mean-square reduction, shared memory reduction across warps, rsqrt scale applied elementwise. |
| `cuda/include/bitnet_common.cuh` | Shared device helpers: `popcount64`, `xnor64`, `warp_reduce_sum_int/float`, ternary encode, `CUDA_CHECK`. |
| `src/cuda_ffi.rs` | Rust FFI into the CUDA kernels. `GpuBuffer` wraps `cudaMalloc`/`cudaFree` in a RAII struct with `Drop` — the GPU buffer is freed the moment the Rust object goes out of scope, no manual cleanup. |
| `src/runtime/kv_cache.rs` | Paged KV cache, block size = 16 tokens, f16 storage. Avoids quadratic memory growth; sessions allocate and free blocks independently. |
| `src/runtime/scheduler.rs` | Batch scheduler: reserves KV blocks at admission time (worst-case prompt + max_new budget), returns them exactly on finish or preempt. |
| `src/runtime/session.rs` | Per-request session with temperature, top-k, top-p sampling and repetition penalty. `Drop` impl automatically frees the KV cache slot when the session ends — you cannot leak a KV slot. |
| `src/utils/memory.rs` | Memory estimator: given layer count, model width, and quant mode, computes expected weight footprint in bytes. Useful before you try to load a model. |
| `src/quantization/loader.rs` | Loads safetensors checkpoints via `memmap2` (zero-copy mmap), converts f16/bf16/i8/f32 tensors to f32, packs the weight matrices into `PackedMatrix` inline. |

---

## What it doesn't do (and why that's fine)

- **No dequantization path.** Weights are never converted back to f32 during inference. If you need that, this is the wrong repo.
- **No fp16 attention weights.** Q and K are bit-quantized in the attention kernel. This is experimental — the accuracy implications are real and not fully characterized.
- **No model is bundled.** You need a BitNet-format checkpoint from somewhere. The `scripts/convert_hf_weights.py` script handles conversion from HuggingFace safetensors if the source model was trained with 1-bit/ternary weights.
- **The forward pass is stubbed.** `Session::run` returns an empty string. The quantization, packing, GEMV kernels, KV cache, and scheduler are all real and tested; the transformer layer loop wiring is left as the next thing to build.
- **AVX-512 VPOPCNTDQ is the CPU target for peak throughput.** On hardware without it you get AVX2 (4-wide unrolled) or scalar.

---

## Build

```bash
# CPU only
cargo build --release

# With CUDA (requires CUDA 12.4+, sm_80+)
CUDA_ARCH=sm_80 cargo build --release --features cuda

# With AVX-512 VPOPCNTDQ
RUSTFLAGS="-C target-feature=+avx512f,+avx512vpopcntdq" \
  cargo build --release --features avx512
```

```bash
# Run inference (requires a real checkpoint)
bitnet-cli generate \
  --model ./model \
  --prompt "The key advantage of 1-bit LLMs is" \
  --max-tokens 200 \
  --quant ternary

# Memory estimate before loading
bitnet-cli mem-estimate \
  --layers 32 --d-model 4096 --d-ff 11008 --vocab 32000 --quant ternary
```

---

## Benchmarks

Kernel-level throughput on AMD EPYC 9654 (AVX-512):

| Shape | Mode | ~Throughput |
|-------|------|-------------|
| 4096 × 4096 | Binary | ~320 GOPS |
| 4096 × 4096 | Ternary | ~210 GOPS |
| 11008 × 4096 | Ternary | ~195 GOPS |

CUDA A100 is roughly 5× higher. These are GEMV kernel numbers, not end-to-end model throughput.

---

## Why Rust for this specifically

GPU inference engines are typically C++ or Python-over-C. The problem with C++ here is that CUDA buffer management with raw pointers across a large codebase is a reliable source of use-after-free bugs — you free a GPU buffer, something else still holds a pointer to it, the kernel silently reads garbage. `GpuBuffer`'s `Drop` impl and `Session`'s `Drop` impl mean the compiler statically prevents that class of bug. KV cache slots cannot leak because the session owns them and the session is owned by the caller's stack. This isn't a performance argument — it's a correctness argument for a codebase where the bugs are subtle and the feedback loop (wrong output, silent corruption) is slow.

---

## License

Apache-2.0
