#include "bitnet_common.cuh"
#include <cuda_fp16.h>

__global__ void pack_activations_fp16(
    const __half* __restrict__ x,
    uint64_t*     __restrict__ x_pack,
    int K)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int word_idx = tid;
    int base     = word_idx * 64;
    if (base >= K) return;

    uint64_t word = 0ULL;
#pragma unroll
    for (int b = 0; b < 64 && (base + b) < K; ++b) {
        uint16_t raw;
        __half v = x[base + b];
        memcpy(&raw, &v, sizeof(raw));
        // sign bit=0 means positive in fp16; set the packed bit for positive values
        // to match the CPU quantise_activation convention (bit=1 ↔ v >= 0).
        uint64_t positive = ((raw >> 15) ^ 1ULL) & 1ULL;
        word |= (positive << b);
    }
    x_pack[word_idx] = word;
}

extern "C"
__global__ void bitgemv_1bit(
    const uint64_t* __restrict__ W_packed,
    const uint64_t* __restrict__ x_packed,
    float*          __restrict__ y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words,
    int             K)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int row     = blockIdx.x * TILE_M + warp_id;

    if (row >= N) return;

    const uint64_t* w_row = W_packed + (size_t)row * K_words;

    int accum = 0;
#pragma unroll 4
    for (int w = lane; w < K_words; w += WARP_SIZE) {
        uint64_t ww = w_row[w];
        uint64_t xw = x_packed[w];
        accum += popcount64(xnor64(ww, xw));
    }

    accum = warp_reduce_sum_int(accum);

    if (lane == 0) {
        // K_words*64 counts the packed word capacity, which is >= the real
        // row width K whenever K isn't a multiple of 64. The unused tail
        // bits in both W_packed and x_packed are always 0 by construction
        // (see pack.rs / quantise_activation), and xnor64(0, 0) == ~0,
        // i.e. every padding bit is popcounted as a spurious "agreement".
        // There are exactly (K_words*64 - K) such phantom agreements per
        // row; subtract them out of accum before applying the
        // 2*popcount - K formula so this matches the true, unpadded dot
        // product for any K, not just multiples of 64.
        int pad_bits         = K_words * 64 - K;
        int accum_corrected  = accum - pad_bits;
        float dot = (float)(2 * accum_corrected - K);
        atomicAdd(&y[row], w_scale * x_scale * dot);
    }
}

extern "C"
__global__ void bitgemv_ternary(
    const uint64_t* __restrict__ W_mag,
    const uint64_t* __restrict__ W_sign,
    const uint64_t* __restrict__ x_packed,
    float*          __restrict__ y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int row     = blockIdx.x * TILE_M + warp_id;

    if (row >= N) return;

    const uint64_t* wm = W_mag  + (size_t)row * K_words;
    const uint64_t* ws = W_sign + (size_t)row * K_words;

    // No padding correction needed here: W_mag's padding bits are always 0
    // by construction, and every term below is gated through "& mag", so
    // padding word positions contribute exactly 0 to `dot` regardless of
    // K vs K_words*64.
    int dot = 0;
#pragma unroll 4
    for (int w = lane; w < K_words; w += WARP_SIZE) {
        uint64_t mag  = wm[w];
        uint64_t sign = ws[w];
        uint64_t xw   = x_packed[w];
        dot += popcount64(mag & xnor64(sign, xw));
        dot -= popcount64(mag & (sign ^ xw));
    }

    dot = warp_reduce_sum_int(dot);

    if (lane == 0)
        atomicAdd(&y[row], w_scale * x_scale * (float)dot);
}

extern "C"
__global__ void add_bias_relu(
    float*       __restrict__ y,
    const float* __restrict__ bias,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = y[i] + bias[i];
        y[i] = v > 0.f ? v : 0.f;
    }
}

extern "C" void launch_bitgemv_1bit(
    const uint64_t* W_packed,
    const uint64_t* x_packed,
    float*          y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words,
    int             K,
    cudaStream_t    stream)
{
    dim3 grid((N + TILE_M - 1) / TILE_M);
    dim3 block(THREADS_PER_CTA);
    bitgemv_1bit<<<grid, block, 0, stream>>>(
        W_packed, x_packed, y, w_scale, x_scale, N, K_words, K);
}

extern "C" void launch_bitgemv_ternary(
    const uint64_t* W_mag,
    const uint64_t* W_sign,
    const uint64_t* x_packed,
    float*          y,
    float           w_scale,
    float           x_scale,
    int             N,
    int             K_words,
    cudaStream_t    stream)
{
    dim3 grid((N + TILE_M - 1) / TILE_M);
    dim3 block(THREADS_PER_CTA);
    bitgemv_ternary<<<grid, block, 0, stream>>>(
        W_mag, W_sign, x_packed, y, w_scale, x_scale, N, K_words);
}
