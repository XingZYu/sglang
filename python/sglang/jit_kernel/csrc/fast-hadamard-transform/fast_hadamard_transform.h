/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Copied from https://github.com/sgl-project/fast-hadamard-transform

#pragma once

#include <cstdint>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
  using index_t = int64_t;

  int batch, dim, log_N;

  index_t x_batch_stride;
  index_t out_batch_stride;

  float scale;

  // Common data pointers.
  void* __restrict__ x_ptr;
  void* __restrict__ out_ptr;

  // Optional per-element sign vectors for fused WHT rotation.
  // When non-null, load multiplies by signs1 and store multiplies by signs2.
  // Shape: (dim,) float32. Set to nullptr to skip.
  const float* __restrict__ signs1_ptr;
  const float* __restrict__ signs2_ptr;

  // Optional output buffer for L2 norms (fused norm+normalize+WHT).
  // Shape: (batch,) float32. Set to nullptr to skip norm computation.
  float* __restrict__ out_norms_ptr;

  // Dual-input mode (for K/V without torch.cat):
  // When x2_ptr != nullptr, blocks with batch_id >= split_batch read from x2_ptr.
  // Blocks with batch_id < split_batch still read from x_ptr.
  void* __restrict__ x2_ptr;
  index_t x2_batch_stride;
  int split_batch;  // Number of batches sourced from x_ptr (rest from x2_ptr)
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Params for the fully-fused norm+WHT+searchsorted+pack+scatter_store kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

struct FusedNormWhtPackStoreParams {
  using index_t = int64_t;

  // Grid: (2 * tokens * heads, 1, 1) — each block handles one (token, head, K_or_V)
  int batch;           // = 2 * tokens * heads
  int dim;             // head dimension (power of 2)
  int log_N;           // log2(dim)

  // Dual-input pointers (K and V)
  void* __restrict__ x_ptr;       // K: (tokens*heads, dim) bf16/fp16
  void* __restrict__ x2_ptr;      // V: (tokens*heads, dim) bf16/fp16
  index_t x_batch_stride;         // stride between consecutive (token,head) rows for K
  index_t x2_batch_stride;        // stride between consecutive (token,head) rows for V
  int split_batch;                 // = tokens * heads (K occupies first split_batch blocks)

  // WHT parameters
  const float* __restrict__ signs1_ptr;  // (dim,) float32
  const float* __restrict__ signs2_ptr;  // (dim,) float32
  float scale;                            // 1/sqrt(dim)

  // Quantization parameters
  const float* __restrict__ boundaries_ptr;  // (n_boundaries,) float32
  const float* __restrict__ centroids_ptr;   // (n_centroids,) float32
  int n_boundaries;                           // 15 for 4-bit, 3 for 2-bit

  // Scatter output: packed KV buffers
  uint8_t* __restrict__ k_buffer;    // (pool_size, heads, packed_dim) uint8
  uint8_t* __restrict__ v_buffer;    // (pool_size, heads, packed_dim) uint8

  // Scatter output: dscale buffers (bf16)
  void* __restrict__ k_dscale;       // (pool_size, heads) bf16
  void* __restrict__ v_dscale;       // (pool_size, heads) bf16

  // Scatter indices
  const int64_t* __restrict__ loc;   // (tokens,) int64 — pool slot per token

  // Buffer strides
  index_t kb_stride_s;  // k_buffer.stride(0) — pool_size dim stride
  index_t kb_stride_h;  // k_buffer.stride(1) — heads dim stride
  index_t vb_stride_s;  // v_buffer.stride(0)
  index_t vb_stride_h;  // v_buffer.stride(1)
  index_t kds_stride;   // k_dscale.stride(0) — pool_size dim stride
  index_t vds_stride;   // v_dscale.stride(0)

  // Derived dimensions
  int tokens;
  int heads;
  int packed_dim;   // dim / pack_ratio (64 for 4-bit dim=128, 32 for 2-bit dim=128)
};
