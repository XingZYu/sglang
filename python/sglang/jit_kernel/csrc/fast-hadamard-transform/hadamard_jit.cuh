/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "fast_hadamard_transform_special.h"
#include "static_switch.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

using ::bf16_t;
using ::fp16_t;
using ::HadamardParamsBase;

constexpr inline int ceil_log2(int val) {
  int log = 0;
  int p = 1;
  while (p < val) {
    p <<= 1;
    ++log;
  }
  return log;
}

template <int kNThreads_, int kLogN_, typename input_t_>
struct FastHadamardKernelTraits {
  using input_t = input_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kLogN = kLogN_;
  static constexpr int N = 1 << kLogN;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  static constexpr int kNChunks = N / (kNElts * kNThreads);
  static constexpr int kSmemExchangeSize = (N * 4) < (32 * 1024) ? (N * 4) : (32 * 1024);
  static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
  static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
  static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNThreads_, int kLogN_, int kMultiple, int kMaxDim, int kMaxSmem, typename input_t_>
struct FastHadamardMNKernelTraits {
  using input_t = input_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kLogN = kLogN_;
  static constexpr int N = (1 << kLogN) * kMultiple;
  static_assert(N <= kMaxDim);
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = 4;
  static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  static constexpr int kNChunks = N / (kNElts * kNThreads);
  static_assert(kNChunks == kMultiple);
  static constexpr int kSmemExchangeSize = (N * 4) < kMaxSmem ? (N * 4) : kMaxSmem;
  static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
  static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
  static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard12NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 12, 12 * 1024, 24 * 1024, input_t_>;

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard20NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 20, 20 * 1024, 40 * 1024, input_t_>;

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard28NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 28, 28 * 1024, 28 * 1024, input_t_>;

template <int kNThreads_, int kLogN_, typename input_t_>
using FastHadamard40NTraits = FastHadamardMNKernelTraits<kNThreads_, kLogN_, 40, 40 * 1024, 40 * 1024, input_t_>;

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_12(float x[kNChunks][12]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_12(x[c]);
  }
}

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_20(float x[kNChunks][20]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_20(x[c]);
  }
}

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_28(float x[kNChunks][28]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_28(x[c]);
  }
}

template <int kNChunks>
SGL_DEVICE void hadamard_mult_thread_chunk_40(float x[kNChunks][40]) {
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    hadamard_mult_thread_40(x[c]);
  }
}

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void fast_hadamard_transform_kernel(HadamardParamsBase params) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(Ktraits::kNElts);
  static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");

  constexpr int kWarpSize = kNThreads < 32 ? kNThreads : 32;
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");

  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

  constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
  constexpr int kNExchanges = kNChunks / kChunksPerExchange;
  static_assert(kNExchanges * kChunksPerExchange == kNChunks);

  extern __shared__ char smem_[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_);

  const int batch_id = static_cast<int>(blockIdx.x);
  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride;

  float x_vals[kNChunks][kNElts];
  load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

  hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
    hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals_transposed[i][c] = x_vals[c][i];
      }
    }

    if constexpr (kNChunks == 12) {
      hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 20) {
      hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 28) {
      hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 40) {
      hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
    } else {
      constexpr int kLogNChunks = cilog2(kNChunks);
      static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
      hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[c][i] = x_vals_transposed[i][c];
      }
    }
  }

  store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale);
}

// Fused WHT rotation kernel: out = signs2 * H(signs1 * x) * scale
template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void fast_hadamard_transform_with_signs_kernel(HadamardParamsBase params) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(Ktraits::kNElts);
  static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");

  constexpr int kWarpSize = kNThreads < 32 ? kNThreads : 32;
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");

  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

  constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
  constexpr int kNExchanges = kNChunks / kChunksPerExchange;
  static_assert(kNExchanges * kChunksPerExchange == kNChunks);

  extern __shared__ char smem_[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_);

  const int batch_id = static_cast<int>(blockIdx.x);
  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride;

  float x_vals[kNChunks][kNElts];
  // Fused: load + multiply by signs1
  load_input_with_signs<kNChunks, kNElts, input_t>(x, x_vals, params.dim, params.signs1_ptr);

  hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
    hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals_transposed[i][c] = x_vals[c][i];
      }
    }

    if constexpr (kNChunks == 12) {
      hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 20) {
      hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 28) {
      hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 40) {
      hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
    } else {
      constexpr int kLogNChunks = cilog2(kNChunks);
      static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
      hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[c][i] = x_vals_transposed[i][c];
      }
    }
  }

  // Fused: multiply by signs2 + scale + store
  store_output_with_signs<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale, params.signs2_ptr);
}

// Fused norm+normalize+WHT kernel: computes L2 norm, normalizes, then WHT with signs
template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void fast_hadamard_transform_with_signs_and_norm_kernel(HadamardParamsBase params) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(Ktraits::kNElts);
  static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");

  constexpr int kWarpSize = kNThreads < 32 ? kNThreads : 32;
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");

  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

  constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
  constexpr int kNExchanges = kNChunks / kChunksPerExchange;
  static_assert(kNExchanges * kChunksPerExchange == kNChunks);

  extern __shared__ char smem_[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_);

  const int batch_id = static_cast<int>(blockIdx.x);
  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride;
  // Output is always float32 for the fused norm+WHT kernel (downstream pack expects float32)
  float* out = reinterpret_cast<float*>(params.out_ptr) + batch_id * params.out_batch_stride;

  float x_vals[kNChunks][kNElts];
  // Fused: load + multiply by signs1
  load_input_with_signs<kNChunks, kNElts, input_t>(x, x_vals, params.dim, params.signs1_ptr);

  // ---- Fused norm + normalize (before Hadamard butterfly) ----
  // 1. Each thread computes partial sum of squares
  float norm_sq = 0.0f;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      norm_sq += x_vals[c][i] * x_vals[c][i];
    }
  }

  // 2. Block-level reduction
  if constexpr (kNThreads == 1) {
    // Single thread: norm_sq is already complete
  } else if constexpr (kNWarps == 1) {
    // Single warp: use warp-level Allreduce
    SumOp<float> sum_op;
    norm_sq = Allreduce<kWarpSize>::run(norm_sq, sum_op);
  } else {
    // Multi-warp: warp reduce + cross-warp reduce via shared memory
    SumOp<float> sum_op;
    norm_sq = Allreduce<kWarpSize>::run(norm_sq, sum_op);

    float* smem_reduce = reinterpret_cast<float*>(smem_);
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (lane_id == 0) {
      smem_reduce[warp_id] = norm_sq;
    }
    __syncthreads();
    // Thread 0 reduces across warps
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kNWarps; ++w) {
        total += smem_reduce[w];
      }
      smem_reduce[0] = total;
    }
    __syncthreads();
    norm_sq = smem_reduce[0];
  }

  // 3. Compute norm and write output
  float norm = sqrtf(norm_sq);
  if (threadIdx.x == 0) {
    params.out_norms_ptr[batch_id] = norm;
  }

  // 4. Normalize x_vals in-place
  float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 1.0f;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      x_vals[c][i] *= inv_norm;
    }
  }
  // ---- End fused norm + normalize ----

  hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
    hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals_transposed[i][c] = x_vals[c][i];
      }
    }

    if constexpr (kNChunks == 12) {
      hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 20) {
      hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 28) {
      hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 40) {
      hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
    } else {
      constexpr int kLogNChunks = cilog2(kNChunks);
      static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
      hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[c][i] = x_vals_transposed[i][c];
      }
    }
  }

  // Fused: multiply by signs2 + scale + store (always float32 output)
  store_output_with_signs<kNChunks, kNElts, float>(out, x_vals, params.dim, params.scale, params.signs2_ptr);
}

template <typename Ktraits>
inline void set_max_dynamic_smem() {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  if constexpr (kSmemSize >= 48 * 1024) {
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
  }
}

template <typename Ktraits>
inline void launch_kernel(HadamardParamsBase& params, DLDevice device) {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  set_max_dynamic_smem<Ktraits>();
  auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
  host::LaunchKernel(dim3(params.batch), dim3(Ktraits::kNThreads), device, kSmemSize)(kernel, params);
  host::RuntimeDeviceCheck();
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamardKernelTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 3) {
    fast_hadamard_transform_launch<1, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_launch<2, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_launch<4, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_launch<8, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_launch<16, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_launch<32, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_launch<32, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_launch<128, 10, input_t>(params, device);
  } else if (params.log_N == 11) {
    fast_hadamard_transform_launch<256, 11, input_t>(params, device);
  } else if (params.log_N == 12) {
    fast_hadamard_transform_launch<256, 12, input_t>(params, device);
  } else if (params.log_N == 13) {
    fast_hadamard_transform_launch<256, 13, input_t>(params, device);
  } else if (params.log_N == 14) {
    fast_hadamard_transform_launch<256, 14, input_t>(params, device);
  } else if (params.log_N == 15) {
    fast_hadamard_transform_launch<256, 15, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_12N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard12NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_12N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_12N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_12N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_12N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_12N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_12N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_12N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_12N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_12N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_12N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_12N: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_20N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard20NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_20N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_20N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_20N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_20N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_20N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_20N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_20N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_20N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_20N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_20N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_20N: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_28N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard28NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_28N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_28N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_28N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_28N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_28N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_28N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_28N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_28N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_28N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_28N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_28N: unsupported log_N=", params.log_N);
  }
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_40N_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamard40NTraits<kNThreads, kLogN, input_t>;
  launch_kernel<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_40N_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 2) {
    fast_hadamard_transform_40N_launch<1, 2, input_t>(params, device);
  } else if (params.log_N == 3) {
    fast_hadamard_transform_40N_launch<2, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_40N_launch<4, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_40N_launch<8, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_40N_launch<16, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_40N_launch<32, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_40N_launch<64, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_40N_launch<128, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_40N_launch<256, 10, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_40N: unsupported log_N=", params.log_N);
  }
}

inline void set_hadamard_params(
    HadamardParamsBase& params,
    int64_t batch,
    int64_t dim,
    int64_t multiple,
    const tvm::ffi::TensorView x,
    const tvm::ffi::TensorView out,
    float scale) {
  std::memset(&params, 0, sizeof(params));
  params.batch = static_cast<int>(batch);
  params.dim = static_cast<int>(dim);
  params.log_N = ceil_log2(static_cast<int>(dim / multiple));
  params.x_ptr = const_cast<void*>(x.data_ptr());
  params.out_ptr = const_cast<void*>(out.data_ptr());
  params.x_batch_stride = x.stride(0);
  params.out_batch_stride = out.stride(0);
  params.scale = scale;
}

template <int kMultiple, typename DType>
inline void run_hadamard(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
  using namespace host;

  auto N = SymbolicSize{"batch"};
  auto D = SymbolicSize{"dim"};
  auto SX = SymbolicSize{"x_batch_stride"};
  auto SO = SymbolicSize{"out_batch_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({N, D}).with_strides({SX, 1}).with_dtype<DType>().with_device(device).verify(x);
  TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device).verify(out);

  const int64_t batch = N.unwrap();
  const int64_t dim = D.unwrap();

  RuntimeCheck(dim % kMultiple == 0, "hadamard: dim must be divisible by ", kMultiple);

  HadamardParamsBase params;
  set_hadamard_params(params, batch, dim, kMultiple, x, out, scale);

  if constexpr (kMultiple == 1) {
    RuntimeCheck(dim % 8 == 0, "fast_hadamard_transform only supports hidden dim divisible by 8");
    RuntimeCheck(dim <= 32768, "fast_hadamard_transform only supports hidden dim <= 32768");
    fast_hadamard_transform_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 12) {
    RuntimeCheck(dim % (4 * 12) == 0, "fast_hadamard_transform_12N only supports hidden dim divisible by 48");
    RuntimeCheck(dim <= 12 * 1024, "fast_hadamard_transform_12N only supports hidden dim <= 12288");
    fast_hadamard_transform_12N_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 20) {
    RuntimeCheck(dim % (4 * 20) == 0, "fast_hadamard_transform_20N only supports hidden dim divisible by 80");
    RuntimeCheck(dim <= 20 * 1024, "fast_hadamard_transform_20N only supports hidden dim <= 20480");
    fast_hadamard_transform_20N_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 28) {
    RuntimeCheck(dim % (4 * 28) == 0, "fast_hadamard_transform_28N only supports hidden dim divisible by 112");
    RuntimeCheck(dim <= 28 * 1024, "fast_hadamard_transform_28N only supports hidden dim <= 28672");
    fast_hadamard_transform_28N_cuda<DType>(params, device.unwrap());
  } else if constexpr (kMultiple == 40) {
    RuntimeCheck(dim % (4 * 40) == 0, "fast_hadamard_transform_40N only supports hidden dim divisible by 160");
    RuntimeCheck(dim <= 40 * 1024, "fast_hadamard_transform_40N only supports hidden dim <= 40960");
    fast_hadamard_transform_40N_cuda<DType>(params, device.unwrap());
  } else {
    Panic("Unsupported multiple");
  }
}

template <typename DType>
struct HadamardKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<1, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard12NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<12, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard20NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<20, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard28NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<28, DType>(x, out, scale);
  }
};

template <typename DType>
struct Hadamard40NKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out, float scale) {
    run_hadamard<40, DType>(x, out, scale);
  }
};


// --- Fused WHT rotation: launch helpers ---

template <typename Ktraits>
inline void launch_kernel_with_signs(HadamardParamsBase& params, DLDevice device) {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  set_max_dynamic_smem<Ktraits>();
  auto kernel = &fast_hadamard_transform_with_signs_kernel<Ktraits>;
  host::LaunchKernel(dim3(params.batch), dim3(Ktraits::kNThreads), device, kSmemSize)(kernel, params);
  host::RuntimeDeviceCheck();
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_with_signs_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamardKernelTraits<kNThreads, kLogN, input_t>;
  launch_kernel_with_signs<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_with_signs_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 3) {
    fast_hadamard_transform_with_signs_launch<1, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_with_signs_launch<2, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_with_signs_launch<4, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_with_signs_launch<8, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_with_signs_launch<16, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_with_signs_launch<32, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_with_signs_launch<32, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_with_signs_launch<128, 10, input_t>(params, device);
  } else if (params.log_N == 11) {
    fast_hadamard_transform_with_signs_launch<256, 11, input_t>(params, device);
  } else if (params.log_N == 12) {
    fast_hadamard_transform_with_signs_launch<256, 12, input_t>(params, device);
  } else if (params.log_N == 13) {
    fast_hadamard_transform_with_signs_launch<256, 13, input_t>(params, device);
  } else if (params.log_N == 14) {
    fast_hadamard_transform_with_signs_launch<256, 14, input_t>(params, device);
  } else if (params.log_N == 15) {
    fast_hadamard_transform_with_signs_launch<256, 15, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_with_signs: unsupported log_N=", params.log_N);
  }
}

inline void set_hadamard_params_with_signs(
    HadamardParamsBase& params,
    int64_t batch,
    int64_t dim,
    int64_t multiple,
    const tvm::ffi::TensorView x,
    const tvm::ffi::TensorView out,
    const tvm::ffi::TensorView signs1,
    const tvm::ffi::TensorView signs2,
    float scale) {
  set_hadamard_params(params, batch, dim, multiple, x, out, scale);
  params.signs1_ptr = reinterpret_cast<const float*>(signs1.data_ptr());
  params.signs2_ptr = reinterpret_cast<const float*>(signs2.data_ptr());
}

template <int kMultiple, typename DType>
inline void run_hadamard_with_signs(
    const tvm::ffi::TensorView x, const tvm::ffi::TensorView out,
    const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
    float scale) {
  using namespace host;

  auto N = SymbolicSize{"batch"};
  auto D = SymbolicSize{"dim"};
  auto SX = SymbolicSize{"x_batch_stride"};
  auto SO = SymbolicSize{"out_batch_stride"};
  auto DS = SymbolicSize{"signs_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({N, D}).with_strides({SX, 1}).with_dtype<DType>().with_device(device).verify(x);
  TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<DType>().with_device(device).verify(out);
  TensorMatcher({DS}).with_dtype<float>().with_device(device).verify(signs1);
  TensorMatcher({DS}).with_dtype<float>().with_device(device).verify(signs2);

  const int64_t batch = N.unwrap();
  const int64_t dim = D.unwrap();

  RuntimeCheck(dim % 8 == 0, "fast_hadamard_transform_with_signs only supports hidden dim divisible by 8");
  RuntimeCheck(dim <= 32768, "fast_hadamard_transform_with_signs only supports hidden dim <= 32768");
  RuntimeCheck(DS.unwrap() == dim, "signs dim must match input dim");

  HadamardParamsBase params;
  set_hadamard_params_with_signs(params, batch, dim, kMultiple, x, out, signs1, signs2, scale);
  fast_hadamard_transform_with_signs_cuda<DType>(params, device.unwrap());
}

template <typename DType>
struct HadamardWithSignsKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
                  float scale) {
    run_hadamard_with_signs<1, DType>(x, out, signs1, signs2, scale);
  }
};


// --- Fused norm+normalize+WHT: launch helpers ---

template <typename Ktraits>
inline void launch_kernel_with_signs_and_norm(HadamardParamsBase& params, DLDevice device) {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  if constexpr (kSmemSize >= 48 * 1024) {
    auto kernel = &fast_hadamard_transform_with_signs_and_norm_kernel<Ktraits>;
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
  }
  auto kernel = &fast_hadamard_transform_with_signs_and_norm_kernel<Ktraits>;
  host::LaunchKernel(dim3(params.batch), dim3(Ktraits::kNThreads), device, kSmemSize)(kernel, params);
  host::RuntimeDeviceCheck();
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_with_signs_and_norm_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamardKernelTraits<kNThreads, kLogN, input_t>;
  launch_kernel_with_signs_and_norm<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_with_signs_and_norm_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 3) {
    fast_hadamard_transform_with_signs_and_norm_launch<1, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_with_signs_and_norm_launch<2, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_with_signs_and_norm_launch<4, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_with_signs_and_norm_launch<8, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_with_signs_and_norm_launch<16, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_with_signs_and_norm_launch<32, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_with_signs_and_norm_launch<32, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_with_signs_and_norm_launch<128, 10, input_t>(params, device);
  } else if (params.log_N == 11) {
    fast_hadamard_transform_with_signs_and_norm_launch<256, 11, input_t>(params, device);
  } else if (params.log_N == 12) {
    fast_hadamard_transform_with_signs_and_norm_launch<256, 12, input_t>(params, device);
  } else if (params.log_N == 13) {
    fast_hadamard_transform_with_signs_and_norm_launch<256, 13, input_t>(params, device);
  } else if (params.log_N == 14) {
    fast_hadamard_transform_with_signs_and_norm_launch<256, 14, input_t>(params, device);
  } else if (params.log_N == 15) {
    fast_hadamard_transform_with_signs_and_norm_launch<256, 15, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_with_signs_and_norm: unsupported log_N=", params.log_N);
  }
}

template <int kMultiple, typename DType>
inline void run_hadamard_with_signs_and_norm(
    const tvm::ffi::TensorView x, const tvm::ffi::TensorView out,
    const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
    const tvm::ffi::TensorView out_norms,
    float scale) {
  using namespace host;

  auto N = SymbolicSize{"batch"};
  auto D = SymbolicSize{"dim"};
  auto SX = SymbolicSize{"x_batch_stride"};
  auto SO = SymbolicSize{"out_batch_stride"};
  auto DS = SymbolicSize{"signs_dim"};
  auto NN = SymbolicSize{"norms_batch"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({N, D}).with_strides({SX, 1}).with_dtype<DType>().with_device(device).verify(x);
  // Output is always float32 regardless of input dtype
  TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<float>().with_device(device).verify(out);
  TensorMatcher({DS}).with_dtype<float>().with_device(device).verify(signs1);
  TensorMatcher({DS}).with_dtype<float>().with_device(device).verify(signs2);
  TensorMatcher({NN}).with_dtype<float>().with_device(device).verify(out_norms);

  const int64_t batch = N.unwrap();
  const int64_t dim = D.unwrap();

  RuntimeCheck(dim % 8 == 0, "fast_hadamard_transform_with_signs_and_norm only supports hidden dim divisible by 8");
  RuntimeCheck(dim <= 32768, "fast_hadamard_transform_with_signs_and_norm only supports hidden dim <= 32768");
  RuntimeCheck(DS.unwrap() == dim, "signs dim must match input dim");
  RuntimeCheck(NN.unwrap() >= batch, "out_norms must have at least batch elements");

  HadamardParamsBase params;
  set_hadamard_params_with_signs(params, batch, dim, kMultiple, x, out, signs1, signs2, scale);
  params.out_norms_ptr = reinterpret_cast<float*>(const_cast<void*>(out_norms.data_ptr()));

  fast_hadamard_transform_with_signs_and_norm_cuda<DType>(params, device.unwrap());
}

template <typename DType>
struct HadamardWithSignsAndNormKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
                  const tvm::ffi::TensorView out_norms,
                  float scale) {
    run_hadamard_with_signs_and_norm<1, DType>(x, out, signs1, signs2, out_norms, scale);
  }
};


// --- Dual-input fused norm+normalize+WHT kernel (K/V without torch.cat) ---

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads) void fast_hadamard_transform_with_signs_and_norm_dual_kernel(HadamardParamsBase params) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(Ktraits::kNElts);
  static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");

  constexpr int kWarpSize = kNThreads < 32 ? kNThreads : 32;
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");

  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

  constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
  constexpr int kNExchanges = kNChunks / kChunksPerExchange;
  static_assert(kNExchanges * kChunksPerExchange == kNChunks);

  extern __shared__ char smem_[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_);

  const int batch_id = static_cast<int>(blockIdx.x);

  // Dual-input selection: blocks [0, split_batch) read from x_ptr,
  // blocks [split_batch, batch) read from x2_ptr.
  input_t* x;
  if (batch_id < params.split_batch) {
    x = reinterpret_cast<input_t*>(params.x_ptr) + batch_id * params.x_batch_stride;
  } else {
    int local_id = batch_id - params.split_batch;
    x = reinterpret_cast<input_t*>(params.x2_ptr) + local_id * params.x2_batch_stride;
  }

  // Output is always float32 for the fused norm+WHT kernel
  float* out = reinterpret_cast<float*>(params.out_ptr) + batch_id * params.out_batch_stride;

  float x_vals[kNChunks][kNElts];
  // Fused: load + multiply by signs1
  load_input_with_signs<kNChunks, kNElts, input_t>(x, x_vals, params.dim, params.signs1_ptr);

  // ---- Fused norm + normalize ----
  float norm_sq = 0.0f;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      norm_sq += x_vals[c][i] * x_vals[c][i];
    }
  }

  if constexpr (kNThreads == 1) {
  } else if constexpr (kNWarps == 1) {
    SumOp<float> sum_op;
    norm_sq = Allreduce<kWarpSize>::run(norm_sq, sum_op);
  } else {
    SumOp<float> sum_op;
    norm_sq = Allreduce<kWarpSize>::run(norm_sq, sum_op);
    float* smem_reduce = reinterpret_cast<float*>(smem_);
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (lane_id == 0) {
      smem_reduce[warp_id] = norm_sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kNWarps; ++w) {
        total += smem_reduce[w];
      }
      smem_reduce[0] = total;
    }
    __syncthreads();
    norm_sq = smem_reduce[0];
  }

  float norm = sqrtf(norm_sq);
  if (threadIdx.x == 0) {
    params.out_norms_ptr[batch_id] = norm;
  }

  float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 1.0f;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      x_vals[c][i] *= inv_norm;
    }
  }
  // ---- End fused norm + normalize ----

  hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
    hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals_transposed[i][c] = x_vals[c][i];
      }
    }

    if constexpr (kNChunks == 12) {
      hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 20) {
      hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 28) {
      hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 40) {
      hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
    } else {
      constexpr int kLogNChunks = cilog2(kNChunks);
      static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
      hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[c][i] = x_vals_transposed[i][c];
      }
    }
  }

  // Fused: multiply by signs2 + scale + store (always float32 output)
  store_output_with_signs<kNChunks, kNElts, float>(out, x_vals, params.dim, params.scale, params.signs2_ptr);
}

// --- Dual-input launch helpers ---

template <typename Ktraits>
inline void launch_kernel_with_signs_and_norm_dual(HadamardParamsBase& params, DLDevice device) {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  if constexpr (kSmemSize >= 48 * 1024) {
    auto kernel = &fast_hadamard_transform_with_signs_and_norm_dual_kernel<Ktraits>;
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
  }
  auto kernel = &fast_hadamard_transform_with_signs_and_norm_dual_kernel<Ktraits>;
  host::LaunchKernel(dim3(params.batch), dim3(Ktraits::kNThreads), device, kSmemSize)(kernel, params);
  host::RuntimeDeviceCheck();
}

template <int kNThreads, int kLogN, typename input_t>
inline void fast_hadamard_transform_with_signs_and_norm_dual_launch(HadamardParamsBase& params, DLDevice device) {
  using Ktraits = FastHadamardKernelTraits<kNThreads, kLogN, input_t>;
  launch_kernel_with_signs_and_norm_dual<Ktraits>(params, device);
}

template <typename input_t>
inline void fast_hadamard_transform_with_signs_and_norm_dual_cuda(HadamardParamsBase& params, DLDevice device) {
  if (params.log_N == 3) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<1, 3, input_t>(params, device);
  } else if (params.log_N == 4) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<2, 4, input_t>(params, device);
  } else if (params.log_N == 5) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<4, 5, input_t>(params, device);
  } else if (params.log_N == 6) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<8, 6, input_t>(params, device);
  } else if (params.log_N == 7) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<16, 7, input_t>(params, device);
  } else if (params.log_N == 8) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<32, 8, input_t>(params, device);
  } else if (params.log_N == 9) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<32, 9, input_t>(params, device);
  } else if (params.log_N == 10) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<128, 10, input_t>(params, device);
  } else if (params.log_N == 11) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<256, 11, input_t>(params, device);
  } else if (params.log_N == 12) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<256, 12, input_t>(params, device);
  } else if (params.log_N == 13) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<256, 13, input_t>(params, device);
  } else if (params.log_N == 14) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<256, 14, input_t>(params, device);
  } else if (params.log_N == 15) {
    fast_hadamard_transform_with_signs_and_norm_dual_launch<256, 15, input_t>(params, device);
  } else {
    host::Panic("fast_hadamard_transform_with_signs_and_norm_dual: unsupported log_N=", params.log_N);
  }
}

template <int kMultiple, typename DType>
inline void run_hadamard_with_signs_and_norm_dual(
    const tvm::ffi::TensorView x1, const tvm::ffi::TensorView x2,
    const tvm::ffi::TensorView out,
    const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
    const tvm::ffi::TensorView out_norms,
    float scale) {
  using namespace host;

  auto N1 = SymbolicSize{"batch1"};
  auto N2 = SymbolicSize{"batch2"};
  auto D = SymbolicSize{"dim"};
  auto SX1 = SymbolicSize{"x1_batch_stride"};
  auto SX2 = SymbolicSize{"x2_batch_stride"};
  auto SO = SymbolicSize{"out_batch_stride"};
  auto DS = SymbolicSize{"signs_dim"};
  auto NN = SymbolicSize{"norms_batch"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({N1, D}).with_strides({SX1, 1}).with_dtype<DType>().with_device(device).verify(x1);
  TensorMatcher({N2, D}).with_strides({SX2, 1}).with_dtype<DType>().with_device(device).verify(x2);
  // Output covers both x1 and x2 batch dims
  auto N_TOTAL = SymbolicSize{"total_batch"};
  TensorMatcher({N_TOTAL, D}).with_strides({SO, 1}).with_dtype<float>().with_device(device).verify(out);
  TensorMatcher({DS}).with_dtype<float>().with_device(device).verify(signs1);
  TensorMatcher({DS}).with_dtype<float>().with_device(device).verify(signs2);
  TensorMatcher({NN}).with_dtype<float>().with_device(device).verify(out_norms);

  const int64_t batch1 = N1.unwrap();
  const int64_t batch2 = N2.unwrap();
  const int64_t total_batch = N_TOTAL.unwrap();
  const int64_t dim = D.unwrap();

  RuntimeCheck(total_batch == batch1 + batch2, "out batch must equal x1 batch + x2 batch");
  RuntimeCheck(dim % 8 == 0, "fast_hadamard_transform_with_signs_and_norm_dual only supports hidden dim divisible by 8");
  RuntimeCheck(dim <= 32768, "fast_hadamard_transform_with_signs_and_norm_dual only supports hidden dim <= 32768");
  RuntimeCheck(DS.unwrap() == dim, "signs dim must match input dim");
  RuntimeCheck(NN.unwrap() >= total_batch, "out_norms must have at least total_batch elements");

  HadamardParamsBase params;
  std::memset(&params, 0, sizeof(params));
  params.batch = static_cast<int>(total_batch);
  params.dim = static_cast<int>(dim);
  params.log_N = ceil_log2(static_cast<int>(dim / kMultiple));
  params.x_ptr = const_cast<void*>(x1.data_ptr());
  params.out_ptr = const_cast<void*>(out.data_ptr());
  params.x_batch_stride = x1.stride(0);
  params.out_batch_stride = out.stride(0);
  params.scale = scale;
  params.signs1_ptr = reinterpret_cast<const float*>(signs1.data_ptr());
  params.signs2_ptr = reinterpret_cast<const float*>(signs2.data_ptr());
  params.out_norms_ptr = reinterpret_cast<float*>(const_cast<void*>(out_norms.data_ptr()));

  // Dual-input fields
  params.x2_ptr = const_cast<void*>(x2.data_ptr());
  params.x2_batch_stride = x2.stride(0);
  params.split_batch = static_cast<int>(batch1);

  fast_hadamard_transform_with_signs_and_norm_dual_cuda<DType>(params, device.unwrap());
}

template <typename DType>
struct HadamardWithSignsAndNormDualKernel {
  static void run(const tvm::ffi::TensorView x1, const tvm::ffi::TensorView x2,
                  const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
                  const tvm::ffi::TensorView out_norms,
                  float scale) {
    run_hadamard_with_signs_and_norm_dual<1, DType>(x1, x2, out, signs1, signs2, out_norms, scale);
  }
};


// =============================================================================
// Phase 2: Fully-fused norm+WHT+searchsorted+pack+scatter_store kernel
// Single CUDA kernel: bf16 load → registers → norm+WHT → quant+pack → scatter store
// Eliminates the WHT→Pack HBM round-trip (saves 512B read + 512B write per (token,head)).
// =============================================================================

using ::FusedNormWhtPackStoreParams;

template <typename Ktraits, int kBitWidth>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fused_norm_wht_pack_store_kernel(FusedNormWhtPackStoreParams params) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(Ktraits::kNElts);
  static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");

  constexpr int kWarpSize = kNThreads < 32 ? kNThreads : 32;
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");

  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");

  constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
  constexpr int kNExchanges = kNChunks / kChunksPerExchange;
  static_assert(kNExchanges * kChunksPerExchange == kNChunks);

  // Pack ratio: how many elements per packed byte
  constexpr int kPackRatio = (kBitWidth == 4) ? 2 : 4;
  // Number of packed bytes each thread produces
  constexpr int kPackedPerThread = kNChunks * kNElts / kPackRatio;

  extern __shared__ char smem_[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_);

  const int batch_id = static_cast<int>(blockIdx.x);

  // ===== Dual-input selection =====
  const bool is_v = (batch_id >= params.split_batch);
  const int local_batch = is_v ? (batch_id - params.split_batch) : batch_id;
  const int token_id = local_batch / params.heads;
  const int head_id = local_batch % params.heads;

  input_t* x;
  if (!is_v) {
    x = reinterpret_cast<input_t*>(params.x_ptr) + local_batch * params.x_batch_stride;
  } else {
    x = reinterpret_cast<input_t*>(params.x2_ptr) + local_batch * params.x2_batch_stride;
  }

  // ===== Phase 1: Load + Signs1 =====
  float x_vals[kNChunks][kNElts];
  load_input_with_signs<kNChunks, kNElts, input_t>(x, x_vals, params.dim, params.signs1_ptr);

  // ===== Phase 1b: Compute L2 norm + normalize =====
  float norm_sq = 0.0f;
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      norm_sq += x_vals[c][i] * x_vals[c][i];
    }
  }

  // Block-level norm reduction
  if constexpr (kNThreads == 1) {
    // Single thread: norm_sq is already complete
  } else if constexpr (kNWarps == 1) {
    SumOp<float> sum_op;
    norm_sq = Allreduce<kWarpSize>::run(norm_sq, sum_op);
  } else {
    SumOp<float> sum_op;
    norm_sq = Allreduce<kWarpSize>::run(norm_sq, sum_op);
    float* smem_reduce = reinterpret_cast<float*>(smem_);
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (lane_id == 0) {
      smem_reduce[warp_id] = norm_sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kNWarps; ++w) {
        total += smem_reduce[w];
      }
      smem_reduce[0] = total;
    }
    __syncthreads();
    norm_sq = smem_reduce[0];
  }

  float norm = sqrtf(norm_sq);
  float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 1.0f;

#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      x_vals[c][i] *= inv_norm;
    }
  }

  // ===== Phase 2: WHT Butterfly =====
  hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
    hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals_transposed[i][c] = x_vals[c][i];
      }
    }

    if constexpr (kNChunks == 12) {
      hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 20) {
      hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 28) {
      hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
    } else if constexpr (kNChunks == 40) {
      hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
    } else {
      constexpr int kLogNChunks = cilog2(kNChunks);
      static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
      hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[c][i] = x_vals_transposed[i][c];
      }
    }
  }

  // Apply signs2 and scale (in-place, no store to HBM)
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    int base_idx = (c * kNThreads + threadIdx.x) * kNElts;
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      x_vals[c][i] *= params.signs2_ptr[base_idx + i] * params.scale;
    }
  }

  // ===== Phase 3: Searchsorted + Pack =====
  // Load boundaries and centroids into registers (small enough: <=15 boundaries, <=16 centroids)
  const float* __restrict__ boundaries = params.boundaries_ptr;
  const float* __restrict__ centroids = params.centroids_ptr;

  // Compile-time boundary count derived from bit width
  constexpr int kNBoundaries = (1 << kBitWidth) - 1;  // 15 for 4-bit, 3 for 2-bit

  uint8_t packed[kPackedPerThread];
  float qnorm_sq = 0.0f;

  if constexpr (kBitWidth == 4) {
    // 4-bit: pack pairs of elements into uint8 (low nibble = even, high nibble = odd)
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; i += 2) {
        float val_even = x_vals[c][i];
        float val_odd = x_vals[c][i + 1];

        // Searchsorted: count boundaries less than val (fully unrolled)
        int idx_even = 0;
        int idx_odd = 0;
#pragma unroll
        for (int b = 0; b < kNBoundaries; ++b) {
          float bound = boundaries[b];
          idx_even += (val_even > bound) ? 1 : 0;
          idx_odd += (val_odd > bound) ? 1 : 0;
        }

        // Accumulate qnorm from codebook centroids
        float c_even = centroids[idx_even];
        float c_odd = centroids[idx_odd];
        qnorm_sq += c_even * c_even + c_odd * c_odd;

        // Pack: low nibble = even index, high nibble = odd index
        int pack_idx = (c * kNElts + i) / 2;
        packed[pack_idx] = static_cast<uint8_t>(((idx_odd & 0xF) << 4) | (idx_even & 0xF));
      }
    }
  } else {
    // 2-bit: pack quads of elements into uint8 (bits [1:0]=idx0, [3:2]=idx1, [5:4]=idx2, [7:6]=idx3)
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; i += 4) {
        float val0 = x_vals[c][i];
        float val1 = x_vals[c][i + 1];
        float val2 = x_vals[c][i + 2];
        float val3 = x_vals[c][i + 3];

        int idx0 = 0, idx1 = 0, idx2 = 0, idx3 = 0;
#pragma unroll
        for (int b = 0; b < kNBoundaries; ++b) {
          float bound = boundaries[b];
          idx0 += (val0 > bound) ? 1 : 0;
          idx1 += (val1 > bound) ? 1 : 0;
          idx2 += (val2 > bound) ? 1 : 0;
          idx3 += (val3 > bound) ? 1 : 0;
        }

        float c0 = centroids[idx0];
        float c1 = centroids[idx1];
        float c2 = centroids[idx2];
        float c3 = centroids[idx3];
        qnorm_sq += c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;

        int pack_idx = (c * kNElts + i) / 4;
        packed[pack_idx] = static_cast<uint8_t>(
            ((idx3 & 0x3) << 6) | ((idx2 & 0x3) << 4) | ((idx1 & 0x3) << 2) | (idx0 & 0x3));
      }
    }
  }

  // ===== Phase 4: Block-level qnorm reduction + compute dscale =====
  if constexpr (kNThreads == 1) {
    // Single thread: qnorm_sq is already complete
  } else if constexpr (kNWarps == 1) {
    SumOp<float> sum_op;
    qnorm_sq = Allreduce<kWarpSize>::run(qnorm_sq, sum_op);
  } else {
    SumOp<float> sum_op;
    qnorm_sq = Allreduce<kWarpSize>::run(qnorm_sq, sum_op);
    float* smem_reduce = reinterpret_cast<float*>(smem_);
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    if (lane_id == 0) {
      smem_reduce[warp_id] = qnorm_sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kNWarps; ++w) {
        total += smem_reduce[w];
      }
      smem_reduce[0] = total;
    }
    __syncthreads();
    qnorm_sq = smem_reduce[0];
  }

  float qnorm = sqrtf(qnorm_sq);
  float safe_qnorm = (qnorm > 1e-10f) ? qnorm : 1.0f;
  float dscale_val = norm / safe_qnorm;

  // ===== Phase 5: Scatter Store =====
  int64_t pool_slot = params.loc[token_id];

  // Store packed bytes to correct global positions.
  // Element at x_vals[c][i] has global dim position: (c * kNThreads + threadIdx.x) * kNElts + i
  // Its packed byte goes to output offset: global_pos / kPackRatio
  uint8_t* dst_buf = is_v ? params.v_buffer : params.k_buffer;
  int64_t dst_stride_s = is_v ? params.vb_stride_s : params.kb_stride_s;
  int64_t dst_stride_h = is_v ? params.vb_stride_h : params.kb_stride_h;

  uint8_t* dst = dst_buf + pool_slot * dst_stride_s + head_id * dst_stride_h;

  if constexpr (kNChunks == 1) {
    // Optimization: when kNChunks==1, thread's packed bytes are already contiguous
    int thread_packed_offset = threadIdx.x * kPackedPerThread;
    if constexpr (kPackedPerThread >= 4 && kPackedPerThread % 4 == 0) {
#pragma unroll
      for (int i = 0; i < kPackedPerThread; i += 4) {
        uint32_t val;
        memcpy(&val, &packed[i], 4);
        *reinterpret_cast<uint32_t*>(dst + thread_packed_offset + i) = val;
      }
    } else if constexpr (kPackedPerThread >= 2 && kPackedPerThread % 2 == 0) {
#pragma unroll
      for (int i = 0; i < kPackedPerThread; i += 2) {
        uint16_t val;
        memcpy(&val, &packed[i], 2);
        *reinterpret_cast<uint16_t*>(dst + thread_packed_offset + i) = val;
      }
    } else {
#pragma unroll
      for (int i = 0; i < kPackedPerThread; i++) {
        dst[thread_packed_offset + i] = packed[i];
      }
    }
  } else {
    // General case: each chunk's packed bytes go to non-contiguous positions
    // Per chunk: kNElts/kPackRatio packed bytes, starting at global byte offset
    // (c * kNThreads + threadIdx.x) * kNElts / kPackRatio
    constexpr int kBytesPerChunk = kNElts / kPackRatio;
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
      int global_byte_offset = (c * kNThreads + threadIdx.x) * kBytesPerChunk;
      int local_offset = c * kBytesPerChunk;
      if constexpr (kBytesPerChunk >= 4 && kBytesPerChunk % 4 == 0) {
#pragma unroll
        for (int i = 0; i < kBytesPerChunk; i += 4) {
          uint32_t val;
          memcpy(&val, &packed[local_offset + i], 4);
          *reinterpret_cast<uint32_t*>(dst + global_byte_offset + i) = val;
        }
      } else if constexpr (kBytesPerChunk >= 2 && kBytesPerChunk % 2 == 0) {
#pragma unroll
        for (int i = 0; i < kBytesPerChunk; i += 2) {
          uint16_t val;
          memcpy(&val, &packed[local_offset + i], 2);
          *reinterpret_cast<uint16_t*>(dst + global_byte_offset + i) = val;
        }
      } else {
#pragma unroll
        for (int i = 0; i < kBytesPerChunk; i++) {
          dst[global_byte_offset + i] = packed[local_offset + i];
        }
      }
    }
  }

  // Store dscale (only thread 0 writes to avoid conflicts)
  if (threadIdx.x == 0) {
    __nv_bfloat16* ds_buf = is_v
        ? reinterpret_cast<__nv_bfloat16*>(params.v_dscale)
        : reinterpret_cast<__nv_bfloat16*>(params.k_dscale);
    int64_t ds_stride = is_v ? params.vds_stride : params.kds_stride;
    ds_buf[pool_slot * ds_stride + head_id] = __float2bfloat16(dscale_val);
  }
}

// --- Fused norm+WHT+pack+store: launch helpers ---

template <typename Ktraits, int kBitWidth>
inline void launch_fused_norm_wht_pack_store(FusedNormWhtPackStoreParams& params, DLDevice device) {
  constexpr int kSmemSize = Ktraits::kSmemSize;
  if constexpr (kSmemSize >= 48 * 1024) {
    auto kernel = &fused_norm_wht_pack_store_kernel<Ktraits, kBitWidth>;
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
  }
  auto kernel = &fused_norm_wht_pack_store_kernel<Ktraits, kBitWidth>;
  host::LaunchKernel(dim3(params.batch), dim3(Ktraits::kNThreads), device, kSmemSize)(kernel, params);
  host::RuntimeDeviceCheck();
}

template <int kNThreads, int kLogN, typename input_t, int kBitWidth>
inline void fused_norm_wht_pack_store_launch(FusedNormWhtPackStoreParams& params, DLDevice device) {
  using Ktraits = FastHadamardKernelTraits<kNThreads, kLogN, input_t>;
  launch_fused_norm_wht_pack_store<Ktraits, kBitWidth>(params, device);
}

template <typename input_t, int kBitWidth>
inline void fused_norm_wht_pack_store_cuda(FusedNormWhtPackStoreParams& params, DLDevice device) {
  if (params.log_N == 3) {
    fused_norm_wht_pack_store_launch<1, 3, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 4) {
    fused_norm_wht_pack_store_launch<2, 4, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 5) {
    fused_norm_wht_pack_store_launch<4, 5, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 6) {
    fused_norm_wht_pack_store_launch<8, 6, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 7) {
    fused_norm_wht_pack_store_launch<16, 7, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 8) {
    fused_norm_wht_pack_store_launch<32, 8, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 9) {
    fused_norm_wht_pack_store_launch<32, 9, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 10) {
    fused_norm_wht_pack_store_launch<128, 10, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 11) {
    fused_norm_wht_pack_store_launch<256, 11, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 12) {
    fused_norm_wht_pack_store_launch<256, 12, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 13) {
    fused_norm_wht_pack_store_launch<256, 13, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 14) {
    fused_norm_wht_pack_store_launch<256, 14, input_t, kBitWidth>(params, device);
  } else if (params.log_N == 15) {
    fused_norm_wht_pack_store_launch<256, 15, input_t, kBitWidth>(params, device);
  } else {
    host::Panic("fused_norm_wht_pack_store: unsupported log_N=", params.log_N);
  }
}

template <int kMultiple, typename DType, int kBitWidth>
inline void run_fused_norm_wht_pack_store(
    const tvm::ffi::TensorView x1, const tvm::ffi::TensorView x2,
    const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
    float scale,
    const tvm::ffi::TensorView boundaries, const tvm::ffi::TensorView centroids,
    const tvm::ffi::TensorView k_buffer, const tvm::ffi::TensorView v_buffer,
    const tvm::ffi::TensorView k_dscale, const tvm::ffi::TensorView v_dscale,
    const tvm::ffi::TensorView loc,
    int tokens, int heads) {
  using namespace host;

  auto B = SymbolicSize{"batch"};
  auto D = SymbolicSize{"dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  // Verify x1 to bind device and extract dim
  TensorMatcher({B, D}).with_dtype<DType>().with_device(device).verify(x1);

  const int64_t dim = D.unwrap();
  const int64_t total_batch = 2 * static_cast<int64_t>(tokens) * static_cast<int64_t>(heads);

  RuntimeCheck(dim % 8 == 0, "fused_norm_wht_pack_store: dim must be divisible by 8");
  RuntimeCheck(dim <= 32768, "fused_norm_wht_pack_store: dim must be <= 32768");

  FusedNormWhtPackStoreParams params;
  std::memset(&params, 0, sizeof(params));

  params.batch = static_cast<int>(total_batch);
  params.dim = static_cast<int>(dim);
  params.log_N = ceil_log2(static_cast<int>(dim / kMultiple));
  params.x_ptr = const_cast<void*>(x1.data_ptr());
  params.x2_ptr = const_cast<void*>(x2.data_ptr());
  params.x_batch_stride = x1.stride(0);
  params.x2_batch_stride = x2.stride(0);
  params.split_batch = tokens * heads;
  params.signs1_ptr = reinterpret_cast<const float*>(signs1.data_ptr());
  params.signs2_ptr = reinterpret_cast<const float*>(signs2.data_ptr());
  params.scale = scale;
  params.boundaries_ptr = reinterpret_cast<const float*>(boundaries.data_ptr());
  params.centroids_ptr = reinterpret_cast<const float*>(centroids.data_ptr());
  params.n_boundaries = (1 << kBitWidth) - 1;  // compile-time: 4bit→15, 2bit→3
  params.k_buffer = reinterpret_cast<uint8_t*>(const_cast<void*>(k_buffer.data_ptr()));
  params.v_buffer = reinterpret_cast<uint8_t*>(const_cast<void*>(v_buffer.data_ptr()));
  params.k_dscale = const_cast<void*>(k_dscale.data_ptr());
  params.v_dscale = const_cast<void*>(v_dscale.data_ptr());
  params.loc = reinterpret_cast<const int64_t*>(loc.data_ptr());
  params.kb_stride_s = k_buffer.stride(0);
  params.kb_stride_h = k_buffer.stride(1);
  params.vb_stride_s = v_buffer.stride(0);
  params.vb_stride_h = v_buffer.stride(1);
  params.kds_stride = k_dscale.stride(0);
  params.vds_stride = v_dscale.stride(0);
  params.tokens = tokens;
  params.heads = heads;
  params.packed_dim = static_cast<int>(dim) / (kBitWidth == 4 ? 2 : 4);

  fused_norm_wht_pack_store_cuda<DType, kBitWidth>(params, device.unwrap());
}

template <typename DType>
struct FusedNormWhtPackStore4bitKernel {
  static void run(const tvm::ffi::TensorView x1, const tvm::ffi::TensorView x2,
                  const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
                  float scale,
                  const tvm::ffi::TensorView boundaries, const tvm::ffi::TensorView centroids,
                  const tvm::ffi::TensorView k_buffer, const tvm::ffi::TensorView v_buffer,
                  const tvm::ffi::TensorView k_dscale, const tvm::ffi::TensorView v_dscale,
                  const tvm::ffi::TensorView loc,
                  int tokens, int heads) {
    run_fused_norm_wht_pack_store<1, DType, 4>(
        x1, x2, signs1, signs2, scale, boundaries, centroids,
        k_buffer, v_buffer, k_dscale, v_dscale, loc, tokens, heads);
  }
};

template <typename DType>
struct FusedNormWhtPackStore2bitKernel {
  static void run(const tvm::ffi::TensorView x1, const tvm::ffi::TensorView x2,
                  const tvm::ffi::TensorView signs1, const tvm::ffi::TensorView signs2,
                  float scale,
                  const tvm::ffi::TensorView boundaries, const tvm::ffi::TensorView centroids,
                  const tvm::ffi::TensorView k_buffer, const tvm::ffi::TensorView v_buffer,
                  const tvm::ffi::TensorView k_dscale, const tvm::ffi::TensorView v_dscale,
                  const tvm::ffi::TensorView loc,
                  int tokens, int heads) {
    run_fused_norm_wht_pack_store<1, DType, 2>(
        x1, x2, signs1, signs2, scale, boundaries, centroids,
        k_buffer, v_buffer, k_dscale, v_dscale, loc, tokens, heads);
  }
};

}  // namespace
