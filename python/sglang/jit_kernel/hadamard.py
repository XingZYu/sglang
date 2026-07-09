from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from sglang.jit_kernel.utils import KERNEL_PATH, cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_hadamard_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    hadamard_include_dir = (KERNEL_PATH / "csrc" / "fast-hadamard-transform").resolve()
    return load_jit(
        "hadamard",
        *args,
        cuda_files=["fast-hadamard-transform/hadamard_jit.cuh"],
        cuda_wrappers=[
            ("hadamard_transform", f"HadamardKernel<{args}>::run"),
            ("hadamard_transform_with_signs", f"HadamardWithSignsKernel<{args}>::run"),
            ("hadamard_transform_with_signs_and_norm", f"HadamardWithSignsAndNormKernel<{args}>::run"),
            ("hadamard_transform_with_signs_and_norm_dual", f"HadamardWithSignsAndNormDualKernel<{args}>::run"),
            ("fused_norm_wht_pack_store_4bit", f"FusedNormWhtPackStore4bitKernel<{args}>::run"),
            ("fused_norm_wht_pack_store_2bit", f"FusedNormWhtPackStore2bitKernel<{args}>::run"),
            ("hadamard_transform_12n", f"Hadamard12NKernel<{args}>::run"),
            ("hadamard_transform_20n", f"Hadamard20NKernel<{args}>::run"),
            ("hadamard_transform_28n", f"Hadamard28NKernel<{args}>::run"),
            ("hadamard_transform_40n", f"Hadamard40NKernel<{args}>::run"),
        ],
        extra_include_paths=[str(hadamard_include_dir)],
    )


def _hadamard_transform_impl(
    x: torch.Tensor,
    scale: float,
    pad_multiple: int,
    kernel_fn: Callable,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError(f"{kernel_fn.__name__} only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    needs_pad = dim_og % pad_multiple != 0
    if needs_pad:
        x = torch.nn.functional.pad(x, (0, pad_multiple - dim_og % pad_multiple))

    out = torch.empty_like(x)
    kernel_fn(x, out, scale)

    if needs_pad:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def _hadamard_transform_fake_impl(
    x: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(fake_impl=_hadamard_transform_fake_impl)
def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 8, module.hadamard_transform)


def hadamard_transform_with_signs(
    x: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused WHT rotation: out = signs2 * H(signs1 * x) * scale.

    Fuses signs1 multiply, Hadamard transform, and signs2 multiply into a
    single CUDA kernel launch, eliminating 2 elementwise kernel launches.

    Args:
        x: (..., dim) tensor, any dtype. Will be cast to float32 internally.
        signs1: (dim,) float32 sign vector applied before Hadamard.
        signs2: (dim,) float32 sign vector applied after Hadamard.
        scale: scalar multiplier (typically 1/sqrt(dim)).
        out: optional pre-allocated output tensor (same shape as x).
            If None, will be allocated internally.

    Returns:
        out: same shape as x, float32.
    """
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform_with_signs only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)

    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    # Use x's native dtype — the CUDA kernel handles bf16/fp16 I/O
    # with float32 computation internally (load converts to float,
    # store converts back to input_t)
    if out is not None:
        out = out.reshape(-1, dim_og)
    else:
        out = torch.empty_like(x)
    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform_with_signs(x, out, signs1, signs2, scale)

    return out.reshape(shapes_og)


def hadamard_transform_with_signs_and_norm(
    x: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_norms: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused norm+normalize+WHT: computes L2 norm, normalizes, then applies WHT.

    Equivalent to:
        norms = torch.linalg.norm(x, dim=-1)
        x_unit = x / norms.unsqueeze(-1).clamp(min=1e-10)
        y = hadamard_transform_with_signs(x_unit, signs1, signs2, scale)

    But fused into a single CUDA kernel launch.

    Args:
        x: (..., dim) tensor, any dtype. Will be cast to float32 internally.
        signs1: (dim,) float32 sign vector applied before Hadamard.
        signs2: (dim,) float32 sign vector applied after Hadamard.
        scale: scalar multiplier (typically 1/sqrt(dim)).
        out: optional pre-allocated (..., dim) output tensor for WHT result.
            If None, will be allocated internally.
        out_norms: optional pre-allocated (...,) float32 tensor for norms output.
            If None, will be allocated internally.

    Returns:
        (y, norms): y is WHT-rotated unit vector, norms is L2 norms.
    """
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform_with_signs_and_norm only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    batch_shape = shapes_og[:-1]

    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    batch = x.shape[0]

    if out is not None:
        out = out.reshape(-1, dim_og)
    else:
        # Output is always float32 (downstream pack kernel expects float32)
        out = torch.empty(batch, dim_og, dtype=torch.float32, device=x.device)

    if out_norms is not None:
        out_norms_flat = out_norms.reshape(-1)
    else:
        out_norms_flat = torch.empty(batch, dtype=torch.float32, device=x.device)

    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform_with_signs_and_norm(x, out, signs1, signs2, out_norms_flat, scale)

    return out.reshape(shapes_og), out_norms_flat.reshape(batch_shape)


def hadamard_transform_with_signs_and_norm_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_norms: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused norm+normalize+WHT for K and V without torch.cat.

    Accepts K and V as separate tensors, launching a single CUDA kernel
    that reads from dual input pointers. Eliminates the ~13µs memcpy
    overhead of concatenation.

    Args:
        k: (tokens, heads, dim) tensor — K input.
        v: (tokens, heads, dim) tensor — V input (same shape as k).
        signs1: (dim,) float32 sign vector applied before Hadamard.
        signs2: (dim,) float32 sign vector applied after Hadamard.
        scale: scalar multiplier (typically 1/sqrt(dim)).
        out: optional pre-allocated (2*tokens*heads, dim) float32 output.
        out_norms: optional pre-allocated (2*tokens*heads,) float32 norms.

    Returns:
        (y, norms): y is (2*tokens, heads, dim) float32 WHT result [K; V],
                    norms is (2*tokens, heads) float32 L2 norms [K; V].
    """
    if not k.is_cuda:
        raise RuntimeError("hadamard_transform_with_signs_and_norm_kv only supports CUDA tensors")

    tokens, heads, dim = k.shape
    total_batch = 2 * tokens * heads

    # Flatten K and V to (tokens*heads, dim) for the kernel
    k_flat = k.reshape(tokens * heads, dim)
    v_flat = v.reshape(tokens * heads, dim)
    if k_flat.stride(-1) != 1:
        k_flat = k_flat.contiguous()
    if v_flat.stride(-1) != 1:
        v_flat = v_flat.contiguous()

    if out is not None:
        out_flat = out.reshape(total_batch, dim)
    else:
        out_flat = torch.empty(total_batch, dim, dtype=torch.float32, device=k.device)

    if out_norms is not None:
        out_norms_flat = out_norms.reshape(-1)
    else:
        out_norms_flat = torch.empty(total_batch, dtype=torch.float32, device=k.device)

    module = _jit_hadamard_module(k.dtype)
    module.hadamard_transform_with_signs_and_norm_dual(
        k_flat, v_flat, out_flat, signs1, signs2, out_norms_flat, scale
    )

    return out_flat.reshape(2 * tokens, heads, dim), out_norms_flat.reshape(2 * tokens, heads)


def hadamard_transform_12n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 12, module.hadamard_transform_12n)


def hadamard_transform_20n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 20, module.hadamard_transform_20n)


def hadamard_transform_28n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 28, module.hadamard_transform_28n)


def hadamard_transform_40n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    module = _jit_hadamard_module(x.dtype)
    return _hadamard_transform_impl(x, scale, 4 * 40, module.hadamard_transform_40n)


def fused_norm_wht_pack_store_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    scale: float,
    boundaries: torch.Tensor,
    centroids: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_dscale: torch.Tensor,
    v_dscale: torch.Tensor,
    loc: torch.Tensor,
    bit_width: int = 4,
) -> None:
    """Fully fused KV quantize+store: 1 CUDA kernel, zero HBM intermediate.

    Combines norm+normalize+WHT+searchsorted+pack+scatter_store into a single
    CUDA kernel launch, eliminating the WHT→Pack HBM round-trip entirely.

    Args:
        k: (tokens, heads, dim) bf16/fp16 — K input.
        v: (tokens, heads, dim) bf16/fp16 — V input.
        signs1: (dim,) float32 sign vector applied before Hadamard.
        signs2: (dim,) float32 sign vector applied after Hadamard.
        scale: scalar multiplier (typically 1/sqrt(dim)).
        boundaries: (n_boundaries,) float32 — quantization boundaries.
        centroids: (n_centroids,) float32 — codebook centroids.
        k_buffer: (pool_size, heads, packed_dim) uint8 — K output pool.
        v_buffer: (pool_size, heads, packed_dim) uint8 — V output pool.
        k_dscale: (pool_size, heads) bf16 — K dscale output pool.
        v_dscale: (pool_size, heads) bf16 — V dscale output pool.
        loc: (tokens,) int64 — pool slot indices for scatter.
        bit_width: 4 or 2.
    """
    if not k.is_cuda:
        raise RuntimeError("fused_norm_wht_pack_store_kv only supports CUDA tensors")

    tokens, heads, dim = k.shape

    # Flatten K and V to (tokens*heads, dim) for the kernel
    k_flat = k.reshape(tokens * heads, dim)
    v_flat = v.reshape(tokens * heads, dim)
    if k_flat.stride(-1) != 1:
        k_flat = k_flat.contiguous()
    if v_flat.stride(-1) != 1:
        v_flat = v_flat.contiguous()

    module = _jit_hadamard_module(k.dtype)
    if bit_width == 4:
        module.fused_norm_wht_pack_store_4bit(
            k_flat, v_flat, signs1, signs2, scale,
            boundaries, centroids,
            k_buffer, v_buffer, k_dscale, v_dscale,
            loc, tokens, heads,
        )
    elif bit_width == 2:
        module.fused_norm_wht_pack_store_2bit(
            k_flat, v_flat, signs1, signs2, scale,
            boundaries, centroids,
            k_buffer, v_buffer, k_dscale, v_dscale,
            loc, tokens, heads,
        )
    else:
        raise ValueError(f"Unsupported bit_width: {bit_width}. Only 2 and 4 are supported.")
