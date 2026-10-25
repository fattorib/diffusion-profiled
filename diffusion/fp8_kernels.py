from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch.amp.autocast_mode import custom_fwd, custom_bwd

# seems like fwd and bwd can be done with e4m3 at this scale
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX_E4M3 = torch.finfo(torch.float8_e4m3fn).max


# TODO: might expand these over time if we do IN64x64
# (input channels, output channels, input height, input width)
SUPPORTED_UNET_SHAPES = (
    (128, 128, 32, 32),
    (128, 256, 32, 32),
    (256, 256, 16, 16),
    (512, 256, 16, 16),
    (256, 256, 8, 8),
    (512, 256, 8, 8),
    (512, 128, 32, 32),
    (256, 256, 32, 32),
    (256, 256, 4, 4),
)


def triton_config(
    block_m: int,
    block_n: int,
    block_k: int,
    *,
    num_warps: int,
    num_stages: int,
    split_k: int | None = None,
) -> triton.Config:
    kwargs = {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}
    if split_k is not None:
        kwargs["SPLIT_K"] = split_k
    return triton.Config(
        kwargs,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def triton_launch_kwargs(config: triton.Config) -> dict[str, int]:
    """Return launch metadata accepted by torch.compile's Triton capture."""
    return {key: value for key, value in config.all_kwargs().items() if key != "num_ctas"}


@triton.jit
def partial_amax_kernel(
    x,
    partials,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + offsets, mask=offsets < n_elements, other=0.0)
    partial_amax = tl.max(tl.abs(values.to(tl.float32)), axis=0)
    tl.store(partials + tl.program_id(0), partial_amax)


# this compute the partials for the current amax as well as scales the tensor
# using the historical amax
@triton.jit
def partial_amax_kernel_delayed(
    x,
    quantized,
    scale,
    partials,
    n_elements,
    FP8_MAX_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    quant_scale = tl.load(scale)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    partial_amax = tl.max(tl.abs(values.to(tl.float32)), axis=0)
    scaled = values / quant_scale
    scaled = tl.maximum(-FP8_MAX_VALUE, tl.minimum(scaled, FP8_MAX_VALUE))
    tl.store(quantized + offsets, scaled, mask=mask)
    tl.store(partials + tl.program_id(0), partial_amax)


@triton.jit
def finalize_amax_kernel(
    partials,
    scale,
    n_partials: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    partial_values = tl.load(partials + offsets, mask=offsets < n_partials, other=0.0)
    amax = tl.max(partial_values, axis=0)
    quant_scale = tl.where(amax > 0.0, amax / FP8_MAX_VALUE, 1.0)
    tl.store(scale, quant_scale)


@triton.jit
def quantize_fp8_kernel(
    x,
    quantized,
    scale,
    n_elements,
    FP8_MAX_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    quant_scale = tl.load(scale)
    scaled = values / quant_scale
    scaled = tl.maximum(-FP8_MAX_VALUE, tl.minimum(scaled, FP8_MAX_VALUE))
    tl.store(quantized + offsets, scaled, mask=mask)


DX_CONFIG_BY_SHAPE = {
    shape: triton_config(128, 64, 32, num_warps=4, num_stages=3) for shape in SUPPORTED_UNET_SHAPES
}
DX_CONFIG_BY_SHAPE[(512, 256, 16, 16)] = triton_config(256, 64, 32, num_warps=8, num_stages=3)

# NOTE: only for the splitK param
DW_CONFIG_BY_SHAPE = {
    (128, 128, 32, 32): 32,
    (128, 256, 32, 32): 16,
    (256, 256, 16, 16): 8,
    (512, 256, 16, 16): 4,
    (256, 256, 8, 8): 4,
    (256, 256, 4, 4): 1,
    (512, 256, 8, 8): 4,
    (512, 128, 32, 32): 16,
    (256, 256, 32, 32): 8,
}


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3
        ),
        triton.Config(
            kwargs={"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            kwargs={"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            kwargs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            kwargs={"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            kwargs={"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            kwargs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
    ],
    key=["IN_H", "IN_W", "CIN", "COUT"],
)
@triton.jit
def conv2d_fwd_fp8_kernel(
    x,
    w,
    y,
    x_scale,
    w_scale,
    sxn,
    sxc,
    sxh,
    sxw,
    swo,
    swi,
    swh,
    sww,
    syn,
    syc,
    syh,
    syw,
    BATCH: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one [M, N] output tile, reducing over [KH, KW, CIN].

    M is flattened [batch, output height, output width].

    """
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    oc = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    ow = m % IN_W
    nh = m // IN_W
    oh = nh % IN_H
    n = nh // IN_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # NOTE: hardcoded for KH = KW = 3
    KH: tl.constexpr = 3  # type: ignore
    KW: tl.constexpr = 3  # type: ignore

    ih = oh - 1

    for kh in range(KH):
        iw = ow - 1
        for kw in range(KW):
            x_ptrs = x + (n[:, None] * sxn + ih[:, None] * sxh + iw[:, None] * sxw)
            w_ptrs = w + (oc[None, :] * swo + kh * swh + kw * sww)

            x_mask_static = (
                (n[:, None] < BATCH)
                & (ih[:, None] >= 0)
                & (ih[:, None] < IN_H)
                & (iw[:, None] >= 0)
                & (iw[:, None] < IN_W)
            )

            for c0 in range(0, CIN, BLOCK_K):
                ic = c0 + tl.arange(0, BLOCK_K)

                x_mask = x_mask_static & (ic[None, :] < CIN)

                x_tile = tl.load(x_ptrs + ic[None, :], mask=x_mask, other=0.0)
                w_tile = tl.load(w_ptrs + ic[:, None])

                acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)
            iw += 1

        ih += 1

    y_ptrs = y + (n[:, None] * syn + oc[None, :] * syc + oh[:, None] * syh + ow[:, None] * syw)
    y_mask = (n[:, None] < BATCH) & (oc[None, :] < COUT)
    output_scale = tl.load(x_scale) * tl.load(w_scale)
    tl.store(y_ptrs, acc * output_scale, mask=y_mask)


@triton.jit
def conv2d_dx_fp8_kernel(
    dy,
    w,
    dx,
    dy_scale,
    w_scale,
    sdyn,
    sdyc,
    sdyh,
    sdyw,
    swo,
    swi,
    swh,
    sww,
    sdxn,
    sdxc,
    sdxh,
    sdxw,
    BATCH: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    ic = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

    iw = m % IN_W
    nh = m // IN_W
    ih = nh % IN_H
    n = nh // IN_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # NOTE: hardcoded for KH = KW = 3
    KH: tl.constexpr = 3  # type: ignore
    KW: tl.constexpr = 3  # type: ignore

    for kh in range(KH):
        oh = ih + 1 - kh
        for kw in range(KW):
            ow = iw + 1 - kw
            valid_spatial = (oh >= 0) & (oh < IN_H) & (ow >= 0) & (ow < IN_W)
            for c0 in range(0, COUT, BLOCK_K):
                oc = c0 + tl.arange(0, BLOCK_K)

                dy_ptrs = dy + (
                    n[:, None] * sdyn + oc[None, :] * sdyc + oh[:, None] * sdyh + ow[:, None] * sdyw
                )
                dy_mask = (n[:, None] < BATCH) & (oc[None, :] < COUT) & valid_spatial[:, None]
                dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

                w_ptrs = w + (oc[:, None] * swo + ic[None, :] * swi + kh * swh + kw * sww)

                w_mask = (oc[:, None] < COUT) & (ic[None, :] < CIN)
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(dy_tile, w_tile, out_dtype=tl.float32)

    dx_ptrs = dx + (
        n[:, None] * sdxn + ic[None, :] * sdxc + ih[:, None] * sdxh + iw[:, None] * sdxw
    )
    dx_mask = (n[:, None] < BATCH) & (ic[None, :] < CIN)
    output_scale = tl.load(dy_scale) * tl.load(w_scale)
    tl.store(dx_ptrs, acc * output_scale, mask=dx_mask)


@triton.jit
def transpose_fp8_2d_kernel(
    src,
    dst,
    ROWS,
    COLS,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r[:, None] < ROWS) & (c[None, :] < COLS)
    tile = tl.load(src + r[:, None] * COLS + c[None, :], mask=mask)
    mask_t = (c[:, None] < COLS) & (r[None, :] < ROWS)
    tl.store(dst + c[:, None] * ROWS + r[None, :], tl.trans(tile), mask=mask_t)


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=["IN_H", "IN_W", "CIN", "COUT", "SPLIT_K"],
)
@triton.jit
def conv2d_dw_fp8_kernel(
    x,
    dy_t,
    partials,
    BATCH: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    oc = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(2)

    ic = m % CIN
    kernel_pos = tl.program_id(1) * BLOCK_N // CIN

    # NOTE: hardcoded for KH = KW = 3
    KH: tl.constexpr = 3  # type: ignore
    KW: tl.constexpr = 3  # type: ignore

    kh = kernel_pos // KW
    kw = kernel_pos % KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_total = BATCH * IN_H * IN_W
    k_per_split = k_total // SPLIT_K
    k_start = split * k_per_split
    for k in range(0, k_per_split, BLOCK_K):
        r = k_start + k + tl.arange(0, BLOCK_K)
        ow = r % IN_W
        nh = r // IN_W
        oh = nh % IN_H

        ih = oh - 1 + kh
        iw = ow - 1 + kw

        dy_ptrs = dy_t + oc[:, None] * k_total + r[None, :]
        dy_tile = tl.load(dy_ptrs)

        tap_delta = ((kh - 1) * IN_W + (kw - 1)) * CIN
        x_ptrs = x + r[:, None] * CIN + tap_delta + ic[None, :]
        x_mask = (
            (ih[:, None] >= 0) & (ih[:, None] < IN_H) & (iw[:, None] >= 0) & (iw[:, None] < IN_W)
        )
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(dy_tile, x_tile, out_dtype=tl.float32)

    partial_ptrs = (
        partials + split * (COUT * CIN * KH * KW) + oc[:, None] * (CIN * KH * KW) + m[None, :]
    )
    partial_mask = (oc[:, None] < COUT) & (m[None, :] < CIN * KH * KW)
    tl.store(partial_ptrs, acc, mask=partial_mask)


@triton.jit
def conv2d_dw_reduce_fp8_kernel(
    partials,
    dw,
    x_scale,
    dy_scale,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    KH = 3
    KW = 3
    total = CIN * KH * KW * COUT
    mask = offsets < total

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(partials + split * total + offsets, mask=mask, other=0.0)

    output_scale = tl.load(x_scale) * tl.load(dy_scale)
    tl.store(dw + offsets, acc * output_scale, mask=mask)


def torch_checks_conv(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> None:
    assert a.ndim == b.ndim == 4
    assert a.is_cuda and b.is_cuda and a.device == b.device
    # assert a.dtype == b.dtype == dtype
    assert a.is_contiguous() or a.is_contiguous(memory_format=torch.channels_last)
    assert b.is_contiguous() or b.is_contiguous(memory_format=torch.channels_last)


def fwd_shapes(
    x: torch.Tensor,
    weight: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[int, int, int, int, int]:
    torch_checks_conv(x, weight, dtype)
    batch, cin, height, width = x.shape
    cout, weight_cin, kh, kw = weight.shape
    assert (weight_cin, kh, kw) == (cin, 3, 3)
    assert (cin, cout, height, width) in SUPPORTED_UNET_SHAPES, f"{(cin, cout, height, width)=}"
    return batch, cin, cout, height, width


def xgrad_shapes(
    dy: torch.Tensor,
    weight: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[int, int, int, int, int]:
    torch_checks_conv(dy, weight, dtype)
    batch, cout, height, width = dy.shape
    weight_cout, cin, kh, kw = weight.shape
    assert (weight_cout, kh, kw) == (cout, 3, 3)
    assert (cin, cout, height, width) in SUPPORTED_UNET_SHAPES
    return batch, cin, cout, height, width


def wgrad_shapes(
    x: torch.Tensor,
    dy: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[int, int, int, int, int]:
    torch_checks_conv(x, dy, dtype)
    batch, cin, height, width = x.shape
    dy_batch, cout, dy_height, dy_width = dy.shape
    assert (dy_batch, dy_height, dy_width) == (batch, height, width)
    assert (cin, cout, height, width) in SUPPORTED_UNET_SHAPES
    return batch, cin, cout, height, width


def quantize_fp8_delayed(
    tensor: torch.Tensor, prev_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorwise quantization from bfloat16 to fp8 with delay"""

    assert tensor.dtype == torch.bfloat16 and tensor.is_cuda and tensor.ndim == 4, (
        f"{tensor.dtype} {tensor.is_cuda} {tensor.ndim}"
    )

    n_elements = tensor.numel()
    n_partials = (n_elements + 8192 - 1) // 8192
    partials = torch.empty(n_partials, device=tensor.device, dtype=torch.float32)
    scale = torch.empty((), device=tensor.device, dtype=torch.float32)

    dtype_torch = torch.float8_e4m3fn
    fp8_max = FP8_MAX_E4M3
    quantized = torch.empty_like(tensor, dtype=dtype_torch, memory_format=torch.preserve_format)

    partial_amax_kernel_delayed[(n_partials,)](
        tensor,
        quantized,
        prev_scale,
        partials,
        n_elements,
        FP8_MAX_VALUE=fp8_max,  # type: ignore
        BLOCK_SIZE=8192,  # type: ignore
        num_warps=8,  # type: ignore
    )
    reduction_block = int(triton.next_power_of_2(n_partials))
    finalize_amax_kernel[(1,)](
        partials,
        scale,
        n_partials=n_partials,  # type: ignore
        FP8_MAX_VALUE=fp8_max,  # type: ignore
        BLOCK_SIZE=reduction_block,  # type: ignore
        num_warps=8 if reduction_block >= 2048 else 4,  # type: ignore
    )

    return quantized, scale


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert tensor.dtype == torch.bfloat16 and tensor.is_cuda and tensor.ndim == 4, (
        f"{tensor.dtype} {tensor.is_cuda} {tensor.ndim}"
    )

    n_elements = tensor.numel()
    n_partials = (n_elements + 8192 - 1) // 8192
    partials = torch.empty(n_partials, device=tensor.device, dtype=torch.float32)
    scale = torch.empty((), device=tensor.device, dtype=torch.float32)

    dtype_torch = torch.float8_e4m3fn
    fp8_max = FP8_MAX_E4M3
    quantized = torch.empty_like(tensor, dtype=dtype_torch, memory_format=torch.preserve_format)

    partial_amax_kernel[(n_partials,)](
        tensor,
        partials,
        n_elements,
        BLOCK_SIZE=8192,  # type: ignore
        num_warps=8,  # type: ignore
    )
    reduction_block = int(triton.next_power_of_2(n_partials))
    finalize_amax_kernel[(1,)](
        partials,
        scale,
        n_partials=n_partials,  # type: ignore
        FP8_MAX_VALUE=fp8_max,  # type: ignore
        BLOCK_SIZE=reduction_block,  # type: ignore
        num_warps=8 if reduction_block >= 2048 else 4,  # type: ignore
    )

    quantize_grid = (n_partials,)
    quantize_fp8_kernel[quantize_grid](
        tensor,
        quantized,
        scale,
        n_elements,
        FP8_MAX_VALUE=fp8_max,  # type: ignore
        BLOCK_SIZE=8192,  # type: ignore
        num_warps=8,  # type: ignore
    )

    return quantized, scale


def conv2d_fwd_fp8_prequantized(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    batch, cin, cout, height, width = fwd_shapes(x, weight, FP8_DTYPE)
    torch_checks_scale(x_scale, x.device)
    torch_checks_scale(weight_scale, x.device)
    y = torch.empty(
        (batch, cout, height, width),
        device=x.device,
        dtype=torch.bfloat16,
        memory_format=torch.channels_last,
    )

    grid = lambda META: ( # noqa: E731
        triton.cdiv(batch * height * width, META["BLOCK_M"]),
        triton.cdiv(cout, META["BLOCK_N"]),
    )
    conv2d_fwd_fp8_kernel[grid](
        x,
        weight,
        y,
        x_scale,
        weight_scale,
        *x.stride(),
        *weight.stride(),
        *y.stride(),
        BATCH=batch,  # type: ignore
        IN_H=height,  # type: ignore
        IN_W=width,  # type: ignore
        CIN=cin,  # type: ignore
        COUT=cout,  # type: ignore
    )
    return y


def conv2d_fwd_fp8(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_fp8, x_scale = quantize_fp8(x)
    weight_fp8, weight_scale = quantize_fp8(weight)
    return conv2d_fwd_fp8_prequantized(
        x_fp8,
        weight_fp8,
        x_scale,
        weight_scale,
    )


def conv2d_fwd_fp8_delayed(
    x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor
):
    x_fp8, _ = quantize_fp8_delayed(x, x_scale)
    weight_fp8, _ = quantize_fp8_delayed(weight, w_scale)

    return conv2d_fwd_fp8_prequantized(
        x_fp8,
        weight_fp8,
        x_scale,
        w_scale,
    )


def torch_checks_scale(scale: torch.Tensor, device: torch.device) -> None:
    assert scale.device == device and scale.dtype == torch.float32 and scale.numel() == 1


def conv2d_bwd_input_fp8_prequantized(
    dy: torch.Tensor,
    weight: torch.Tensor,
    dy_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    batch, cin, cout, height, width = xgrad_shapes(dy, weight, FP8_DTYPE)
    torch_checks_scale(dy_scale, dy.device)
    torch_checks_scale(weight_scale, dy.device)
    config = DX_CONFIG_BY_SHAPE[(cin, cout, height, width)]
    dx = torch.empty(
        (batch, cin, height, width),
        device=dy.device,
        dtype=torch.bfloat16,
        memory_format=torch.channels_last,
    )

    # NOTE: best ordering for L2 hit rate
    grid = (
        triton.cdiv(cin, config.kwargs["BLOCK_N"]),
        triton.cdiv(batch * height * width, config.kwargs["BLOCK_M"]),
    )
    conv2d_dx_fp8_kernel[grid](
        dy,
        weight,
        dx,
        dy_scale,
        weight_scale,
        *dy.stride(),
        *weight.stride(),
        *dx.stride(),
        BATCH=batch,  # type: ignore
        IN_H=height,  # type: ignore
        IN_W=width,  # type: ignore
        CIN=cin,  # type: ignore
        COUT=cout,  # type: ignore
        **triton_launch_kwargs(config),  # type: ignore
    )
    return dx


def conv2d_bwd_input_fp8(dy: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dy_fp8, dy_scale = quantize_fp8(dy)
    weight_fp8, weight_scale = quantize_fp8(weight)
    return conv2d_bwd_input_fp8_prequantized(
        dy_fp8,
        weight_fp8,
        dy_scale,
        weight_scale,
    )


def conv2d_bwd_weight_fp8_prequantized(
    x: torch.Tensor,
    dy: torch.Tensor,
    x_scale: torch.Tensor,
    dy_scale: torch.Tensor,
    split_k: int | None = None,
) -> torch.Tensor:
    batch, cin, cout, height, width = wgrad_shapes(x, dy, FP8_DTYPE)

    torch_checks_scale(x_scale, x.device)
    torch_checks_scale(dy_scale, x.device)

    if split_k is None:
        split_k = DW_CONFIG_BY_SHAPE[(cin, cout, height, width)]

    partials = torch.empty(
        (split_k, cout, cin * 3 * 3),
        device=x.device,
        dtype=torch.float32,
    )

    dw = torch.empty(
        (cout, cin, 3, 3),
        device=x.device,
        dtype=torch.bfloat16,
        memory_format=torch.channels_last,
    )

    x_channels_last = x.is_contiguous(memory_format=torch.channels_last)
    dy_channels_last = dy.is_contiguous(memory_format=torch.channels_last)

    if x_channels_last and dy_channels_last:
        # triton fails to properly generate the correct loads for Column-major dY
        # significantly faster to just pre-transpose dY
        spatial = batch * height * width
        dy_t = torch.empty((cout, spatial), device=dy.device, dtype=dy.dtype)
        transpose_grid = (
            triton.cdiv(spatial, 64),
            triton.cdiv(cout, 64),
        )
        transpose_fp8_2d_kernel[transpose_grid](
            dy,
            dy_t,
            spatial,
            cout,
            BLOCK_R=64,  # type: ignore
            BLOCK_C=64,  # type: ignore
            num_warps=4,  # type: ignore
        )

        def grid(meta):
            return (
                triton.cdiv(cout, meta["BLOCK_M"]),
                triton.cdiv(cin * 3 * 3, meta["BLOCK_N"]),
                split_k,
            )

        conv2d_dw_fp8_kernel[grid](
            x,
            dy_t,
            partials,
            BATCH=batch,
            IN_H=height,
            IN_W=width,
            CIN=cin,
            COUT=cout,
            SPLIT_K=split_k,
        )
    else:
        raise NotImplementedError

    reduce_grid = (triton.cdiv(cin * 3 * 3 * cout, 256),)

    conv2d_dw_reduce_fp8_kernel[reduce_grid](
        partials,
        dw,
        x_scale,
        dy_scale,
        CIN=cin,  # type: ignore
        COUT=cout,  # type: ignore
        SPLIT_K=split_k,  # type: ignore
        BLOCK_SIZE=256,  # type: ignore
        num_warps=4,  # type: ignore
    )

    return dw


def conv2d_bwd_weight_fp8(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    x_fp8, x_scale = quantize_fp8(x)
    dy_fp8, dy_scale = quantize_fp8(dy)
    return conv2d_bwd_weight_fp8_prequantized(
        x_fp8,
        dy_fp8,
        x_scale,
        dy_scale,
    )


# NOTE: helpful ref: https://github.com/meta-pytorch/float8_experimental/blob/main/float8_experimental/float8_linear.py
class Conv2dFP8Specialized(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        hist_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
        dy_ptrs: tuple[torch.Tensor, torch.Tensor] | None = None,
        do_delayed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # .to(torch.bfloat16) is much faster than custom_fwd(device_type="cuda", cast_inputs = torch.bfloat16)
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)

        assert x.is_contiguous(memory_format=torch.channels_last)

        ctx.do_delayed = do_delayed
        if do_delayed:
            assert hist_scales is not None
            assert dy_ptrs is not None
            x_scale, weight_scale = hist_scales
            x_fp8, x_scale_curr = quantize_fp8_delayed(x, x_scale)
            weight_fp8, weight_scale_curr = quantize_fp8_delayed(weight, weight_scale)

            dy_amax_window, window_pos = dy_ptrs
            dy_scale_hist = torch.max(dy_amax_window)

            ctx.dy_window = dy_amax_window
            ctx.window_pos = window_pos
            ctx.save_for_backward(x_fp8, weight_fp8, x_scale, weight_scale, dy_scale_hist)

        else:
            x_fp8, x_scale = quantize_fp8(x)
            weight_fp8, weight_scale = quantize_fp8(weight)

            ctx.save_for_backward(x_fp8, weight_fp8, x_scale, weight_scale)

        out = conv2d_fwd_fp8_prequantized(
            x_fp8,
            weight_fp8,
            x_scale,
            weight_scale,
        )

        if do_delayed:
            ctx.mark_non_differentiable(x_scale_curr, weight_scale_curr)  # type: ignore
            return out, x_scale_curr, weight_scale_curr
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        dy = cast(torch.Tensor, grad_outputs[0])

        # NOTE: this copy is likely not needed, however
        # 1. not clear whether torch.compile actually keeps this in the graph
        # 2. changing formats for read / write makes quantize_fp8 significantly more confusing
        dy = dy.contiguous(memory_format=torch.channels_last)

        if ctx.do_delayed:
            x_fp8, weight_fp8, x_scale, weight_scale, dy_scale = ctx.saved_tensors
            dy_fp8, dy_scale_curr = quantize_fp8_delayed(dy, dy_scale)

            # update the dY scale
            window_idx = ctx.window_pos.unsqueeze(0)
            ctx.dy_window.scatter_(0, window_idx, dy_scale_curr.unsqueeze(0))

        else:
            x_fp8, weight_fp8, x_scale, weight_scale = ctx.saved_tensors
            dy_fp8, dy_scale = quantize_fp8(dy)

        assert dy_fp8.is_contiguous(memory_format=torch.channels_last)
        assert weight_fp8.is_contiguous(memory_format=torch.channels_last)
        assert x_fp8.is_contiguous(memory_format=torch.channels_last)

        dx = conv2d_bwd_input_fp8_prequantized(
            dy_fp8,
            weight_fp8,
            dy_scale,
            weight_scale,
        )

        dw = conv2d_bwd_weight_fp8_prequantized(
            x_fp8,
            dy_fp8,
            x_scale,
            dy_scale,
        )

        return dx, dw, None, None, None


def triton_conv2d_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    hist_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
    dy_ptrs: tuple[torch.Tensor, torch.Tensor] | None = None,
    do_delayed: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return Conv2dFP8Specialized.apply(x, weight, hist_scales, dy_ptrs, do_delayed)
