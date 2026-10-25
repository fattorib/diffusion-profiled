"""
Analytic arithmetic-intensity model for the UNet in unet.py.

Counts, for a batched forward AND backward pass:
  - FLOPs  (fwd / bwd)
  - bytes read+written (fwd / bwd)

This reports *algorithmic* arithmetic intensity: logical tensor bytes at
operator boundaries, not measured DRAM traffic.  Cache reuse, compiler fusion,
workspace traffic, autocast conversions, allocator traffic, and the optimizer
are outside the model.  With those boundaries fixed, the counts are
deterministic and auditable.

Counting rules:
  * MAC = 2 FLOPs.
  * GEMM-like ops (conv, linear):
      fwd  flops = 2*B*out_positions*Cin*Cout*K^2
      fwd  bytes = act_in + act_out + weight
      wgrad flops/bytes ~= fwd
      dgrad flops/bytes ~= fwd when the input requires a gradient
      (wgrad and dgrad are counted as separate kernels)
  * SDPA attention (flash-style, no N^2 materialization):
      fwd  flops = 4*B*L^2*C (QK^T + AV) + ~4*B*nh*L^2 (softmax)
      bwd  recomputes QK^T/softmax, then computes dV, dP, dQ, dK
      fwd  bytes = read Q,K,V + write O = 4*B*L*C*elt
      bwd  bytes = 2x fwd (read Q,K,V,O,dO; write dQ,dK,dV)
  * Norm and transcendental FLOPs use the explicit per-element estimates in
    the helpers below.  Norm statistics are modeled as fp32.
  * Dropout includes its uint8 mask and is enabled when train.py enables it.
  * Resampling (nearest up/down), concat: 0 FLOPs, pure data movement.

Config below matches train.py (32M-param CIFAR-10 model). Edit and rerun.
No torch/GPU needed unless VERIFY=True (executed-shape cross-check on CPU).
"""

# pyrefly: ignore-errors
from dataclasses import dataclass

# ----------------------------- config (edit me) -----------------------------
BATCH_SIZE = 128
BASE_HIDDEN = 128  # model dimension (base channels)
SCALES = [1, 2, 2, 2]
N_HEADS = 4
ATTN_RES = [16, 8]
RESOLUTION = 32
IN_CHANNELS = 3
N_RESBLOCKS = 2
PRED_VARIANCE = False
ELT_SIZE = 2  # bytes per element: fp32=4, bf16/fp16=2
NORM_STAT_SIZE = 4  # mean/rstd are accumulated and stored in fp32
P_DROP = 0.2  # must match train.py

VERIFY = False  # cross-check analytic shapes against the executed UNet
VERIFY_BATCH = 1  # shape verification is linear in batch; one is enough
# -----------------------------------------------------------------------------


@dataclass
class Counter:
    flops_fwd: float = 0.0
    flops_bwd: float = 0.0
    bytes_fwd: float = 0.0  # read + written
    bytes_bwd: float = 0.0
    gemm_flops_fwd: float = 0.0
    sdpa_flops_fwd: float = 0.0
    gemm_calls: int = 0
    sdpa_calls: int = 0

    def __iadd__(self, other: "Counter") -> "Counter":
        self.flops_fwd += other.flops_fwd
        self.flops_bwd += other.flops_bwd
        self.bytes_fwd += other.bytes_fwd
        self.bytes_bwd += other.bytes_bwd
        self.gemm_flops_fwd += other.gemm_flops_fwd
        self.sdpa_flops_fwd += other.sdpa_flops_fwd
        self.gemm_calls += other.gemm_calls
        self.sdpa_calls += other.sdpa_calls
        return self


# ---------------------------------------------------------------------------
# op-level counters
# ---------------------------------------------------------------------------


def gemm_like(
    flops_fwd: float,
    act_in: float,
    act_out: float,
    weight_elems: float,
    elt: int,
    *,
    needs_input_grad: bool = True,
) -> Counter:
    """Count a convolution/linear and its separate wgrad/dgrad kernels.

    A wgrad reads ``x`` and ``dy`` and writes ``dw``.  A dgrad reads ``dy``
    and ``w`` and writes ``dx``.  Each therefore moves the same number of
    logical elements as the forward operation.  The first image convolution
    and first timestep linear do not need dgrad because their inputs do not
    require gradients during training.
    """
    bytes_fwd = (act_in + act_out + weight_elems) * elt
    backward_passes = 1 + int(needs_input_grad)  # wgrad, plus optional dgrad
    return Counter(
        flops_fwd=flops_fwd,
        flops_bwd=backward_passes * flops_fwd,
        bytes_fwd=bytes_fwd,
        bytes_bwd=backward_passes * bytes_fwd,
        gemm_flops_fwd=flops_fwd,
        gemm_calls=1,
    )


def conv2d(
    B: int,
    Cin: int,
    Cout: int,
    H: int,
    W: int,
    K: int,
    elt: int,
    *,
    needs_input_grad: bool = True,
) -> Counter:
    flops = 2.0 * B * H * W * Cin * Cout * K * K
    return gemm_like(
        flops,
        B * Cin * H * W,
        B * Cout * H * W,
        Cin * Cout * K * K,
        elt,
        needs_input_grad=needs_input_grad,
    )


def linear(
    B: int,
    L: int,
    Cin: int,
    Cout: int,
    elt: int,
    *,
    needs_input_grad: bool = True,
) -> Counter:
    flops = 2.0 * B * L * Cin * Cout
    return gemm_like(
        flops,
        B * L * Cin,
        B * L * Cout,
        Cin * Cout,
        elt,
        needs_input_grad=needs_input_grad,
    )


def silu(numel: float, elt: int) -> Counter:
    """Fused SiLU; exp/reciprocal each count as one FLOP by convention."""
    return Counter(
        flops_fwd=4 * numel,
        flops_bwd=8 * numel,
        bytes_fwd=2 * numel * elt,  # read x, write y
        bytes_bwd=3 * numel * elt,  # read x/dy, write dx
    )


def norm(
    numel: float,
    param_elems: float,
    stat_elems: float,
    elt: int,
    stat_elt: int,
) -> Counter:
    """Affine GroupNorm/LayerNorm with saved mean and reciprocal stddev.

    Seven forward and ten backward FLOPs per activation are conventional
    estimates for reductions, normalization, affine transform, and parameter
    gradients.  Bytes enumerate the logical tensors consumed and produced by
    a fused normalization kernel.
    """
    return Counter(
        flops_fwd=7 * numel,
        flops_bwd=10 * numel,
        bytes_fwd=(2 * numel + 2 * param_elems) * elt + 2 * stat_elems * stat_elt,
        bytes_bwd=(3 * numel + 3 * param_elems) * elt + 2 * stat_elems * stat_elt,
    )


def modulation(numel: float, broadcast_elems: float, elt: int) -> Counter:
    """Count ``(w + 1) * x + b`` including broadcast-gradient reductions."""
    return Counter(
        flops_fwd=3 * numel,
        flops_bwd=4 * numel,
        bytes_fwd=(4 * numel + 4 * broadcast_elems) * elt,
        bytes_bwd=(5 * numel + 3 * broadcast_elems) * elt,
    )


def residual_add_scale(numel: float, elt: int) -> Counter:
    """Count eager ``(a + b) / sqrt(2)`` and its backward pass."""
    return Counter(
        flops_fwd=2 * numel,
        flops_bwd=numel,
        bytes_fwd=5 * numel * elt,  # add kernel, then scale kernel
        bytes_bwd=5 * numel * elt,
    )


def dropout(numel: float, p_drop: float, elt: int) -> Counter:
    """Training dropout, including a one-byte saved mask; RNG FLOPs excluded."""
    if p_drop == 0.0:
        return Counter()
    mask_bytes = numel
    return Counter(
        flops_fwd=numel,
        flops_bwd=numel,
        bytes_fwd=2 * numel * elt + mask_bytes,
        bytes_bwd=2 * numel * elt + mask_bytes,
    )


def sdpa(B: int, nh: int, L: int, C: int, elt: int) -> Counter:
    hd = C // nh
    scores = B * nh * L * L
    matmul_fwd = 4.0 * B * nh * L * L * hd  # QK^T and AV
    softmax_fwd = 4.0 * scores
    flops_fwd = matmul_fwd + softmax_fwd

    # Flash-style backward recomputes QK^T/softmax because probabilities were
    # not materialized, then performs dV, dP, dQ, and dK.  The GEMM work is
    # therefore 2.5x forward; softmax recomputation + backward is ~2x forward.
    flops_bwd = 2.5 * matmul_fwd + 2.0 * softmax_fwd
    stats_bytes = B * nh * L * NORM_STAT_SIZE  # saved log-sum-exp
    bytes_fwd = 4 * B * L * C * elt + stats_bytes
    bytes_bwd = 8 * B * L * C * elt + stats_bytes
    return Counter(
        flops_fwd,
        flops_bwd,
        bytes_fwd,
        bytes_bwd,
        sdpa_flops_fwd=flops_fwd,
        sdpa_calls=1,
    )


def movement(
    numel_read_fwd: float,
    numel_write_fwd: float,
    numel_read_bwd: float,
    numel_write_bwd: float,
    elt: int,
) -> Counter:
    """Pure logical tensor movement."""
    return Counter(
        bytes_fwd=(numel_read_fwd + numel_write_fwd) * elt,
        bytes_bwd=(numel_read_bwd + numel_write_bwd) * elt,
    )


def resample_nearest(numel_in: float, numel_out: float, elt: int) -> Counter:
    # Downsampling reads only selected source elements.  Upsampling logically
    # reads every source element once; cache behavior is deliberately ignored.
    source_read = min(numel_in, numel_out)
    return movement(source_read, numel_out, numel_out, numel_in, elt)


def concat(numel: float, elt: int) -> Counter:
    return movement(numel, numel, numel, numel, elt)


def contiguous(numel: float, elt: int) -> Counter:
    return movement(numel, numel, numel, numel, elt)


# ---------------------------------------------------------------------------
# module-level counters (mirror unet.py)
# ---------------------------------------------------------------------------


def residual_block(
    B: int, Cin: int, Cout: int, res: int, C_base: int, p_drop: float, elt: int
) -> Counter:
    c = Counter()
    n_in = B * Cin * res * res
    n_out = B * Cout * res * res

    c += norm(n_in, Cin, B * 32, elt, NORM_STAT_SIZE)  # GroupNorm 1
    c += silu(n_in, elt)
    c += conv2d(B, Cin, Cout, res, res, 3, elt)  # conv_1
    # time embed -> (w, b); tiny per-sample linear
    c += linear(B, 1, 4 * C_base, 2 * Cout, elt)
    c += norm(n_out, Cout, B * 32, elt, NORM_STAT_SIZE)  # GroupNorm 2
    c += modulation(n_out, B * Cout, elt)  # (w+1)*x + b
    c += silu(n_out, elt)
    c += dropout(n_out, p_drop, elt)
    c += conv2d(B, Cout, Cout, res, res, 3, elt)  # conv_2
    if Cin != Cout:  # skip connection
        c += conv2d(B, Cin, Cout, res, res, 1, elt)
    c += residual_add_scale(n_out, elt)
    return c


def attention(B: int, C: int, res: int, nh: int, elt: int) -> Counter:
    c = Counter()
    L = res * res
    n = B * L * C
    # b c h w -> b (h w) c is a view, so it moves no data by itself.
    c += norm(n, C, B * L, elt, NORM_STAT_SIZE)  # LayerNorm
    c += linear(B, L, C, 3 * C, elt)  # qkv
    c += sdpa(B, nh, L, C, elt)  # SDPA
    c += linear(B, L, C, C, elt)  # out proj
    c += residual_add_scale(n, elt)  # add, then final scale
    c += contiguous(n, elt)  # explicit .contiguous()
    return c


def stage(
    B: int,
    in_ch: int,
    hid_ch: int,
    out_ch: int,
    res: int,
    num_blocks: int,
    do_attn: bool,
    nh: int,
    C_base: int,
    p_drop: float,
    elt: int,
) -> Counter:
    c = Counter()
    # first block: in -> hid
    c += residual_block(B, in_ch, hid_ch, res, C_base, p_drop, elt)
    if do_attn:
        c += attention(B, hid_ch, res, nh, elt)
    # middle blocks: hid -> hid
    for _ in range(num_blocks - 2):
        c += residual_block(B, hid_ch, hid_ch, res, C_base, p_drop, elt)
        if do_attn:
            c += attention(B, hid_ch, res, nh, elt)
    # last block: hid -> out
    c += residual_block(B, hid_ch, out_ch, res, C_base, p_drop, elt)
    if do_attn:
        c += attention(B, out_ch, res, nh, elt)
    return c


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------


def unet_counts(
    B: int,
    in_ch: int,
    C: int,
    scales: list[int],
    nh: int,
    attn_res: list[int],
    res0: int,
    n_resblocks: int,
    p_drop: float,
    elt: int,
) -> dict[str, Counter]:
    parts: dict[str, Counter] = {}

    # time embed MLP: [B, C] -> [B, 4C] -> [B, 4C]
    te = Counter()
    te += linear(B, 1, C, 4 * C, elt, needs_input_grad=False)
    te += silu(B * 4 * C, elt)
    te += linear(B, 1, 4 * C, 4 * C, elt)
    te += silu(B * 4 * C, elt)
    parts["time_embed"] = te

    parts["conv_in"] = conv2d(
        B,
        in_ch,
        scales[0] * C,
        res0,
        res0,
        3,
        elt,
        needs_input_grad=False,
    )

    n_scales = len(scales)
    # down stages: stage i at spatial res0 >> i, channels scales[i]*C -> scales[i+1]*C
    for i in range(n_scales - 1):
        res = res0 >> i
        cc, cn = scales[i] * C, scales[i + 1] * C
        c = Counter()
        if i > 0:  # downsample before stage (nearest): read full, write quarter
            c += resample_nearest(B * cc * (2 * res) ** 2, B * cc * res * res, elt)
        c += stage(B, cc, cc, cn, res, n_resblocks, res in attn_res, nh, C, p_drop, elt)
        parts[f"down_stage_{i} (res={res})"] = c

    res_mid = res0 >> (n_scales - 1)
    mid_ch = scales[-1] * C

    # unet.py has both a final non-attending down stage and a separate,
    # self-attending middle stage at the same 4x4 resolution.
    bottom_down = Counter()
    bottom_down += resample_nearest(
        B * mid_ch * (2 * res_mid) ** 2,
        B * mid_ch * res_mid * res_mid,
        elt,
    )
    bottom_down += stage(B, mid_ch, mid_ch, mid_ch, res_mid, n_resblocks, False, nh, C, p_drop, elt)
    parts[f"bottom_down (res={res_mid})"] = bottom_down

    parts[f"mid (res={res_mid})"] = stage(
        B, mid_ch, mid_ch, mid_ch, res_mid, n_resblocks, True, nh, C, p_drop, elt
    )

    # up stages, in execution order: mid-level first, then reversed loop order
    up_specs = []  # (res, in_ch, hid_ch, out_ch, do_attn, upsample_before, skip_ch)
    up_specs.append((res_mid, 2 * mid_ch, mid_ch, mid_ch, False, False, mid_ch))
    for i in reversed(range(n_scales - 1)):
        res = res0 >> i
        cc, cn = scales[i] * C, scales[i + 1] * C
        up_specs.append((res, 2 * cn, cc, cc, res in attn_res, True, cn))

    for j, (res, ic, hc, oc, da, up, skip_ch) in enumerate(up_specs):
        c = Counter()
        if up:  # nearest 2x upsample of previous output
            prev = up_specs[j - 1][3]
            c += resample_nearest(
                B * prev * (res // 2) ** 2,
                B * prev * res * res,
                elt,
            )
        # concat with skip: read both, write combined
        prev_ch = up_specs[j - 1][3] if j > 0 else mid_ch
        c += concat(B * (prev_ch + skip_ch) * res * res, elt)
        c += stage(B, ic, hc, oc, res, n_resblocks + 1, da, nh, C, p_drop, elt)
        parts[f"up_stage_{j} (res={res})"] = c

    # output head at full res
    ch1 = scales[0] * C
    out = Counter()
    n = B * ch1 * res0 * res0
    out += norm(n, ch1, B * 32, elt, NORM_STAT_SIZE)
    out += silu(n, elt)
    out += conv2d(B, ch1, in_ch * (1), res0, res0, 3, elt)
    parts["out_head"] = out

    return parts


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def fmt_flops(x: float) -> str:
    return f"{x / 1e9:8.2f} G"


def fmt_bytes(x: float) -> str:
    return f"{x / 1e9:8.3f} GB" if x >= 1e9 else f"{x / 1e6:8.2f} MB"


def report(parts: dict[str, Counter], B: int, elt: int) -> Counter:
    total = Counter()
    for c in parts.values():
        total += c

    hdr = (
        f"{'component':<26} {'flops_fwd':>10} {'flops_bwd':>10} "
        f"{'bytes_fwd':>10} {'bytes_bwd':>10} {'AI_fwd':>7} {'AI_bwd':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, c in parts.items():
        ai_f = c.flops_fwd / c.bytes_fwd if c.bytes_fwd else 0
        ai_b = c.flops_bwd / c.bytes_bwd if c.bytes_bwd else 0
        print(
            f"{name:<26} {fmt_flops(c.flops_fwd)} {fmt_flops(c.flops_bwd)} "
            f"{fmt_bytes(c.bytes_fwd)} {fmt_bytes(c.bytes_bwd)} "
            f"{ai_f:7.1f} {ai_b:7.1f}"
        )
    print("-" * len(hdr))
    ai_f = total.flops_fwd / total.bytes_fwd
    ai_b = total.flops_bwd / total.bytes_bwd
    ai_t = (total.flops_fwd + total.flops_bwd) / (total.bytes_fwd + total.bytes_bwd)
    print(
        f"{'TOTAL':<26} {fmt_flops(total.flops_fwd)} {fmt_flops(total.flops_bwd)} "
        f"{fmt_bytes(total.bytes_fwd)} {fmt_bytes(total.bytes_bwd)} "
        f"{ai_f:7.1f} {ai_b:7.1f}"
    )
    print()
    print(f"batch={B}, elt_size={elt}B")
    print(
        f"fwd: {total.flops_fwd / 1e9:.2f} GFLOP, {total.bytes_fwd / 1e9:.3f} GB "
        f"-> AI = {ai_f:.1f} FLOP/byte"
    )
    print(
        f"bwd: {total.flops_bwd / 1e9:.2f} GFLOP, {total.bytes_bwd / 1e9:.3f} GB "
        f"-> AI = {ai_b:.1f} FLOP/byte"
    )
    print(
        f"fwd+bwd: {(total.flops_fwd + total.flops_bwd) / 1e9:.2f} GFLOP, "
        f"{(total.bytes_fwd + total.bytes_bwd) / 1e9:.3f} GB "
        f"-> AI = {ai_t:.1f} FLOP/byte"
    )
    return total


def verify(B: int) -> None:
    """Cross-check GEMM/SDPA topology and shapes against an executed UNet.

    torch.profiler does not annotate SDPA or convolution backward FLOPs, so it
    cannot verify this model.  Forward hooks give an independent source of the
    actual module call count and tensor shapes instead.
    """
    import math

    import torch
    from diffusion.unet import CHWAttention, UNet, UNetConfig

    torch.manual_seed(0)
    model = UNet(
        UNetConfig(
            IN_CHANNELS,
            base_hidden=BASE_HIDDEN,
            scales=SCALES,
            n_heads=N_HEADS,
            p_drop=P_DROP,
            attn_resolutions=ATTN_RES,
            in_resolution=RESOLUTION,
            n_resblocks=N_RESBLOCKS,
        )
    ).eval()
    x = torch.randn(B, IN_CHANNELS, RESOLUTION, RESOLUTION)
    t = torch.randint(0, 1000, (B,))

    observed_gemm_flops = 0
    observed_gemm_calls = 0
    observed_sdpa_flops = 0
    observed_sdpa_calls = 0

    def gemm_hook(module, _inputs, output) -> None:
        nonlocal observed_gemm_flops, observed_gemm_calls
        observed_gemm_calls += 1
        if isinstance(module, torch.nn.Conv2d):
            kh, kw = module.kernel_size
            observed_gemm_flops += (
                2 * output.numel() * (module.in_channels // module.groups) * kh * kw
            )
        elif isinstance(module, torch.nn.Linear):
            observed_gemm_flops += 2 * output.numel() * module.in_features
        else:  # pragma: no cover - hook registration controls this
            raise TypeError(type(module))

    def attention_hook(module, inputs, _output) -> None:
        nonlocal observed_sdpa_flops, observed_sdpa_calls
        assert isinstance(module, CHWAttention)
        batch, channels, height, width = inputs[0].shape
        length = height * width
        observed_sdpa_calls += 1
        observed_sdpa_flops += (
            4 * batch * length * length * channels + 4 * batch * module.nh * length * length
        )

    handles = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(gemm_hook))
        elif isinstance(module, CHWAttention):
            handles.append(module.register_forward_hook(attention_hook))

    try:
        with torch.no_grad():
            model(x, t)
    finally:
        for handle in handles:
            handle.remove()

    expected = Counter()
    for counter in unet_counts(
        B=B,
        in_ch=IN_CHANNELS,
        C=BASE_HIDDEN,
        scales=SCALES,
        nh=N_HEADS,
        attn_res=ATTN_RES,
        res0=RESOLUTION,
        n_resblocks=N_RESBLOCKS,
        p_drop=P_DROP,
        elt=ELT_SIZE,
    ).values():
        expected += counter

    assert observed_gemm_calls == expected.gemm_calls, (
        observed_gemm_calls,
        expected.gemm_calls,
    )
    assert observed_sdpa_calls == expected.sdpa_calls, (
        observed_sdpa_calls,
        expected.sdpa_calls,
    )
    assert math.isclose(observed_gemm_flops, expected.gemm_flops_fwd, abs_tol=0.5)
    assert math.isclose(observed_sdpa_flops, expected.sdpa_flops_fwd, abs_tol=0.5)

    print(f"\n[executed-shape cross-check, batch={B}]")
    print(f"GEMM-like: {observed_gemm_calls} calls, {observed_gemm_flops / 1e9:.6f} GFLOP [ok]")
    print(f"SDPA:      {observed_sdpa_calls} calls, {observed_sdpa_flops / 1e9:.6f} GFLOP [ok]")


def main() -> None:
    parts = unet_counts(
        B=BATCH_SIZE,
        in_ch=IN_CHANNELS,
        C=BASE_HIDDEN,
        scales=SCALES,
        nh=N_HEADS,
        attn_res=ATTN_RES,
        res0=RESOLUTION,
        n_resblocks=N_RESBLOCKS,
        p_drop=P_DROP,
        elt=ELT_SIZE,
    )
    report(parts, BATCH_SIZE, ELT_SIZE)

    if VERIFY:
        verify(VERIFY_BATCH)


if __name__ == "__main__":
    main()