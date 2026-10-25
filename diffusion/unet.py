import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass, asdict
from einops import rearrange
from collections import deque
from .fp8_kernels import SUPPORTED_UNET_SHAPES, triton_conv2d_fp8


@dataclass
class UNetConfig:
    in_channels: int
    base_hidden: int
    scales: list[int]
    n_heads: int
    p_drop: float
    attn_resolutions: list[int]
    in_resolution: int
    n_resblocks: int
    fp8_convs_enabled: bool = False
    fp8_scaling_type: str = "current"
    fp8_amax_history: int = 1024

    def as_dict(self):
        return asdict(self)


# TODO: perhaps subclass this from nn.conv2d
class Conv2dFP8(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fp8_convs_enabled: bool,
        scaling_type: str = "current",
        amax_history: int | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)

        self.scaling_type = scaling_type
        self.fp8_convs_enabled = fp8_convs_enabled

        assert self.scaling_type in ["current", "delayed"]

        # TODO: technically this works fine but the first few gradient steps are uncalibrated
        if scaling_type == "delayed":
            assert amax_history is not None
            self.amax_history = amax_history
            self.register_buffer("amax_window_weights", torch.zeros(amax_history))
            self.register_buffer("amax_window_acts", torch.zeros(amax_history))
            self.register_buffer("amax_window_dy", torch.zeros(amax_history))
            self.register_buffer("window_pos", torch.zeros((), dtype=torch.long))

    def conv_fp8(self, x: torch.Tensor, weight: torch.Tensor) -> bool:
        shape = (x.shape[1], weight.shape[0], x.shape[2], x.shape[3])
        return self.fp8_convs_enabled and shape in SUPPORTED_UNET_SHAPES

    def forward(self, x):
        if self.conv_fp8(x, self.conv.weight):
            if self.scaling_type == "current":
                out = triton_conv2d_fp8(
                    x.contiguous(memory_format=torch.channels_last), self.conv.weight
                )
                return out
            else:
                with torch.no_grad():
                    x_scale_hist = torch.max(self.amax_window_acts)  # type: ignore
                    w_scale_hist = torch.max(self.amax_window_weights)  # type: ignore

                out, x_scale_curr, w_scale_curr = triton_conv2d_fp8(
                    x.contiguous(memory_format=torch.channels_last),
                    self.conv.weight,
                    (x_scale_hist, w_scale_hist),
                    (self.amax_window_dy, self.window_pos),  # type: ignore
                    do_delayed=True,
                )

                with torch.no_grad():
                    window_idx = self.window_pos.unsqueeze(0)  # type: ignore
                    self.amax_window_weights.scatter_(  # type: ignore
                        0, window_idx, w_scale_curr.detach().unsqueeze(0)
                    )
                    self.amax_window_acts.scatter_(  # type: ignore
                        0, window_idx, x_scale_curr.detach().unsqueeze(0)
                    )
                    self.window_pos.add_(1).remainder_(self.amax_history)  # type: ignore
                return out

        else:
            return self.conv(x)


class TimeEmbed(nn.Module):
    def __init__(self, embed_channels: int, channels: int):
        super().__init__()

        self.mlp = nn.Linear(embed_channels, channels, bias=False)

    def forward(self, t: torch.Tensor):
        return self.mlp(t)[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        config: UNetConfig,
        in_chann: int,
        out_chann: int,
        p_drop: float = 0.0,
    ):
        super().__init__()

        self.conv_1 = Conv2dFP8(
            in_chann,
            out_chann,
            fp8_convs_enabled=config.fp8_convs_enabled,
            scaling_type=config.fp8_scaling_type,
            amax_history=config.fp8_amax_history,
        )
        self.conv_2 = Conv2dFP8(
            out_chann,
            out_chann,
            fp8_convs_enabled=config.fp8_convs_enabled,
            scaling_type=config.fp8_scaling_type,
            amax_history=config.fp8_amax_history,
        )

        self.norm1 = nn.GroupNorm(32, in_chann)
        self.norm2 = nn.GroupNorm(32, out_chann)

        self.t_embed_wb = TimeEmbed(4 * config.base_hidden, 2 * out_chann)

        self.skip_connection = nn.Identity()
        if in_chann != out_chann:
            self.skip_connection = nn.Conv2d(in_chann, out_chann, (1, 1), bias=False)  # type: ignore

        self.dropout = nn.Dropout(p_drop)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.conv_2.conv.weight)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        resid = x

        x = self.conv_1(F.silu(self.norm1(x)))

        wb = self.t_embed_wb(t)

        w, b = torch.chunk(wb, chunks=2, dim=1)

        x = (w + 1) * self.norm2(x) + b

        x = self.conv_2(self.dropout(F.silu(x)))

        return (self.skip_connection(resid) + x) / math.sqrt(2.0)


class CHWAttention(nn.Module):
    def __init__(self, config: UNetConfig, hidden_size: int):
        super().__init__()
        self.config = config

        self.nh = config.n_heads

        self.qkv = nn.Linear(in_features=hidden_size, out_features=3 * hidden_size, bias=False)
        self.out = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

        self.norm = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        _, _, h, w = x.shape

        x = rearrange(x, "b c h w -> b (h w) c")

        x = self.norm(x)

        qkv = self.qkv(x)

        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        queries = rearrange(q, "b l (nh hd) -> b nh l hd", nh=self.nh)
        keys = rearrange(k, "b l (nh hd) -> b nh l hd", nh=self.nh)
        values = rearrange(v, "b l (nh hd) -> b nh l hd", nh=self.nh)

        attn_out = F.scaled_dot_product_attention(queries, keys, values)

        o = self.out(rearrange(attn_out, "b nh l hd -> b l (nh hd)"))

        out: torch.Tensor = rearrange(o, "b (h w) c -> b c h w", h=h, w=w) + resid

        return out / math.sqrt(2.0)


class Stage(nn.Module):
    def __init__(
        self,
        config: UNetConfig,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        attention: bool = False,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        self.blocks.append(
            ResidualBlock(config, in_channels, hidden_channels, p_drop=config.p_drop)
        )
        self.attns.append(CHWAttention(config, hidden_channels) if attention else nn.Identity())

        for _ in range(num_blocks - 2):
            self.blocks.append(
                ResidualBlock(config, hidden_channels, hidden_channels, p_drop=config.p_drop)
            )
            self.attns.append(CHWAttention(config, hidden_channels) if attention else nn.Identity())

        self.blocks.append(
            ResidualBlock(config, hidden_channels, out_channels, p_drop=config.p_drop)
        )
        self.attns.append(CHWAttention(config, out_channels) if attention else nn.Identity())

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        for block, attn in zip(self.blocks, self.attns):
            x = block(x, t)
            x = attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()

        self.config = config

        self.base_channs = config.base_hidden

        channs_1 = config.scales[0] * self.base_channs

        self.conv_in = nn.Conv2d(
            config.in_channels, channs_1, kernel_size=(3, 3), padding=1, bias=False
        )

        down_stages = deque()
        up_stages = deque()

        # build the UNet stages
        res = config.in_resolution
        for i in range(0, len(config.scales) - 1):
            scales_curr, scales_next = config.scales[i], config.scales[i + 1]

            chann_curr, chann_next = (
                self.base_channs * scales_curr,
                self.base_channs * scales_next,
            )

            do_attn = res in config.attn_resolutions

            stage_down = Stage(
                config,
                chann_curr,
                chann_curr,
                chann_next,
                attention=do_attn,
                num_blocks=config.n_resblocks,
            )

            stage_up = Stage(
                config,
                chann_next + chann_next,
                chann_curr,
                chann_curr,
                attention=do_attn,
                num_blocks=config.n_resblocks + 1,
            )

            down_stages.append(stage_down)
            up_stages.appendleft(stage_up)

            res = res // 2

        mid_stage = config.scales[-1] * self.base_channs

        stage_down = Stage(config, mid_stage, mid_stage, mid_stage, num_blocks=config.n_resblocks)
        stage_up = Stage(
            config,
            mid_stage + mid_stage,
            mid_stage,
            mid_stage,
            num_blocks=config.n_resblocks + 1,
        )

        down_stages.append(stage_down)
        up_stages.appendleft(stage_up)

        self.mid = Stage(
            config,
            mid_stage,
            mid_stage,
            mid_stage,
            attention=True, # TODO: this is wrong!
            num_blocks=config.n_resblocks,
        )

        self.down_stages = nn.ModuleList(list(down_stages))
        self.up_stages = nn.ModuleList(list(up_stages))

        out_channels = self.config.in_channels

        self.out = nn.Sequential(
            nn.GroupNorm(32, channs_1),
            nn.SiLU(),
            nn.Conv2d(channs_1, out_channels, kernel_size=(3, 3), padding=1, bias=False),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.base_channs, 4 * self.config.base_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(4 * self.config.base_hidden, 4 * self.config.base_hidden, bias=False),
            nn.SiLU(),
        )

        self.reset_parameters()

        self.downsample = partial(F.interpolate, scale_factor=0.5, mode="nearest")
        self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")

        half = self.base_channs // 2
        freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * (math.log(10000.0) / (half)))

        self.register_buffer("freqs", freqs, persistent=False)

    def _timestep(self, t: torch.Tensor):
        emb = (t)[:, None].float() * self.freqs[None]  # type: ignore
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.base_channs % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]

    def reset_parameters(self):
        assert isinstance(self.out[-1], nn.Conv2d)
        torch.nn.init.zeros_(self.out[-1].weight)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_embed(self._timestep(t))

        act = self.conv_in(x)  # [B,C,H,W]

        hs = []
        for i, stage in enumerate(self.down_stages):
            if i > 0:
                act = stage(self.downsample(act), t_emb)
            else:
                act = stage(act, t_emb)

            hs.append(act)

        act = self.mid(act, t_emb)

        for i, stage in enumerate(self.up_stages):
            h = hs.pop()
            if i > 0:
                act = stage(torch.concat([self.upsample(act), h], dim=1), t_emb)
            else:
                act = stage(torch.concat([act, h], dim=1), t_emb)

        out = self.out(act)  # [B,C,H,W]

        return out
