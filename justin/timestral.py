"""Vendored Timestral encoder (mistralai/Timestral-*, code Apache-2.0, weights Modified MIT).

Re-implements the encoder half of the inference path of ``timestral.TimestralModel``
(https://github.com/mistralai/timestral, derived from commit a534bdb, Copyright Mistral AI,
Apache-2.0): causal patch scaler with asinh, the ``[values, missing]`` patch embedding, and the
time / time-variate transformer stack up to the final norm. The forecast head, the forecast-query
token and the covariate paths are not needed and are not loaded. State-dict key names match
upstream.

Tensors are shaped [B, C, N, d]: B samples, C variates, N patches (no extra tokens). Time
attention is causal over valid keys and runs along N within each (sample, variate); variate
attention runs along C within each (sample, patch). Unlike Toto, the released checkpoints put a
variate layer in every fourth block, so channels mix inside the stack and not only at its end.

In the upstream forecast pass the context tokens never read the decoder tokens, so the encoder
states computed here are exactly the encoder part of the forecast pass.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LAYER_KIND_AXES: dict[str, tuple[bool, ...]] = {  # ordered attention axes of a block; True = time
    "time": (True,),
    "variate": (False,),
    "time_variate": (True, False),
    "variate_time": (False, True),
}
_GATES = ("none", "sdpa_sigmoid")


@dataclass(frozen=True)
class TimestralConfig:
    """The fields of a checkpoint's ``config.json`` that change the encoder's forward pass."""

    patch_size: int
    d_model: int
    num_heads: int
    mlp_hidden_dim: int
    block_schedule: tuple[str, ...]
    context_length: int
    rope_theta: float = 10000.0
    rope_partial_factor: float = 0.5
    norm_eps: float = 1e-6
    attn_bias: bool = False
    mlp_bias: bool = False
    time_attn_gate: str = "sdpa_sigmoid"
    variate_attn_gate: str = "sdpa_sigmoid"
    minimum_scale: float = 1e-6
    use_asinh: bool = True
    stabilize_with_global: bool = False
    attend_missing_patches: bool = True

    def __post_init__(self):
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.rope_dim <= 0 or self.rope_dim % 2:
            raise ValueError(f"rope_partial_factor gives an invalid rope_dim={self.rope_dim}")
        if any(kind not in LAYER_KIND_AXES for kind in self.block_schedule):
            raise ValueError(f"unknown layer kind in {self.block_schedule}")
        if {self.time_attn_gate, self.variate_attn_gate} - set(_GATES):
            raise ValueError("unknown attention gate")
        if self.stabilize_with_global:
            raise NotImplementedError("stabilize_with_global is not used by the released checkpoints")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.rope_partial_factor)

    @classmethod
    def from_json(cls, path: str) -> TimestralConfig:
        with open(path) as handle:
            raw = json.load(handle)
        known = {field.name for field in fields(cls)}
        raw = {key: value for key, value in raw.items() if key in known}
        raw["block_schedule"] = tuple(raw["block_schedule"])
        return cls(**raw)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.scale


class _ResidualBlock(nn.Module):
    """``output(silu(hidden(x))) + residual(x)``, the patch embedding."""

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(self.activation(self.hidden_layer(x))) + self.residual_layer(x)


class _FusedSwiGLU(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, use_bias: bool):
        super().__init__()
        self.w12 = nn.Linear(embed_dim, 2 * hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w12(x).chunk(2, dim=-1)
        return self.w3(gate * F.silu(up))


class _RoPE(nn.Module):
    """Rotary embedding with sin/cos concatenation; only the first ``rotary_dim`` dims rotate."""

    inv_freq: Tensor

    def __init__(self, dim: int, base: float, rotary_dim: int):
        super().__init__()
        inv_freq = torch.zeros(dim // 2, dtype=torch.float32)
        index = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        inv_freq[: rotary_dim // 2] = 1.0 / (base ** (index / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """(cos, sin), each [rows, N, head_dim], for integer positions [rows, N]."""
        inv = self.inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv @ positions[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _additive(allowed: Tensor, dtype: torch.dtype) -> Tensor:
    """Float mask (0 where attention is allowed, the dtype's minimum elsewhere) from a boolean one."""
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


class _Attention(nn.Module):
    """Multi-head attention over the sequence axis of [batch, seq, D]; rotary on time axes only."""

    def __init__(self, embed_dim: int, num_heads: int, is_time: bool, use_bias: bool, gate: str):
        super().__init__()
        self.num_heads, self.head_dim, self.is_time = num_heads, embed_dim // num_heads, is_time
        self.wQKV = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.wO = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self._gate = nn.Linear(embed_dim, embed_dim, bias=use_bias) if gate == "sdpa_sigmoid" else None

    def forward(self, x: Tensor, mask: Tensor, cos: Tensor | None, sin: Tensor | None) -> Tensor:
        batch, seq, _ = x.shape
        # The fused weight is laid out as (qkv, head_dim, n_heads).
        qkv = self.wQKV(x.contiguous()).reshape(batch, seq, 3, self.head_dim, self.num_heads)
        q, k, v = (t.contiguous() for t in qkv.permute(2, 0, 1, 4, 3).unbind(dim=0))
        if self.is_time:
            assert cos is not None and sin is not None
            cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        if self._gate is not None:
            out = out * self._gate(x).sigmoid()
        return self.wO(out)


class _Block(nn.Module):
    """The attention(s) of one layer kind, each a pre-norm residual, then one SwiGLU MLP."""

    def __init__(self, kind: str, config: TimestralConfig):
        super().__init__()
        axes = LAYER_KIND_AXES[kind]
        self.axes = axes
        self.attn_norms = nn.ModuleList([_RMSNorm(config.d_model, config.norm_eps) for _ in axes])
        self.attns = nn.ModuleList(
            _Attention(
                config.d_model, config.num_heads, is_time, config.attn_bias,
                config.time_attn_gate if is_time else config.variate_attn_gate,
            )
            for is_time in axes
        )  # fmt: skip
        self.mlp_norm = _RMSNorm(config.d_model, config.norm_eps)
        self.mlp = _FusedSwiGLU(config.d_model, config.mlp_hidden_dim, config.mlp_bias)

    def forward(self, x: Tensor, time_mask: Tensor, variate_mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, c, n, d = x.shape
        for norm, attn, is_time in zip(self.attn_norms, self.attns, self.axes, strict=True):
            y = norm(x)
            if is_time:  # along patches within each (sample, variate)
                x = x + attn(y.reshape(b * c, n, d), time_mask, cos, sin).view(b, c, n, d)
            else:  # along variates within each (sample, patch)
                y = y.transpose(1, 2).reshape(b * n, c, d)
                x = x + attn(y, variate_mask, None, None).view(b, n, c, d).transpose(1, 2)
        return x + self.mlp(self.mlp_norm(x))


class _Transformer(nn.Module):
    def __init__(self, config: TimestralConfig):
        super().__init__()
        self.layers = nn.ModuleList(_Block(kind, config) for kind in config.block_schedule)
        self.rope = _RoPE(config.head_dim, config.rope_theta, config.rope_dim)
        self.out_norm = _RMSNorm(config.d_model, config.norm_eps)


class _PatchRepresentation(nn.Module):
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = _ResidualBlock(2 * patch_size, 4 * d_model, d_model, use_bias=True)

    def forward(self, scaled: Tensor, observed: Tensor) -> Tensor:
        """[R, T] normalized values and observation mask -> [R, T / patch_size, d_model]."""
        r, t = scaled.shape
        n = t // self.patch_size
        missing = (~observed).reshape(r, n, self.patch_size).to(scaled.dtype)
        x = torch.cat([scaled.reshape(r, n, self.patch_size), missing], dim=-1)
        return self.patch_proj(x.to(self.patch_proj.hidden_layer.weight.dtype))


def _causal_loc_scale(data: Tensor, observed: Tensor, patch: int, minimum_scale: float) -> tuple[Tensor, Tensor]:
    """Causal running mean and std (Welford form, float64, Bessel-corrected) read at the end of
    each patch and repeated over it. ``data`` is zero where unobserved."""
    w = observed.to(torch.float64)
    x = data.to(torch.float64)
    count = w.cumsum(-1).clamp_min(1.0)
    mean = (w * x).cumsum(-1) / count
    previous = torch.zeros_like(mean)
    previous[..., 1:] = mean[..., :-1]
    m2 = ((x - previous) * (x - mean) * w).cumsum(-1)
    scale = torch.sqrt(m2 / (count - 1.0).clamp(min=1.0)).clamp(min=minimum_scale)

    def per_patch(t: Tensor) -> Tensor:
        return t.unfold(-1, patch, patch)[..., -1].repeat_interleave(patch, dim=-1)

    return per_patch(mean), per_patch(scale)


class Timestral(nn.Module):
    """Timestral encoder: [B,C,T] (+observed mask) -> per-(variate, patch) states [B,C,N,d]."""

    def __init__(self, config: TimestralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.patch_length = config.patch_size
        self.context_length = config.context_length
        self.num_layers = len(config.block_schedule)
        self.representation = _PatchRepresentation(config.patch_size, config.d_model)
        self.transformer = _Transformer(config)

    @classmethod
    def from_pretrained(cls, repo: str, revision: str | None = None) -> Timestral:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config = TimestralConfig.from_json(hf_hub_download(repo, "config.json", revision=revision))
        path = hf_hub_download(repo, "model.safetensors", revision=revision)
        state = {
            key: value
            for key, value in load_file(path, device="cpu").items()
            if not key.startswith("head.") and key != "representation.forecast_query"
        }
        model = cls(config)
        model.load_state_dict(state)
        return model

    def truncate(self, depth: int) -> None:
        """Keep the first ``depth`` layers; the final norm then reads that layer's states."""
        if not 1 <= depth <= self.num_layers:
            raise ValueError(f"depth must be in [1, {self.num_layers}], got {depth}")
        self.transformer.layers = self.transformer.layers[:depth]
        self.num_layers = depth

    def scale(self, series: Tensor, observed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Causal per-patch normalization of [B, C, T]: (scaled, loc, scale), zero where unobserved."""
        data = torch.where(observed, series, 0.0)
        # Float64 statistics, as upstream; MPS has no float64, so compute them on the CPU.
        wide = data.cpu() if data.device.type == "mps" else data
        loc, scale = _causal_loc_scale(wide, observed.to(wide.device), self.patch_length, self.config.minimum_scale)
        loc, scale = loc.to(device=data.device, dtype=data.dtype), scale.to(device=data.device, dtype=data.dtype)
        scaled = (data - loc) / scale.clamp_min(torch.finfo(data.dtype).eps)
        return torch.where(observed, scaled.asinh() if self.config.use_asinh else scaled, 0.0), loc, scale

    def embed(self, series: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
        """Causal std scaling, asinh and patch embedding.

        series/observed: [B, C, T] with T a multiple of the patch size (observed True = a real
        value). Returns (tokens entering layer 0 [B, C, N, d], token_valid [B, C, N]); a patch is
        valid iff it has any observed value.
        """
        b, c, t = series.shape
        p = self.patch_length
        scaled, _, _ = self.scale(series, observed)
        x = self.representation(scaled.reshape(b * c, t), observed.reshape(b * c, t))
        return x.view(b, c, t // p, -1), observed.view(b, c, t // p, p).any(-1)

    def run_layers(
        self,
        x: Tensor,
        token_valid: Tensor,
        start: int = 0,
        stop: int | None = None,
        film: Sequence[tuple[Tensor, Tensor]] | None = None,
    ) -> Tensor:
        """Run layers [start:stop] on [B, C, N, d]; the final norm applies when stop is None.

        ``film`` holds one (scale, shift) pair of [B, d] per leading layer (layer i takes film[i]):
        the states entering that layer become ``x * (1 + scale) + shift``, the same for every
        patch of a sample. Zeros leave the pass unchanged.

        Positions are relative to the first patch any variate of the sample observes, and
        masks come from ``token_valid``, so resuming from any layer given its input activation
        is exact.
        """
        b, c, n, d = x.shape
        valid = token_valid.to(x.device)
        # A missing patch after the series has started is still a key (attend_missing_patches).
        keys = valid | valid.cummax(-1).values if self.config.attend_missing_patches else valid
        allowed = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        eye = torch.eye(n, dtype=torch.bool, device=x.device)
        time_mask = _additive(((allowed[None] & keys.reshape(b * c, 1, n)) | eye[None])[:, None], x.dtype)
        by_patch = keys.transpose(1, 2).reshape(b * n, 1, 1, c)
        variate_eye = torch.eye(c, dtype=torch.bool, device=x.device)
        variate_mask = _additive(by_patch.expand(b * n, 1, c, c) | variate_eye, x.dtype)
        first = valid.any(1).float().argmax(-1)  # [B] first patch any variate observes (0 if none)
        positions = (torch.arange(n, device=x.device)[None] - first[:, None])[:, None, :].expand(b, c, n)
        cos, sin = self.transformer.rope(x, positions.reshape(b * c, n))
        for i, layer in enumerate(
            self.transformer.layers[start : self.num_layers if stop is None else stop], start
        ):
            if film is not None and i < len(film):
                scale, shift = film[i]
                x = x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
            x = layer(x, time_mask, variate_mask, cos, sin)
        return self.transformer.out_norm(x) if stop is None else x

    def forward(
        self, series: Tensor, observed: Tensor, film: Sequence[tuple[Tensor, Tensor]] | None = None
    ) -> tuple[Tensor, Tensor]:
        x, token_valid = self.embed(series, observed)
        return self.run_layers(x, token_valid, film=film), token_valid
