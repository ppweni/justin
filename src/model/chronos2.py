"""Vendored Chronos-2 encoder (amazon/chronos-2, Apache-2.0).

Re-implements only the encoder of ``chronos.chronos2.Chronos2Model`` (instance norm,
patching, time encoding, input embedding, [REG] + future-patch tokens, and the
time-attention / group-attention / feed-forward stack) so ``model.safetensors`` loads
without the ``chronos-forecasting`` dependency. State-dict key names match upstream.

Tensors are shaped [B, C, S, d]: B samples, C variates, S = N patches + [REG] + one
future-patch token. Time attention runs along S within each (sample, variate); group
attention runs along C within each (sample, time slot). That is the upstream
``group_ids = sample index`` layout without materializing a [B*C, B*C] mask.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

CHECKPOINT_REPO = "amazon/chronos-2"


class _LayerNorm(nn.Module):
    """T5-style RMS norm: no bias, no mean subtraction."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return self.weight * x


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(F.relu(self.hidden_layer(x))) + self.residual_layer(x)


class _MultiHeadAttention(nn.Module):
    """Unscaled (scale=1.0) multi-head attention, optionally with rotary embeddings."""

    def __init__(self, d_model: int, num_heads: int, d_kv: int, rope_theta: float | None):
        super().__init__()
        inner = num_heads * d_kv
        self.num_heads, self.d_kv = num_heads, d_kv
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        if rope_theta is not None:
            inv_freq = 1.0 / rope_theta ** (torch.arange(0, d_kv, 2, dtype=torch.float32) / d_kv)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.inv_freq: Tensor | None = None

    def _rotate(self, x: Tensor) -> Tensor:
        assert self.inv_freq is not None
        positions = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)
        freqs = positions[:, None] * self.inv_freq.to(x.device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """x [b, s, d]; mask additive, broadcastable to [b, heads, s, s]."""
        b, s, _ = x.shape

        def heads(t: Tensor) -> Tensor:
            return t.view(b, s, self.num_heads, self.d_kv).transpose(1, 2)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        if self.inv_freq is not None:
            q, k = self._rotate(q), self._rotate(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype), scale=1.0)
        return self.o(out.transpose(1, 2).reshape(b, s, -1))


class _Attention(nn.Module):
    """Pre-norm residual attention layer (time: RoPE; group: none)."""

    def __init__(self, d_model, num_heads, d_kv, eps, rope_theta):
        super().__init__()
        self.self_attention = _MultiHeadAttention(d_model, num_heads, d_kv, rope_theta)
        self.layer_norm = _LayerNorm(d_model, eps)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return x + self.self_attention(self.layer_norm(x), mask)


class _MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.wo(F.relu(self.wi(x)))


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, eps: float):
        super().__init__()
        self.mlp = _MLP(d_model, d_ff)
        self.layer_norm = _LayerNorm(d_model, eps)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.layer_norm(x))


class _Block(nn.Module):
    """layer[0] time attention, layer[1] group attention, layer[2] feed-forward."""

    def __init__(self, d_model, d_ff, num_heads, d_kv, eps, rope_theta):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                _Attention(d_model, num_heads, d_kv, eps, rope_theta),
                _Attention(d_model, num_heads, d_kv, eps, None),
                _FeedForward(d_model, d_ff, eps),
            ]
        )

    def forward(self, x: Tensor, time_mask: Tensor, group_mask: Tensor) -> Tensor:
        b, c, s, d = x.shape
        x = self.layer[0](x.reshape(b * c, s, d), time_mask).view(b, c, s, d)
        grouped = x.transpose(1, 2).reshape(b * s, c, d)
        x = self.layer[1](grouped, group_mask).view(b, s, c, d).transpose(1, 2)
        return self.layer[2](x)


class _Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, d_kv, eps, rope_theta):
        super().__init__()
        self.block = nn.ModuleList(
            [_Block(d_model, d_ff, num_heads, d_kv, eps, rope_theta) for _ in range(num_layers)]
        )
        self.final_layer_norm = _LayerNorm(d_model, eps)


def _additive(valid: Tensor, dtype: torch.dtype) -> Tensor:
    """Bool key validity [n, k] -> additive mask [n, 1, 1, k] (finfo.min where invalid)."""
    return (~valid).to(dtype)[:, None, None, :] * torch.finfo(dtype).min


class Chronos2(nn.Module):
    """Chronos-2 encoder: [B,C,T] (+observed mask) -> per-(variate, token) states [B,C,S,d]."""

    REG_TOKEN_ID = 1

    def __init__(
        self,
        *,
        d_model: int = 768,
        d_ff: int = 3072,
        d_kv: int = 64,
        num_heads: int = 12,
        num_layers: int = 12,
        patch_size: int = 16,
        rope_theta: float = 10000.0,
        layer_norm_epsilon: float = 1e-6,
        context_length: int = 8192,
        time_encoding_scale: int = 8192,
    ):
        super().__init__()
        self.hidden_size = d_model
        self.patch_length = patch_size
        self.context_length = context_length
        self.time_encoding_scale = time_encoding_scale
        self.shared = nn.Embedding(2, d_model)  # [PAD], [REG]
        # [time encoding, patch values, patch mask] per patch
        self.input_patch_embedding = _ResidualBlock(patch_size * 3, d_ff, d_model)
        self.encoder = _Encoder(
            num_layers, d_model, d_ff, num_heads, d_kv, layer_norm_epsilon, rope_theta
        )

    @classmethod
    def from_pretrained(cls, revision: str | None = None) -> "Chronos2":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(CHECKPOINT_REPO, "model.safetensors", revision=revision)
        state = {
            key: value
            for key, value in load_file(path, device="cpu").items()
            if not key.startswith("output_patch_embedding.")
        }
        model = cls()
        model.load_state_dict(state)
        return model

    def embed(self, series: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
        """Instance norm, patching, and input embedding.

        series/observed: [B, C, T] with T a multiple of patch_length (observed True=real
        value). Returns (tokens entering block 0 [B, C, N+2, d], token_valid [B, C, N+2]).
        Tokens are N patches, then [REG], then one future-patch token; REG and the future
        token are valid iff the variate has any observed value.
        """
        b, c, t = series.shape
        p = self.patch_length
        n = t // p
        dtype = self.shared.weight.dtype

        x = torch.where(observed, series.float(), torch.nan)
        loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, 1e-5, scale)
        x = torch.arcsinh((x - loc) / scale)

        mask = observed.to(dtype).view(b, c, n, p)
        values = torch.where(observed, x, 0.0).to(dtype).view(b, c, n, p)
        time = torch.arange(-t, 0, device=series.device, dtype=torch.float32)
        time = (time / self.time_encoding_scale).to(dtype).view(1, 1, n, p).expand(b, c, n, p)
        patches = self.input_patch_embedding(torch.cat([time, values, mask], dim=-1))

        any_observed = observed.any(dim=-1, keepdim=True)  # [B, C, 1]
        reg = self.shared.weight[self.REG_TOKEN_ID].expand(b, c, 1, -1)
        future_time = torch.arange(p, device=series.device, dtype=torch.float32)
        future_time = (future_time / self.time_encoding_scale).to(dtype)
        future_in = torch.cat([future_time, torch.zeros(2 * p, dtype=dtype, device=series.device)])
        future = self.input_patch_embedding(future_in).expand(b, c, 1, -1)

        tokens = torch.cat([patches, reg.to(patches.dtype), future], dim=2)
        patch_valid = mask.sum(dim=-1) > 0
        token_valid = torch.cat([patch_valid, any_observed, any_observed], dim=2)
        return tokens, token_valid

    def run_layers(
        self, x: Tensor, token_valid: Tensor, start: int = 0, stop: int | None = None
    ) -> Tensor:
        """Run blocks [start:stop] on [B, C, S, d]; the final norm applies when stop is None.

        RoPE positions are relative and pads are masked as keys, so resuming from any block
        given the block-start activation is exact, wherever the pads sit along S.
        """
        b, c, s = token_valid.shape
        time_mask = _additive(token_valid.reshape(b * c, s), x.dtype)
        group_mask = _additive(token_valid.transpose(1, 2).reshape(b * s, c), x.dtype)
        for block in self.encoder.block[start:stop]:
            x = block(x, time_mask, group_mask)
        if stop is None:
            x = self.encoder.final_layer_norm(x)
        return x

    def forward(self, series: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
        x, token_valid = self.embed(series, observed)
        return self.run_layers(x, token_valid), token_valid
