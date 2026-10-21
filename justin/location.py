"""Location heads: point at one (channel, patch) of the series, or at "not present".

Inputs follow the encoder/fusion convention: signal [B, M, d_series], question
tokens [B, L, d_text], and boolean masks with True meaning valid. Output logits
are [B, M + 1]; index M always means "not present"; apply softmax for probabilities.

For C channels with S tokens each, token (channel, patch) has index
``channel * S + patch``. Decode a non-null prediction with ``divmod(index, S)``.
Timestral emits only patch tokens.

Patch indices refer to the encoder's cropped, left-padded input, not directly
to original timestamps. For patch width P, crop offset O, and total left padding
A in the retained input, patch t covers original sample indices
``[O + t * P - A, O + (t + 1) * P - A)``; intersect with the observed window.
Map targets after all cropping, padding, and channel reordering. Targets outside
the observed window must be excluded or explicitly defined as absent-in-window.

This module selects one patch, such as an onset. It does not predict spans or
convert timestamps; ``Justin.locate`` maps the chosen patch back to samples.
"""

import torch
from torch import Tensor, nn


class LocationHead(nn.Module):
    """Project a pooled question and each patch into a shared pointer space.

    The learned null key competes with patch keys for "not present". All masked
    patch logits are negative infinity; a fully masked signal therefore has
    probability one on null. Every question must contain at least one valid
    token. Masked embeddings are zeroed before projection, including NaNs.

    Inputs must share the module's device. Floating embeddings are converted to
    its parameter dtype. Gradients flow to unmasked inputs when enabled, allowing
    optional joint fine-tuning; use detached/no-grad features to train only the
    head. Save/load its state_dict separately from the original checkpoint.
    """

    def __init__(self, signal_dim: int, text_dim: int, hidden_dim: int = 256):
        super().__init__()
        if min(signal_dim, text_dim, hidden_dim) < 1:
            raise ValueError("signal_dim, text_dim, and hidden_dim must be positive")
        self.query_projection = nn.Linear(text_dim, hidden_dim)
        self.key_projection = nn.Linear(signal_dim, hidden_dim)
        self.null_key = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.null_key, std=hidden_dim**-0.5)
        self.scale = hidden_dim**-0.5

    def forward(
        self,
        signal: Tensor,
        signal_mask: Tensor,
        question: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        """Return raw [B, M + 1] logits; the final class is always null."""
        if signal.ndim != 3 or question.ndim != 3:
            raise ValueError("signal and question must have [B, tokens, dim] shape")
        if min(signal.shape) < 1 or min(question.shape) < 1:
            raise ValueError("signal and question dimensions must be nonempty")
        if signal.shape[0] != question.shape[0]:
            raise ValueError("signal and question batch sizes must match")
        if (
            signal.shape[-1] != self.key_projection.in_features
            or question.shape[-1] != self.query_projection.in_features
        ):
            raise ValueError("embedding dimensions must match signal_dim and text_dim")
        if signal_mask.shape != signal.shape[:2] or question_mask.shape != question.shape[:2]:
            raise ValueError("masks must match their [B, tokens] input shapes")
        if signal_mask.dtype != torch.bool or question_mask.dtype != torch.bool:
            raise ValueError("masks must be boolean (True = valid)")
        if not signal.is_floating_point() or not question.is_floating_point():
            raise ValueError("signal and question embeddings must be floating point")
        reference = self.query_projection.weight
        if any(
            value.device != reference.device
            for value in (signal, signal_mask, question, question_mask)
        ):
            raise ValueError("inputs and LocationHead must be on the same device")
        if not question_mask.any(dim=1).all():
            raise ValueError("every question needs at least one valid token")

        signal = signal.to(reference.dtype).masked_fill(~signal_mask.unsqueeze(-1), 0)
        question = question.to(reference.dtype).masked_fill(~question_mask.unsqueeze(-1), 0)
        pooled = question.sum(dim=1) / question_mask.sum(dim=1, keepdim=True)
        query = self.query_projection(pooled)
        keys = self.key_projection(signal)
        patch_logits = torch.einsum("bd,bmd->bm", query, keys) * self.scale
        patch_logits = patch_logits.masked_fill(~signal_mask, -torch.inf)
        null_logits = (query * self.null_key).sum(dim=-1, keepdim=True) * self.scale
        return torch.cat((patch_logits, null_logits), dim=-1)


class _LocationBlock(nn.Module):
    """Pre-norm block: read the question (cross-attention), mix neighbouring patches (a depthwise
    convolution along time), then let all tokens and the null token talk (self-attention)."""

    def __init__(self, dim: int, heads: int, kernel: int, dropout: float):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(4))
        self.question = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.local = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.mixing = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, valid: Tensor, question: Tensor, question_pad: Tensor, tokens: int
    ) -> Tensor:
        """x [B, M + 1, dim] (the last token is the null token); valid [B, M + 1]."""
        keep = valid.unsqueeze(-1)
        h = self.norms[0](x)
        x = x + self.dropout(
            self.question(h, question, question, key_padding_mask=question_pad, need_weights=False)[
                0
            ]
        )
        patches = self.norms[1](x[:, :-1]).masked_fill(~keep[:, :-1], 0)
        b, m, dim = patches.shape
        local = patches.reshape(b * (m // tokens), tokens, dim).transpose(1, 2)
        local = self.local(local).transpose(1, 2).reshape(b, m, dim)
        x = torch.cat((x[:, :-1] + self.dropout(local), x[:, -1:]), dim=1)
        h = self.norms[2](x)
        x = x + self.dropout(self.mixing(h, h, h, key_padding_mask=~valid, need_weights=False)[0])
        return x + self.dropout(self.mlp(self.norms[3](x)))


class LocationAttentionHead(nn.Module):
    """A stronger pointer than ``LocationHead``: the question is read token by token.

    ``LocationHead`` reduces the question to one vector and takes one dot product per patch, which
    has to carry the channel named, the event type and the onset together. This head projects the
    patches and the question tokens to ``hidden_dim`` and runs ``layers`` blocks of cross-attention to
    the question (which channel, which event), a local convolution along time (what changed from
    one patch to the next) and self-attention over all patches and a learned null token. A linear
    layer reads one logit per patch; the null token's state gives the "not present" logit.

    Same contract as ``LocationHead`` with one addition: ``forward`` needs ``tokens_per_channel``
    (S) to know the time axis of the flattened [channel * S + patch] layout. Masks, null index
    (always the last class), padding behaviour and the freezing rules are the same.
    """

    def __init__(
        self,
        signal_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        kernel: int = 5,
        dropout: float = 0.1,
        max_tokens: int = 1024,
    ):
        super().__init__()
        if min(signal_dim, text_dim, hidden_dim, layers, heads, kernel, max_tokens) < 1:
            raise ValueError("dimensions, layers, heads, kernel and max_tokens must be positive")
        if hidden_dim % heads or kernel % 2 == 0:
            raise ValueError("hidden_dim must divide by heads and kernel must be odd")
        self.max_tokens = max_tokens
        self.signal_projection = nn.Linear(signal_dim, hidden_dim)
        self.question_projection = nn.Linear(text_dim, hidden_dim)
        self.position = nn.Embedding(max_tokens, hidden_dim)
        self.null_token = nn.Parameter(torch.randn(hidden_dim) * hidden_dim**-0.5)
        self.blocks = nn.ModuleList(
            _LocationBlock(hidden_dim, heads, kernel, dropout) for _ in range(layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.patch_logit = nn.Linear(hidden_dim, 1)
        self.null_logit = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        signal: Tensor,
        signal_mask: Tensor,
        question: Tensor,
        question_mask: Tensor,
        tokens_per_channel: int,
    ) -> Tensor:
        """Return raw [B, M + 1] logits; the final class is always null."""
        if signal.ndim != 3 or question.ndim != 3:
            raise ValueError("signal and question must have [B, tokens, dim] shape")
        if min(signal.shape) < 1 or min(question.shape) < 1:
            raise ValueError("signal and question dimensions must be nonempty")
        if signal.shape[0] != question.shape[0]:
            raise ValueError("signal and question batch sizes must match")
        if (
            signal.shape[-1] != self.signal_projection.in_features
            or question.shape[-1] != self.question_projection.in_features
        ):
            raise ValueError("embedding dimensions must match signal_dim and text_dim")
        if signal_mask.shape != signal.shape[:2] or question_mask.shape != question.shape[:2]:
            raise ValueError("masks must match their [B, tokens] input shapes")
        if signal_mask.dtype != torch.bool or question_mask.dtype != torch.bool:
            raise ValueError("masks must be boolean (True = valid)")
        if not signal.is_floating_point() or not question.is_floating_point():
            raise ValueError("signal and question embeddings must be floating point")
        reference = self.signal_projection.weight
        if any(
            v.device != reference.device for v in (signal, signal_mask, question, question_mask)
        ):
            raise ValueError("inputs and the head must be on the same device")
        if not question_mask.any(dim=1).all():
            raise ValueError("every question needs at least one valid token")
        if not 0 < tokens_per_channel <= self.max_tokens or signal.shape[1] % tokens_per_channel:
            raise ValueError(
                "tokens_per_channel must divide the signal tokens and not exceed max_tokens"
            )

        batch, count, _ = signal.shape
        signal = signal.to(reference.dtype).masked_fill(~signal_mask.unsqueeze(-1), 0)
        question = question.to(reference.dtype).masked_fill(~question_mask.unsqueeze(-1), 0)
        x = self.signal_projection(signal)
        steps = torch.arange(tokens_per_channel, device=x.device).repeat(
            count // tokens_per_channel
        )
        x = (x + self.position(steps)).masked_fill(~signal_mask.unsqueeze(-1), 0)
        x = torch.cat((x, self.null_token.expand(batch, 1, -1)), dim=1)
        valid = torch.cat((signal_mask, signal_mask.new_ones(batch, 1)), dim=1)
        question = self.question_projection(question)
        question_pad = ~question_mask
        for block in self.blocks:
            x = block(x, valid, question, question_pad, tokens_per_channel)
        x = self.out_norm(x)
        patch_logits = self.patch_logit(x[:, :-1]).squeeze(-1).masked_fill(~signal_mask, -torch.inf)
        return torch.cat((patch_logits, self.null_logit(x[:, -1])), dim=-1)


def location_head_from_spec(spec: dict) -> nn.Module:
    """Build an untrained head from the ``head`` metadata of a location export."""
    if spec["type"] == "attn":
        return LocationAttentionHead(
            spec["signal_dim"], spec["text_dim"], spec["hidden_dim"], layers=spec["layers"],
            heads=spec["heads"], kernel=spec["kernel"], dropout=0.0, max_tokens=spec["max_tokens"],
        )  # fmt: skip
    if spec["type"] == "pointer":
        return LocationHead(spec["signal_dim"], spec["text_dim"], spec["hidden_dim"])
    raise ValueError(f"unknown location head type: {spec['type']}")


def location_logits(
    system: nn.Module,
    head: nn.Module,
    series: Tensor,
    padding_mask: Tensor | None,
    questions: list[str],
    channel_names: list[list[str]] | None = None,
    candidates: Tensor | None = None,
) -> Tensor:
    """[B, M + 1] logits of a location head on a frozen ``SystemOne``'s series tokens.

    The series goes to the encoder raw (no baseline features), conditioned on the location
    question when the model has FiLM; a model built with channel names adds each channel's name
    embedding to its tokens, as the head was trained. ``candidates`` [B, M] further restricts the
    selectable patches (for example to drop padded channels); the encoder's validity mask always
    applies. Only the head receives gradients.
    """
    if system.text_encoder is None:
        raise ValueError("a location head needs a model with a text encoder")
    with torch.no_grad():
        conditioning = system._conditioning(questions, series)
        signal, valid = system.series_encoder(series, padding_mask, **conditioning)
        text, text_mask = system.text_encoder.token_states(questions)
        if system.channel_projection is not None:
            if channel_names is None:
                raise ValueError("this model was trained with channel names: pass channel_names")
            bias = system.channel_projection(system.channel_embeddings(channel_names).to(signal))
            b, c = series.shape[:2]
            signal = (signal.reshape(b, c, -1, signal.shape[-1]) + bias[:, :, None]).reshape_as(signal)
    mask = valid if candidates is None else candidates.to(valid.device) & valid
    if isinstance(head, LocationAttentionHead):
        return head(signal, mask, text, text_mask, signal.shape[1] // series.shape[1])
    return head(signal, mask, text, text_mask)
