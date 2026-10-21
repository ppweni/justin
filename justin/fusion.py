"""Cross-attention fusion for System One's series encoder (Chronos-2).

``SystemOne`` wires ``CrossAttentionFusion`` over the
unpooled token states from a ``SeriesEncoder`` (``encoders.py``): the question's
per-token states query the signal's per-patch states.

Shape and mask conventions used throughout:
  * series    : [B, C, T]                float
  * signal    : [B, M, d_ts]             series token states (M = variates x tokens)
  * signal_mask / question_mask : [B, *] bool, True = valid, False = padding
  * question  : [B, L, d_txt]            MiniLM per-token states (L <= 256)
  * z         : [B, hidden_dim]          fused decision vector
"""

from torch import Tensor, nn


class _FusionBlock(nn.Module):
    """One transformer-decoder-style layer: question tokens attend over signal, then FFN."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, key_padding_mask: Tensor) -> Tensor:
        attended, _ = self.attention(
            queries, keys, keys, key_padding_mask=key_padding_mask, need_weights=False
        )
        queries = self.norm1(queries + self.dropout(attended))
        return self.norm2(queries + self.ffn(queries))


class CrossAttentionFusion(nn.Module):
    """Question-conditioned cross-attention over signal tokens -> fused vector z.

    Queries come from the question's per-token states; keys/values from the
    signal's per-token states. Stacks `layers` attention+FFN blocks, then pools
    the query tokens (masked mean), with a residual from the mean question token
    so a question that needs no signal still has a clean text path. With
    ``signal_skip`` a normalized masked-mean of the signal tokens is projected and added
    to z, a direct pooled-signal path (a frozen-feature linear probe scores well on it)
    that the attention blocks then refine.
    """

    def __init__(
        self,
        signal_dim: int,
        text_dim: int,
        *,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 1,
        dropout: float = 0.0,
        signal_skip: bool = False,
    ):
        super().__init__()
        if min(signal_dim, text_dim, hidden_dim, heads, layers) < 1:
            raise ValueError("dims, heads, and layers must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.query_projection = nn.Linear(text_dim, hidden_dim)
        self.kv_projection = nn.Linear(signal_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _FusionBlock(hidden_dim, heads, dropout) for _ in range(layers)
        )
        self.residual = nn.Linear(text_dim, hidden_dim)
        self.signal_skip = (
            nn.Sequential(nn.LayerNorm(signal_dim), nn.Linear(signal_dim, hidden_dim))
            if signal_skip
            else None
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(
        self,
        signal: Tensor,
        signal_mask: Tensor,
        question: Tensor,
        question_mask: Tensor,
    ) -> Tensor:
        """[B,M,d_ts] + [B,L,d_txt] -> fused z [B, hidden_dim]. Masks are True=valid."""
        if signal.ndim != 3 or question.ndim != 3:
            raise ValueError("signal and question must be [B, tokens, dim]")
        if signal_mask.shape != signal.shape[:2] or question_mask.shape != question.shape[:2]:
            raise ValueError("masks must be [B, tokens] matching their token sequence")
        if not signal_mask.any(dim=1).all() or not question_mask.any(dim=1).all():
            raise ValueError("every sample needs at least one valid signal and question token")

        queries = self.query_projection(question.to(self.query_projection.weight.dtype))
        keys = self.kv_projection(signal.to(self.kv_projection.weight.dtype))
        # nn.MultiheadAttention treats True as "ignore", so invert our valid-masks.
        key_padding_mask = ~signal_mask
        for block in self.blocks:
            queries = block(queries, keys, key_padding_mask)

        weights = question_mask.unsqueeze(-1).to(queries.dtype)
        denom = weights.sum(1).clamp_min(1.0)
        pooled = (queries * weights).sum(1) / denom  # [B, hidden_dim]
        question_mean = (question.to(weights.dtype) * weights).sum(1) / denom  # [B, d_txt]
        z = pooled + self.residual(question_mean)  # residual maps d_txt -> hidden_dim
        if self.signal_skip is not None:
            valid = signal_mask.unsqueeze(-1).to(signal.dtype)
            signal_mean = (signal * valid).sum(1) / valid.sum(1).clamp_min(1.0)  # [B, d_ts]
            z = z + self.signal_skip(signal_mean.to(self.query_projection.weight.dtype))
        return self.act(self.norm(z))




class OptionAttention(nn.Module):
    """Option-specific evidence routing for Choice.

    Each option's query (the fused question vector z plus that option's key) reads the
    masked signal tokens of its own sample, then a shared linear scorer turns the routed
    vector into one evidence logit. The scorer starts at zero, so a freshly enabled module
    leaves the dot-product logits unchanged and learns a correction. Options are only
    compared within a sample; nothing mixes across the batch.
    """

    def __init__(self, signal_dim: int, hidden_dim: int, *, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if min(signal_dim, hidden_dim, heads) < 1 or hidden_dim % heads:
            raise ValueError("dims must be positive and hidden_dim divisible by heads")
        self.kv_projection = nn.Linear(signal_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    def forward(self, z: Tensor, keys: Tensor, signal: Tensor, signal_mask: Tensor) -> Tensor:
        """z [B,d], keys [N,d] (projected options), signal [B,M,d_ts], mask [B,M] -> [B,N]."""
        queries = z.unsqueeze(1) + keys.unsqueeze(0).to(z.dtype)  # [B, N, d]
        kv = self.kv_projection(signal.to(self.kv_projection.weight.dtype))
        routed, _ = self.attention(
            queries, kv, kv, key_padding_mask=~signal_mask, need_weights=False
        )
        return self.scorer(self.norm(queries + routed)).squeeze(-1)


class ChannelHead(nn.Module):
    """Per-channel tokens that channels and questions compare, instead of a pool over all patches.

    Each series becomes one token: the masked mean of its patch states plus an embedding of
    its name and a projection of its numbers (how far it deviates, when). Self-attention lets
    channels see each other, so a ranking ("which deviated most?") or a pairing ("did A change
    before B?") is a comparison between tokens. The question's tokens then read the channel
    tokens, giving a summary vector for the fusion, and the tokens are returned for
    option-specific routing (an option that names a service reads that service's channels).
    """

    def __init__(
        self,
        signal_dim: int,
        text_dim: int,
        hidden_dim: int,
        *,
        n_features: int = 0,
        names: bool = True,
        heads: int = 8,
        layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not (names or n_features):
            raise ValueError("a channel head needs channel names or channel features")
        self.pool = nn.Sequential(nn.LayerNorm(signal_dim), nn.Linear(signal_dim, hidden_dim))
        self.names = nn.Linear(text_dim, hidden_dim) if names else None
        self.features = (
            nn.Sequential(nn.Linear(n_features, 64), nn.GELU(), nn.Linear(64, hidden_dim))
            if n_features
            else None
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, heads, 4 * hidden_dim, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            layers,
            enable_nested_tensor=False,
        )  # fmt: skip
        self.query_projection = nn.Linear(text_dim, hidden_dim)
        self.read = _FusionBlock(hidden_dim, heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        signal: Tensor,
        signal_mask: Tensor,
        channels: int,
        question: Tensor,
        question_mask: Tensor,
        names: Tensor | None = None,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """-> (summary [B,H], channel tokens [B,C,H], channel_valid [B,C] bool)."""
        batch, tokens, dim = signal.shape
        if tokens % channels:
            raise ValueError("signal tokens must divide evenly among the channels")
        patches = signal.view(batch, channels, tokens // channels, dim)
        valid_patches = signal_mask.view(batch, channels, tokens // channels, 1).to(signal.dtype)
        mean = (patches * valid_patches).sum(2) / valid_patches.sum(2).clamp_min(1.0)
        channel_valid = signal_mask.view(batch, channels, -1).any(dim=2)
        x = self.pool(mean)
        if self.names is not None:
            if names is None:
                raise ValueError("this channel head needs channel names")
            x = x + self.names(names.to(x))
        if self.features is not None:
            if features is None:
                raise ValueError("this channel head needs channel features")
            x = x + self.features(features.to(x))
        x = self.encoder(x, src_key_padding_mask=~channel_valid)
        queries = self.query_projection(question.to(x))
        queries = self.read(queries, x, ~channel_valid)
        weights = question_mask.unsqueeze(-1).to(queries.dtype)
        summary = (queries * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.norm(summary), x, channel_valid
