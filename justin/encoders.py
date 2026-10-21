"""The frozen backbones: Timestral (any of the four sizes), Chronos-2 and MiniLM."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from .chronos2 import Chronos2
from .timestral import Timestral


class TimestralEncoder(nn.Module):
    """Frozen Timestral encoder (Mistral AI; code Apache-2.0, weights Modified MIT).

    [B, C, T] -> token states [B, M, d] and a bool mask [B, M], with S = N patch tokens of 32
    steps per channel (M = C * S). Series are left-padded (valid data right-aligned); NaN marks a
    missing value. The encoder normalizes every channel itself; do not pre-normalize.
    """

    name: str
    repo: str
    revision: str  # pinned commit; exports record the one they were trained on
    supports_film = True  # the question can condition the leading blocks

    def __init__(self, backbone: Timestral):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.hidden_size = backbone.hidden_size
        self._patch_length = backbone.patch_length

    @classmethod
    def from_pretrained(cls, revision: str | None = None) -> "TimestralEncoder":
        return cls(Timestral.from_pretrained(cls.repo, revision or cls.revision))

    @property
    def num_blocks(self) -> int:
        return self.backbone.num_layers

    @property
    def lora_root(self) -> nn.Module:
        return self.backbone

    def truncate(self, depth: int) -> None:
        self.backbone.truncate(depth)

    def lora_targets(self, boundary: int) -> Callable[[str], bool]:
        """Module names of blocks [boundary:], where LoRA adapters sit."""
        adapted = tuple(f"transformer.layers.{i}." for i in range(boundary, self.num_blocks))
        return lambda name: name.startswith(adapted)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        series: Tensor,
        padding_mask: Tensor | None = None,
        film: Sequence[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """series [B,C,T], padding_mask [B,C,T] (True = left padding), ``film`` per-block [B, d]
        scale and shift (see ``Timestral.run_layers``) -> (states [B, M, d], valid [B, M])."""
        series, observed = _prepare(self.backbone, self._patch_length, series, padding_mask)
        states, token_valid = self.backbone(series, observed, film=film)  # [B, C, S, d]
        b, c, s, _ = states.shape
        return states.reshape(b, c * s, self.hidden_size), token_valid.reshape(b, c * s)


class Timestral5mEncoder(TimestralEncoder):
    """Timestral, 5M parameters: 8 blocks, width 192."""

    name = "timestral5m"
    repo = "mistralai/Timestral-5M"
    revision = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


class Timestral28mEncoder(TimestralEncoder):
    """Timestral, 28M parameters: 12 blocks, width 384."""

    name = "timestral28m"
    repo = "mistralai/Timestral-28M"
    revision = "180698bbf5a50dda3516aa6419798437fb03a430"


class Timestral123mEncoder(TimestralEncoder):
    """Timestral, 123M parameters: 20 blocks, width 640."""

    name = "timestral123m"
    repo = "mistralai/Timestral-123M"
    revision = "5a147287addda85f6acd8da3fb8858fe9af7e8f7"


class Timestral209mEncoder(TimestralEncoder):
    """Timestral, 209M parameters: 24 blocks, width 768."""

    name = "timestral209m"
    repo = "mistralai/Timestral-209M"
    revision = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"


class Chronos2Encoder(nn.Module):
    """Frozen Chronos-2 encoder (amazon/chronos-2, Apache-2.0).

    Channels become variates of one group per sample, so Chronos-2's group attention mixes them.
    Each channel yields S = N + 2 tokens: N patches, a [REG] token and a future-patch token, so
    M = C * S. Missing values (NaN) are unobserved. Not question-conditioned (no FiLM), and its
    special tokens rule out the location head.
    """

    name = "chronos"
    repo = "amazon/chronos-2"
    revision = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    supports_film = False

    def __init__(self, backbone: Chronos2):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.hidden_size = backbone.hidden_size
        self._patch_length = backbone.patch_length

    @classmethod
    def from_pretrained(cls, revision: str | None = None) -> "Chronos2Encoder":
        return cls(Chronos2.from_pretrained(revision or cls.revision))

    @property
    def num_blocks(self) -> int:
        return len(self.backbone.encoder.block)

    @property
    def lora_root(self) -> nn.Module:
        return self.backbone

    def truncate(self, depth: int) -> None:
        raise NotImplementedError("Chronos-2 cannot be cut to a shallower depth")

    def lora_targets(self, boundary: int) -> Callable[[str], bool]:
        adapted = tuple(f"encoder.block.{i}." for i in range(boundary, self.num_blocks))
        return lambda name: name.startswith(adapted)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, series: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """series [B,C,T], padding_mask [B,C,T] (True = left padding) -> (states [B, M, d], valid [B, M])."""
        series, observed = _prepare(self.backbone, self._patch_length, series, padding_mask)
        states, token_valid = self.backbone(series, observed)  # [B, C, S, d]
        b, c, s, _ = states.shape
        return states.reshape(b, c * s, self.hidden_size), token_valid.reshape(b, c * s)


def _prepare(
    backbone: nn.Module, multiple: int, series: Tensor, padding_mask: Tensor | None
) -> tuple[Tensor, Tensor]:
    """Keep the most recent ``context_length`` steps, move to the backbone's device and dtype,
    left-pad T to a multiple of the patch length; -> (series, observed)."""
    if series.ndim != 3 or min(series.shape) == 0:
        raise ValueError("series must have nonempty [batch, channels, time] shape")
    context = backbone.context_length
    series = series[..., -context:]
    if padding_mask is None:
        padding_mask = torch.zeros_like(series, dtype=torch.bool)
    padding_mask = padding_mask[..., -context:]
    if padding_mask.shape != series.shape or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be boolean with the same shape as series")
    reference = next(backbone.parameters())
    series = series.to(device=reference.device, dtype=reference.dtype)
    padding_mask = padding_mask.to(device=reference.device)
    pad = -series.shape[-1] % multiple
    if pad > 0:
        series = F.pad(series, (pad, 0))
        padding_mask = F.pad(padding_mask, (pad, 0), value=True)
    return series, ~padding_mask & ~torch.isnan(series)


SeriesEncoder = TimestralEncoder | Chronos2Encoder
SERIES_ENCODERS: dict[str, type[SeriesEncoder]] = {
    encoder.name: encoder
    for encoder in (
        Chronos2Encoder, Timestral5mEncoder, Timestral28mEncoder, Timestral123mEncoder, Timestral209mEncoder,
    )
}


class TextEncoder(nn.Module):
    """Frozen MiniLM (Apache-2.0): token states and sentence embeddings for questions and options."""

    name = "minilm"
    repo = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    def __init__(self, backbone: PreTrainedModel, tokenizer, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.hidden_size: int = hidden_size or backbone.config.hidden_size

    @classmethod
    def from_pretrained(cls, revision: str | None = None) -> "TextEncoder":
        revision = revision or cls.revision
        return cls(
            AutoModel.from_pretrained(cls.repo, revision=revision),
            AutoTokenizer.from_pretrained(cls.repo, revision=revision),
        )

    def train(self, mode: bool = True):
        return super().train(False)

    def _hidden(self, texts: list[str]):
        """Token states [B, L, H] and the tokenizer output (truncated at 256 tokens)."""
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(next(self.backbone.parameters()).device)
        return self.backbone(**tokens).last_hidden_state, tokens

    def forward(self, texts: list[str]) -> Tensor:
        """Masked-mean pooled, L2-normalized sentence embeddings [B, H]."""
        hidden, tokens = self._hidden(texts)
        mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled, dim=-1)

    def token_states(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """Per-token last hidden state [B, L, H] and bool mask [B, L] (True = valid)."""
        hidden, tokens = self._hidden(texts)
        return hidden, tokens.attention_mask.bool()
