from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from .chronos2 import Chronos2


class SeriesEncoder(nn.Module, ABC):
    """Frozen time-series foundation model; [B,C,T] -> token states [B,M,d] and bool mask [B,M].

    Subclasses wrap a transformer backbone whose blocks can be split at a boundary: the
    frozen prefix [0:boundary] is cacheable, the suffix [boundary:] carries LoRA adapters.
    Per-channel tokens flatten as M = C * S (S tokens per channel). Series are left-padded
    (valid data right-aligned), matching batch collation. The encoder applies its own
    normalization; do not pre-normalize.
    """

    name: str  # cache id, e.g. "chronos"
    hidden_size: int

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()

    @classmethod
    @abstractmethod
    def from_pretrained(cls) -> "SeriesEncoder": ...

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        """Number of transformer blocks LoRA can adapt (boundary is in [0, num_blocks])."""

    @abstractmethod
    def lora_targets(self, boundary: int) -> Callable[[str], bool]:
        """Predicate on backbone-qualified module names selecting blocks [boundary:]."""

    @property
    def lora_root(self) -> nn.Module:
        return self.backbone

    @abstractmethod
    def forward(
        self, series: Tensor, padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """series [B,C,T] float, padding_mask [B,C,T] bool True=left-pad -> (H_ts, token_mask)."""

    @abstractmethod
    def encode_prefix(
        self, series: Tensor, padding_mask: Tensor | None, boundary: int
    ) -> tuple[Tensor, Tensor]:
        """Frozen blocks [0:boundary] -> (prefix [B,C,S,d], token_valid [B,C,S] True=valid)."""

    @abstractmethod
    def encode_suffix(
        self, prefix: Tensor, token_valid: Tensor, boundary: int
    ) -> tuple[Tensor, Tensor]:
        """Adapted blocks [boundary:] on a cached prefix -> (H_ts [B,M,d], token_mask [B,M])."""

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _model_tensor(self) -> Tensor:
        return next(self.backbone.parameters())

    def _validated(
        self, series: Tensor, padding_mask: Tensor | None, multiple: int
    ) -> tuple[Tensor, Tensor]:
        """Validate, move to the backbone's device/dtype, and left-pad T to `multiple`."""
        if series.ndim != 3 or min(series.shape) == 0:
            raise ValueError("series must have nonempty [batch, channels, time] shape")
        if padding_mask is None:
            padding_mask = torch.zeros_like(series, dtype=torch.bool)
        if padding_mask.shape != series.shape or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean with the same shape as series")
        reference = self._model_tensor()
        series = series.to(device=reference.device, dtype=reference.dtype)
        padding_mask = padding_mask.to(device=reference.device)
        pad = -series.shape[-1] % multiple
        if pad > 0:
            series = F.pad(series, (pad, 0))
            padding_mask = F.pad(padding_mask, (pad, 0), value=True)
        return series, padding_mask


class Chronos2Encoder(SeriesEncoder):
    """Frozen vendored Chronos-2 encoder (amazon/chronos-2).

    Channels become variates of one group per sample, so Chronos-2's group attention mixes
    them. Each channel yields S = N + 2 tokens: N patches, a [REG] token and a future-patch
    token (both part of Chronos-2's pretrained input layout; they act as summary tokens),
    so M = C * S. Missing values (NaN) are treated as unobserved.
    """

    name = "chronos"

    def __init__(self, backbone: Chronos2):
        super().__init__(backbone)
        self.backbone: Chronos2
        self.hidden_size = backbone.hidden_size
        self._patch_length = backbone.patch_length

    @classmethod
    def from_pretrained(cls):
        return cls(Chronos2.from_pretrained())

    @property
    def num_blocks(self) -> int:
        return len(self.backbone.encoder.block)

    def lora_targets(self, boundary: int) -> Callable[[str], bool]:
        adapted = tuple(f"encoder.block.{i}." for i in range(boundary, self.num_blocks))
        return lambda name: name.startswith(adapted)

    def _prepare(self, series: Tensor, padding_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        """Validate, keep the most recent context_length steps, pad to patches; -> (x, observed)."""
        if series.ndim == 3:
            series = series[..., -self.backbone.context_length :]
            if padding_mask is not None and padding_mask.ndim == 3:
                padding_mask = padding_mask[..., -self.backbone.context_length :]
        series, padding_mask = self._validated(series, padding_mask, self._patch_length)
        return series, ~padding_mask & ~torch.isnan(series)

    @torch.no_grad()
    def forward(
        self, series: Tensor, padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        series, observed = self._prepare(series, padding_mask)
        states, token_valid = self.backbone(series, observed)  # [B, C, S, d]
        B, C, S, _ = states.shape
        return states.reshape(B, C * S, self.hidden_size), token_valid.reshape(B, C * S)

    @torch.no_grad()
    def encode_prefix(
        self, series: Tensor, padding_mask: Tensor | None, boundary: int
    ) -> tuple[Tensor, Tensor]:
        series, observed = self._prepare(series, padding_mask)
        x, token_valid = self.backbone.embed(series, observed)
        return self.backbone.run_layers(x, token_valid, start=0, stop=boundary), token_valid

    def encode_suffix(
        self, prefix: Tensor, token_valid: Tensor, boundary: int
    ) -> tuple[Tensor, Tensor]:
        B, C, S, _ = prefix.shape
        prefix = prefix.to(dtype=self._model_tensor().dtype)
        states = self.backbone.run_layers(prefix, token_valid, start=boundary)
        return states.reshape(B, C * S, self.hidden_size), token_valid.reshape(B, C * S)


SERIES_ENCODERS: dict[str, type[SeriesEncoder]] = {
    Chronos2Encoder.name: Chronos2Encoder,
}


class TextEncoder(nn.Module):
    """Frozen MiniLM sentence embeddings, shared by questions and options."""

    def __init__(self, backbone: PreTrainedModel, tokenizer):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.hidden_size: int = backbone.config.hidden_size
        # Set True once LoRA adapters are injected, so grad flows to them.
        self.trainable: bool = False

    @classmethod
    def from_pretrained(cls):
        name = "sentence-transformers/all-MiniLM-L6-v2"
        return cls(AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name))

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, texts: list[str]) -> Tensor:
        with torch.set_grad_enabled(self.trainable):
            tokens = self.tokenizer(
                texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(next(self.backbone.parameters()).device)
            hidden = self.backbone(**tokens).last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            return F.normalize(pooled, dim=-1)

    def token_states(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """Per-token last hidden state [B, L, H] and bool mask [B, L] (True=valid)."""
        with torch.set_grad_enabled(self.trainable):
            tokens = self.tokenizer(
                texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(next(self.backbone.parameters()).device)
            hidden = self.backbone(**tokens).last_hidden_state
            return hidden, tokens.attention_mask.bool()
