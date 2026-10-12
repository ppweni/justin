from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor, nn

from .encoders import SERIES_ENCODERS, SeriesEncoder, TextEncoder
from .fusion import CrossAttentionFusion, OptionAttention
from .lora import inject_lora

Head = Literal["choice", "noul", "score"]


class SystemOne(nn.Module):
    """Series encoder (Chronos-2) + MiniLM + cross-attention fusion + task heads.

    Returns raw logits: Choice [B, N], Noul [B], Score [B, K].
    Training uses CE for Choice, CE + ranked probability score for Score,
    and BCEWithLogitsLoss for Noul. At inference,
    apply softmax or sigmoid to logits / temperature. Score levels are a
    fixed rubric indexed 0..K-1; its expected index is sum(k * p[k]).
    """

    def __init__(
        self,
        series_encoder: SeriesEncoder,
        text_encoder: TextEncoder | None = None,
        *,
        num_classes: int | None = None,
        score_levels: int = 5,
        hidden_dim: int = 256,
        fusion_layers: int = 1,
        dropout: float = 0.0,
        option_attention: bool = False,
        signal_skip: bool = False,
        series_lora_layers: int = 0,
        minilm_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        # Calibration temperatures per head, filled when loading a published model.
        self.temperatures: dict[str, float] = {}
        if hidden_dim < 1 or score_levels < 2:
            raise ValueError("hidden_dim must be positive and score_levels >= 2")
        if num_classes is not None and not 2 <= num_classes <= 255:
            raise ValueError("num_classes must be between 2 and 255")
        self.series_encoder = series_encoder
        self.text_encoder = text_encoder
        self.series_lora_layers = series_lora_layers
        text_dim = text_encoder.hidden_size if text_encoder is not None else 0
        signal_dim = series_encoder.hidden_size
        if text_encoder is not None:
            self.fusion: nn.Module = CrossAttentionFusion(
                signal_dim=signal_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                layers=fusion_layers,
                dropout=dropout,
                signal_skip=signal_skip,
            )
        else:
            if signal_skip:
                raise ValueError("signal_skip requires a text encoder")
            self.fusion = nn.Sequential(
                nn.Linear(signal_dim, hidden_dim),
                nn.GELU(),
            )
        self.option_projection = nn.Linear(text_dim, hidden_dim) if text_dim else None
        if option_attention and text_encoder is None:
            raise ValueError("option_attention requires a text encoder")
        self.option_attention = (
            OptionAttention(signal_dim, hidden_dim, dropout=dropout) if option_attention else None
        )
        self.classes = nn.Embedding(num_classes, hidden_dim) if num_classes is not None else None
        self.noul = nn.Linear(hidden_dim, 1)
        self.score = nn.Linear(hidden_dim, score_levels)

        if not 0 <= series_lora_layers <= series_encoder.num_blocks:
            raise ValueError(
                f"series_lora_layers must be in [0, {series_encoder.num_blocks}] "
                f"for {series_encoder.name}"
            )
        self.lora_boundary = series_encoder.num_blocks - series_lora_layers
        if series_lora_layers > 0:
            inject_lora(
                series_encoder.lora_root,
                series_encoder.lora_targets(self.lora_boundary),
                rank=lora_rank,
                alpha=lora_alpha,
            )
        if minilm_lora:
            assert text_encoder is not None, "minilm_lora requires a text encoder"
            inject_lora(
                text_encoder.backbone,
                lambda name: "encoder.layer." in name,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            text_encoder.trainable = True

    @classmethod
    def from_pretrained(
        cls,
        *,
        with_text: bool = True,
        series_encoder: str = "chronos",
        revisions: Mapping[str, str] | None = None,
        **kwargs,
    ):
        """Load frozen backbones; fusion and heads start untrained.

        `revisions` maps "series" and "text" to Hub commits; omitted entries use the pinned
        revision of the encoder class.
        """
        if series_encoder not in SERIES_ENCODERS:
            raise ValueError(f"unknown series_encoder: {series_encoder}")
        revisions = revisions or {}
        return cls(
            SERIES_ENCODERS[series_encoder].from_pretrained(revision=revisions.get("series")),
            TextEncoder.from_pretrained(revision=revisions.get("text")) if with_text else None,
            **kwargs,
        )

    def forward(
        self,
        series: Tensor,
        questions: list[str] | None = None,
        *,
        options: list[str] | None = None,
        head: Head = "choice",
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Series is [B, C, T]; options are shared across the batch.

        Supply one shared question or B questions. Omit questions/options
        for a fixed catalog. Include relevant metadata in question text.
        """
        features = self.series_encoder(series, padding_mask)
        return self.answer(features, questions, options=options, head=head)

    def series_features(self, inputs: dict) -> tuple[Tensor, Tensor]:
        """Signal features from cache: the final H_ts, or the LoRA suffix over a cached prefix."""
        if "signal_prefix" in inputs:
            return self.series_encoder.encode_suffix(
                inputs["signal_prefix"], inputs["prefix_valid"], self.lora_boundary
            )
        return inputs["signal"], inputs["signal_mask"]

    def answer(
        self,
        features: tuple[Tensor, Tensor],
        questions: list[str] | None = None,
        *,
        options: list[str] | None = None,
        head: Head = "choice",
    ) -> Tensor:
        """Reuse series_encoder(series) across independent questions.

        Encodes text on the fly. For a precomputed cache use answer_cached.
        """
        if head not in ("choice", "noul", "score"):
            raise ValueError(f"unknown head: {head}")
        if options is not None and head != "choice":
            raise ValueError("options are only supported by the Choice head")
        if self.text_encoder is None and (questions is not None or options is not None):
            raise ValueError("questions/options require a text encoder")

        batch = features[0].shape[0]
        q_tokens = q_mask = option_embeddings = None
        if self.text_encoder is not None and questions is not None:
            if len(questions) not in (1, batch):
                raise ValueError("supply one question or one per batch item")
            if len(questions) == 1:
                questions = questions * batch
            q_tokens, q_mask = self.text_encoder.token_states(questions)
        if options is not None:
            if not 2 <= len(options) <= 255:
                raise ValueError("Choice requires 2 to 255 options")
            assert self.text_encoder is not None
            option_embeddings = self.text_encoder(options)
        return self.answer_cached(
            features, q_tokens, q_mask, option_embeddings, head=head
        )

    def answer_cached(
        self,
        features: tuple[Tensor, Tensor],
        q_tokens: Tensor | None = None,
        q_mask: Tensor | None = None,
        option_embeddings: Tensor | None = None,
        *,
        head: Head = "choice",
    ) -> Tensor:
        """Answer from precomputed frozen-encoder outputs; no encoder is called.

        features is (H_ts [B,M,d], token_mask [B,M] bool True=valid). q_tokens
        [B,L,d_txt] with q_mask [B,L] are MiniLM token states for the questions;
        option_embeddings [n,d_txt] are pooled MiniLM option vectors (pre-projection).
        """
        if head not in ("choice", "noul", "score"):
            raise ValueError(f"unknown head: {head}")
        if option_embeddings is not None and head != "choice":
            raise ValueError("options are only supported by the Choice head")
        if self.text_encoder is None and (q_tokens is not None or option_embeddings is not None):
            raise ValueError("questions/options require a text encoder")

        signal, signal_mask = features
        signal = signal.to(dtype=next(self.parameters()).dtype)
        if self.text_encoder is not None and q_tokens is None:
            q_tokens = signal.new_zeros(signal.shape[0], 1, self.text_encoder.hidden_size)
            q_mask = torch.ones(signal.shape[0], 1, dtype=torch.bool, device=signal.device)
        return self._decode(signal, signal_mask, q_tokens, q_mask, option_embeddings, head)

    def _decode(
        self,
        signal: Tensor,
        signal_mask: Tensor,
        q_tokens: Tensor | None,
        q_mask: Tensor | None,
        option_embeddings: Tensor | None,
        head: Head,
    ) -> Tensor:
        """Trainable tail shared by answer and answer_cached: fusion + heads."""
        if self.text_encoder is not None:
            assert isinstance(self.fusion, CrossAttentionFusion) and q_tokens is not None
            fused = self.fusion(signal, signal_mask, q_tokens.to(signal.dtype), q_mask)
        else:
            weights = signal_mask.unsqueeze(-1).to(signal.dtype)
            pooled = (signal * weights).sum(1) / weights.sum(1).clamp_min(1)
            fused = self.fusion(pooled)  # type: ignore[arg-type]

        if head == "noul":
            return self.noul(fused).squeeze(-1)
        if head == "score":
            return self.score(fused)
        if option_embeddings is not None:
            assert self.option_projection is not None
            keys = self.option_projection(option_embeddings.to(fused))
        elif self.classes is not None:
            keys = self.classes.weight
        else:
            raise ValueError("Choice requires options or num_classes")
        logits = (fused @ keys.T) / fused.shape[-1] ** 0.5
        if self.option_attention is not None and option_embeddings is not None:
            logits = logits + self.option_attention(fused, keys, signal, signal_mask)
        return logits
