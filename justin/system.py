"""SystemOne: the series and text encoders, question conditioning, fusion and the heads."""

from typing import Literal

import torch
from torch import Tensor, nn

from .encoders import SeriesEncoder, TextEncoder
from .fusion import ChannelHead, CrossAttentionFusion, OptionAttention
from .lora import inject_lora
from .referents import OPTION, QUESTION

Head = Literal["choice", "noul", "score"]


def channel_text(name: str) -> str:
    """A series name such as ``cartservice_latency-50`` as text for the text encoder."""
    return name.replace("_", " ").replace("-", " ")


class SystemOne(nn.Module):
    """Series encoder (Timestral or Chronos-2) + MiniLM + cross-attention fusion + task heads.

    Returns raw logits: Choice [B, N], Noul [B], Score [B, K]. Apply softmax or sigmoid to
    logits / temperature for probabilities. Score levels are a fixed rubric indexed 0..K-1.
    Parameter names match the training code, so exports load unchanged.
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
        channel_names: bool = False,
        channel_features: int = 0,
        channel_head: bool = False,
        referents: bool = False,
        series_condition: int = 0,
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
        if channel_names and text_encoder is None:
            raise ValueError("channel_names requires a text encoder")
        # Adds an embedding of each series' name to that series' tokens, so questions and
        # options can refer to channels ("is cartservice abnormal?").
        self.channel_projection = nn.Linear(text_dim, signal_dim) if channel_names else None
        self._name_embeddings: dict[str, Tensor] = {}
        # Per-channel numbers supplied with the data (e.g. how far a channel deviates), added the
        # same way; see data/deviation.py for why the encoders cannot provide them.
        self.feature_projection = (
            nn.Sequential(nn.Linear(channel_features, 64), nn.GELU(), nn.Linear(64, signal_dim))
            if channel_features
            else None
        )
        if channel_head and not (channel_names or channel_features):
            raise ValueError("channel_head requires channel_names or channel_features")
        # Per-channel tokens that the question and the options read; see ChannelHead.
        self.channel_head = (
            ChannelHead(
                signal_dim, text_dim, hidden_dim,
                n_features=channel_features, names=channel_names, dropout=dropout,
            )  # fmt: skip
            if channel_head
            else None
        )
        self.channel_options = (
            OptionAttention(hidden_dim, hidden_dim, dropout=dropout) if channel_head else None
        )
        # Features of the channels a question or an option names (data/referents.py): the
        # question's go into the fused vector, each option's score reads its own.
        if referents and text_encoder is None:
            raise ValueError("referents require a text encoder")
        self.referent_question = (
            nn.Sequential(nn.Linear(QUESTION, 64), nn.GELU(), nn.Linear(64, hidden_dim))
            if referents
            else None
        )
        self.referent_option = nn.Linear(OPTION, hidden_dim) if referents else None
        self.referent_fuse = nn.Linear(hidden_dim, hidden_dim) if referents else None
        # The options attend to each other (no positions, so their order does not matter): a
        # candidate stands out, or is a victim of a bigger one, only in comparison with the rest.
        self.referent_mixer = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden_dim, 8, 4 * hidden_dim, dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                ),
                2,
                enable_nested_tensor=False,
            )
            if referents
            else None
        )  # fmt: skip
        self.referent_norm = nn.LayerNorm(hidden_dim) if referents else None
        self.referent_score = nn.Linear(hidden_dim, 1) if referents else None
        if option_attention and text_encoder is None:
            raise ValueError("option_attention requires a text encoder")
        self.option_attention = (
            OptionAttention(signal_dim, hidden_dim, dropout=dropout) if option_attention else None
        )
        self.classes = nn.Embedding(num_classes, hidden_dim) if num_classes is not None else None
        self.noul = nn.Linear(hidden_dim, 1)
        self.score = nn.Linear(hidden_dim, score_levels)

        # The question steers the series encoder: one linear layer per leading block turns the
        # question's sentence embedding into a (scale, shift) applied to the states entering that
        # block (FiLM). Zero-initialized, so training starts from the unconditioned encoder.
        self.series_condition = series_condition
        self.condition_film: nn.ModuleList | None = None
        if series_condition:
            if text_encoder is None:
                raise ValueError("series_condition requires a text encoder")
            if not series_encoder.supports_film:
                raise ValueError(f"{type(series_encoder).__name__} cannot be conditioned")
            if not 1 <= series_condition <= series_encoder.num_blocks:
                raise ValueError(f"series_condition must be in [1, {series_encoder.num_blocks}]")
            self.condition_film = nn.ModuleList(
                nn.Linear(text_dim, 2 * signal_dim) for _ in range(series_condition)
            )
            for layer in self.condition_film:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
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

    def forward(
        self,
        series: Tensor,
        questions: list[str] | None = None,
        *,
        options: list[str] | None = None,
        head: Head = "choice",
        padding_mask: Tensor | None = None,
        channel_names: list[list[str]] | None = None,
        channel_features: Tensor | None = None,
        referent_question: Tensor | None = None,
        referent_options: Tensor | None = None,
    ) -> Tensor:
        """Series is [B, C, T]; options are shared across the batch.

        Supply one shared question or B questions. Omit questions/options
        for a fixed catalog. Include relevant metadata in question text.
        `channel_names` holds one name per series for each batch item; `channel_features`
        is [B, C, n_features].
        """
        features = self.series_encoder(series, padding_mask, **self._conditioning(questions, series))
        return self.answer(
            features, questions, options=options, head=head, channel_names=channel_names,
            channel_features=channel_features, referent_question=referent_question,
            referent_options=referent_options,
        )  # fmt: skip

    def _conditioning(self, questions: list[str] | None, series: Tensor) -> dict:
        """The ``film`` argument for the series encoder: per-block (scale, shift) from the questions."""
        if self.condition_film is None:
            return {}
        assert self.text_encoder is not None
        if questions is None:
            raise ValueError("this model conditions the series encoder on the question: pass questions")
        batch = series.shape[0]
        if len(questions) not in (1, batch):
            raise ValueError("supply one question or one per batch item")
        embedding = self.text_encoder(questions * batch if len(questions) == 1 else questions)
        embedding = embedding.to(dtype=next(self.condition_film.parameters()).dtype)
        return {"film": [layer(embedding).chunk(2, dim=-1) for layer in self.condition_film]}

    def channel_embeddings(self, names: list[list[str]]) -> Tensor:
        """Text-encoder embeddings of series names, [B, C, d_txt], memoized per name."""
        assert self.text_encoder is not None
        memo = self._name_embeddings
        missing = sorted({name for row in names for name in row} - memo.keys())
        if missing:
            embedded = self.text_encoder([channel_text(name) for name in missing])
            memo.update({name: vector.detach() for name, vector in zip(missing, embedded, strict=True)})
        return torch.stack([torch.stack([memo[name] for name in row]) for row in names])

    def answer(
        self,
        features: tuple[Tensor, Tensor],
        questions: list[str] | None = None,
        *,
        options: list[str] | None = None,
        head: Head = "choice",
        channel_names: list[list[str]] | None = None,
        channel_features: Tensor | None = None,
        referent_question: Tensor | None = None,
        referent_options: Tensor | None = None,
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
        channel_embeddings = None
        if channel_names is not None:
            if len(channel_names) != batch:
                raise ValueError("supply channel names for every batch item")
            channel_embeddings = self.channel_embeddings(channel_names)
        return self.answer_cached(
            features, q_tokens, q_mask, option_embeddings, head=head,
            channel_embeddings=channel_embeddings, channel_features=channel_features,
            referent_question=referent_question, referent_options=referent_options,
        )  # fmt: skip

    def answer_cached(
        self,
        features: tuple[Tensor, Tensor],
        q_tokens: Tensor | None = None,
        q_mask: Tensor | None = None,
        option_embeddings: Tensor | None = None,
        *,
        head: Head = "choice",
        channel_embeddings: Tensor | None = None,
        channel_features: Tensor | None = None,
        referent_question: Tensor | None = None,
        referent_options: Tensor | None = None,
    ) -> Tensor:
        """Answer from precomputed frozen-encoder outputs; no encoder is called.

        features is (H_ts [B,M,d], token_mask [B,M] bool True=valid). q_tokens
        [B,L,d_txt] with q_mask [B,L] are MiniLM token states for the questions;
        option_embeddings [n,d_txt] are pooled MiniLM option vectors (pre-projection);
        channel_embeddings [B,C,d_txt] are pooled name embeddings of the C series, with
        M = C * S tokens laid out channel by channel; channel_features [B,C,F] are the
        per-channel numbers the model was built for; referent_question [B,QUESTION] and
        referent_options [B,n,OPTION] describe the channels the question and options name.
        """
        if head not in ("choice", "noul", "score"):
            raise ValueError(f"unknown head: {head}")
        if option_embeddings is not None and head != "choice":
            raise ValueError("options are only supported by the Choice head")
        if self.text_encoder is None and (q_tokens is not None or option_embeddings is not None):
            raise ValueError("questions/options require a text encoder")

        signal, signal_mask = features
        signal = signal.to(dtype=next(self.parameters()).dtype)
        if channel_embeddings is not None and self.channel_projection is None:
            raise ValueError("this model was built without channel_names")
        if channel_features is not None and self.feature_projection is None:
            raise ValueError("this model was built without channel_features")
        if self.feature_projection is not None and channel_features is None:
            raise ValueError("this model needs channel_features")
        bias = None
        if channel_embeddings is not None:
            bias = self.channel_projection(channel_embeddings.to(signal))  # [B, C, d]
        if channel_features is not None:
            extra = self.feature_projection(channel_features.to(signal))
            bias = extra if bias is None else bias + extra
        if bias is not None:
            batch, tokens, dim = signal.shape
            channels = bias.shape[1]
            if tokens % channels:
                raise ValueError("signal tokens must divide evenly among the channels")
            signal = (signal.view(batch, channels, tokens // channels, dim) + bias[:, :, None]).reshape(
                batch, tokens, dim
            )
        if self.text_encoder is not None and q_tokens is None:
            q_tokens = signal.new_zeros(signal.shape[0], 1, self.text_encoder.hidden_size)
            q_mask = torch.ones(signal.shape[0], 1, dtype=torch.bool, device=signal.device)
        if (referent_question is None) != (self.referent_question is None):
            raise ValueError(
                "this model needs referent_question" if self.referent_question is not None
                else "this model was built without referents"
            )  # fmt: skip
        referent_state = None
        if self.referent_question is not None:
            if referent_options is None:
                referent_options = referent_question.new_zeros(referent_question.shape[0], 0, OPTION)
            referent_state = (referent_question.to(signal), referent_options.to(signal))
        channel_state = None
        if self.channel_head is not None:
            if channel_embeddings is None and self.channel_head.names is not None:
                raise ValueError("this model needs channel_names")
            if channel_features is None and self.channel_head.features is not None:
                raise ValueError("this model needs channel_features")
            assert q_tokens is not None
            channel_state = (
                bias.shape[1] if bias is not None else None,
                channel_embeddings,
                channel_features,
            )
        return self._decode(
            signal, signal_mask, q_tokens, q_mask, option_embeddings, head, channel_state,
            referent_state,
        )  # fmt: skip

    def _decode(
        self,
        signal: Tensor,
        signal_mask: Tensor,
        q_tokens: Tensor | None,
        q_mask: Tensor | None,
        option_embeddings: Tensor | None,
        head: Head,
        channel_state: tuple | None = None,
        referent_state: tuple | None = None,
    ) -> Tensor:
        """Trainable tail shared by answer and answer_cached: fusion + heads."""
        if self.text_encoder is not None:
            assert isinstance(self.fusion, CrossAttentionFusion) and q_tokens is not None
            fused = self.fusion(signal, signal_mask, q_tokens.to(signal.dtype), q_mask)
        else:
            weights = signal_mask.unsqueeze(-1).to(signal.dtype)
            pooled = (signal * weights).sum(1) / weights.sum(1).clamp_min(1)
            fused = self.fusion(pooled)  # type: ignore[arg-type]
        channel_tokens = channel_valid = None
        if channel_state is not None:
            assert self.channel_head is not None and q_tokens is not None and q_mask is not None
            channels, names, features = channel_state
            summary, channel_tokens, channel_valid = self.channel_head(
                signal, signal_mask, channels, q_tokens.to(signal.dtype), q_mask, names, features
            )
            fused = fused + summary
        if referent_state is not None:
            fused = fused + self.referent_question(referent_state[0])

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
        if referent_state is not None and referent_state[1].shape[1] == logits.shape[1]:
            tokens = self.referent_fuse(fused)[:, None] + self.referent_option(referent_state[1])
            mixed = self.referent_mixer(tokens)
            logits = logits + self.referent_score(self.referent_norm(mixed)).squeeze(-1)
        if self.option_attention is not None and option_embeddings is not None:
            logits = logits + self.option_attention(fused, keys, signal, signal_mask)
        if self.channel_options is not None and option_embeddings is not None:
            logits = logits + self.channel_options(fused, keys, channel_tokens, channel_valid)
        return logits
