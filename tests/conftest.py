import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from justin.encoders import TextEncoder, TimestralEncoder
from justin.timestral import Timestral, TimestralConfig


class TinyTimestral(TimestralEncoder):
    name = "timestral5m"  # so exports of it name a supported encoder
    repo = "test/tiny-timestral"
    revision = "0" * 40


@pytest.fixture
def series_encoder():
    torch.manual_seed(0)
    config = TimestralConfig(
        patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=24, context_length=64,
        block_schedule=("time", "time_variate", "time", "time_variate"),
    )  # fmt: skip
    return TinyTimestral(Timestral(config))


@pytest.fixture
def text_encoder(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "is", "this", "rising", "falling",
             "flat", "yes", "no", "?", "cpu", "memory", "where", "spike"]  # fmt: skip
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    torch.manual_seed(0)
    backbone = BertModel(
        BertConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=24,
                   num_hidden_layers=1, num_attention_heads=2)  # fmt: skip
    )
    return TextEncoder(backbone, BertTokenizer(vocab=str(path)))
