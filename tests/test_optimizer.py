from src.config import Config
from src.model import SystemOne
from src.training.optimizer import build_optimizer


def test_optimizer_single_group_without_lora(make_chronos2_encoder, text_encoder):
    model = SystemOne(make_chronos2_encoder(), text_encoder, num_classes=3, hidden_dim=16)
    optimizer = build_optimizer(model, Config(learning_rate=0.01, weight_decay=0.02))
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 0.02


def test_optimizer_splits_lora_into_its_own_group(make_chronos2_encoder, text_encoder):
    model = SystemOne(
        make_chronos2_encoder(), text_encoder,
        num_classes=3, hidden_dim=16, series_lora_layers=1, minilm_lora=True,
        lora_rank=4, lora_alpha=8,
    )
    config = Config(learning_rate=0.01, lora_learning_rate=0.0001, weight_decay=0.02)
    optimizer = build_optimizer(model, config)
    assert len(optimizer.param_groups) == 2
    head, adapters = optimizer.param_groups
    assert head["lr"] == 0.01 and head["weight_decay"] == 0.02
    assert adapters["lr"] == 0.0001 and adapters["weight_decay"] == 0.0
    # Every adapter param is trainable and lives only in the adapter group.
    assert all(p.requires_grad for p in adapters["params"])
    assert len(adapters["params"]) > 0
    head_ids = {id(p) for p in head["params"]}
    assert not any(id(p) in head_ids for p in adapters["params"])
