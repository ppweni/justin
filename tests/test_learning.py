"""Learning regression; opt into cached pretrained encoders with JUSTIN_TEST_DEVICE."""

import os

import pytest
import torch

from src.config import Config
from src.data import Sample, TimeSeriesDataset
from src.model import SystemOne
from src.training.mixing import BalancedBatches
from src.training.optimizer import build_optimizer
from src.training.train import train_epoch
from src.training.validation import validate_datasets


def trend_batches(channels, lengths, *, seed, size):
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for name, count in channels.items():
        samples: list[Sample] = []
        for index in range(size):
            target = index % 2
            ramp = torch.linspace(-1, 1, lengths[name]) * (1 if target else -1)
            noise = 0.1 * torch.randn(count, lengths[name], generator=generator)
            samples.append(
                {
                    "series": ramp.expand(count, -1) + noise,
                    "question": "is this rising?",
                    "options": ("falling", "rising") if name in ("har", "tsqa") else (),
                    "target": target,
                    "head": "noul" if name == "ecg" else "choice",
                    "group": str(index),
                }
            )
        loaders[name] = [
            {**TimeSeriesDataset.collate(samples[start : start + 4]), "dataset": name}
            for start in range(0, size, 4)
        ]
        if name == "sleep":
            for batch in loaders[name]:
                batch["head"] = "score"
    return loaders


@pytest.mark.parametrize(
    "pretrained",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not os.environ.get("JUSTIN_TEST_DEVICE"),
                reason="set JUSTIN_TEST_DEVICE=cpu/mps/cuda; pretrained weights must be cached",
            ),
        ),
    ],
    ids=["tiny", "pretrained"],
)
def test_joint_training_learns_signal(make_chronos2_encoder, text_encoder, pretrained):
    channels = {"har": 3, "sleep": 1, "tsqa": 1, "ecg": 12}
    config = Config(device=os.environ.get("JUSTIN_TEST_DEVICE", "cpu") if pretrained else "cpu")
    if pretrained:
        model = SystemOne.from_pretrained(score_levels=2)
        lengths = {"har": 128, "sleep": 1500, "tsqa": 512, "ecg": 1000}
    else:
        model = SystemOne(make_chronos2_encoder(), text_encoder, score_levels=2)
        lengths = dict.fromkeys(channels, 8)
    model.to(config.device)
    train_batches = trend_batches(channels, lengths, seed=7, size=8)
    held_out = trend_batches(channels, lengths, seed=19, size=32)
    before = validate_datasets(model, held_out)
    frozen = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    optimizer = build_optimizer(model, config)
    for epoch in range(200):
        train_epoch(model, BalancedBatches(train_batches), optimizer)
        if (epoch + 1) % 25 == 0:
            print(f"Learning check: {epoch + 1}/200 rounds", flush=True)
    fitted = validate_datasets(model, train_batches)
    after = validate_datasets(model, held_out)
    without_signal = validate_datasets(
        model,
        {
            name: [{**batch, "series": torch.zeros_like(batch["series"])} for batch in batches]
            for name, batches in held_out.items()
        },
    )
    for name in channels:
        print(f"{name}: before={before[name]}, after={after[name]}", flush=True)
    for name in channels:
        assert fitted[name]["accuracy"] >= 0.95
        assert after[name]["accuracy"] >= 0.95
        assert after[name]["loss"] < before[name]["loss"] * 0.3
        assert without_signal[name]["accuracy"] == 0.5
    # Backbone parameters must be bit-identical after training
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert parameter.grad is None
            torch.testing.assert_close(parameter.cpu(), frozen[name], rtol=0, atol=0)
