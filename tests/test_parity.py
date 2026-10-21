"""Parity with the training package: the fixtures `scripts/make_rust_fixtures.py` writes (in the
training repository) hold inputs and the full package's outputs on CPU float32. These tests run
the same inputs through this package. Point JUSTIN_FIXTURES at the fixture folder (default: the
Rust crate's, ../justin_rs/tests/fixtures); without fixtures the tests skip."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

from justin import Answer, Found, Justin, Location, Question
from justin.features import prepare

FIXTURES = Path(os.environ.get("JUSTIN_FIXTURES", Path(__file__).parents[2] / "justin_rs" / "tests" / "fixtures"))
pytestmark = pytest.mark.skipif(not (FIXTURES / "models.json").is_file(), reason=f"no fixtures in {FIXTURES}")


def values(items):
    if isinstance(items, list):
        return [values(v) for v in items]
    return math.nan if items is None else items


def baseline(value):
    return tuple(value) if isinstance(value, list) else value


def query(item) -> Question | Location:
    answers = item["answers"]
    if answers is None:
        return Location(item["question"])
    return Question(item["question"], answers["options"] if isinstance(answers, dict) else answers)


def test_features_match_the_training_package():
    for case in json.loads((FIXTURES / "features.json").read_text()):
        p = prepare(
            torch.tensor(values(case["series"])).numpy(), case["question"], case["options"], case["names"],
            step=case["step"], baseline=baseline(case["baseline"]),
        )  # fmt: skip
        expected = case["expected"]
        assert p.question == expected["question"]
        for actual, wanted in [
            (p.series, expected["series"]), (p.channel_features, expected["channel_features"]),
            (p.referent_question, expected["referent_question"]), (p.referent_options, expected["referent_options"]),
        ]:  # fmt: skip
            torch.testing.assert_close(actual, torch.tensor(values(wanted)).reshape(actual.shape), rtol=0, atol=0)


def test_justin_matches_the_training_package_end_to_end():
    fixture = json.loads((FIXTURES / "models.json").read_text())
    models = {}
    for name, files in fixture["models"].items():
        if Path(files["export"]).is_file():
            models[name] = Justin.from_pretrained(files["export"], device="cpu", location=files["location"])
    worst, count = 0.0, 0
    for case in fixture["cases"]:
        model = models.get(case["model"])
        if model is None:
            continue
        outputs = model.ask(
            values(case["series"]), [query(q) for q in case["queries"]], case["channel_names"],
            step=case["step"], baseline=baseline(case["baseline"]),
        )  # fmt: skip
        for output, expected in zip(outputs, case["expected"], strict=True):
            logits = torch.tensor(output.logits)
            wanted = torch.tensor(values(expected["logits"]), dtype=torch.float32)
            assert torch.equal(logits.isfinite(), wanted.isfinite()), case["id"]
            finite = wanted.isfinite()
            worst = max(worst, float((logits[finite] - wanted[finite]).abs().max()))
            if isinstance(output, Found):
                assert output.location == expected["location"] or (
                    output.location is not None
                    and {k: v for k, v in output.location.items() if k != "probability"}
                    == {k: v for k, v in expected["location"].items() if k != "probability"}
                ), case["id"]
            else:
                assert isinstance(output, Answer)
                torch.testing.assert_close(
                    torch.tensor(output.probabilities), torch.tensor(expected["probabilities"]), rtol=0, atol=1e-6
                )
            count += 1
    assert count > 100
    print(f"{count} queries, logits max |diff| {worst:.2e}")
    assert worst <= 1e-5


@pytest.mark.skipif(not (FIXTURES / "chronos.json").is_file(), reason="no Chronos-2 fixtures")
def test_the_published_chronos_model_matches_the_training_package():
    fixture = json.loads((FIXTURES / "chronos.json").read_text())
    files = fixture["models"]["chronos"]
    model = Justin.from_pretrained(files["repo"], filename=files["filename"], revision=files["revision"], device="cpu")
    worst = 0.0
    for case in fixture["cases"]:
        outputs = model.ask(values(case["series"]), [query(q) for q in case["queries"]])
        for output, expected in zip(outputs, case["expected"], strict=True):
            worst = max(worst, float((torch.tensor(output.logits) - torch.tensor(expected["logits"])).abs().max()))
            torch.testing.assert_close(
                torch.tensor(output.probabilities), torch.tensor(expected["probabilities"]), rtol=0, atol=1e-6
            )
    print(f"chronos: logits max |diff| {worst:.2e}")
    assert worst <= 1e-5
    with pytest.raises(ValueError, match="no location head"):
        model.locate(values(fixture["cases"][0]["series"]), "Where is the spike?")
