import pytest
import torch
from torch.nn import functional as F

from justin.training.loss import compute_loss, ranked_probability_score


def test_ranked_probability_score_matches_hand_calculation():
    # Prediction [0.2, 0.3, 0.5], target level 1:
    # CDF errors at the two thresholds are 0.2 and -0.5.
    logits = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64).log()
    assert ranked_probability_score(logits, torch.tensor([1])).item() == pytest.approx(0.145)


def test_distant_errors_cost_more_with_identical_cross_entropy():
    target = torch.tensor([0])
    near = torch.tensor([[0.1, 0.8, 0.1]]).log()
    far = torch.tensor([[0.1, 0.1, 0.8]]).log()
    torch.testing.assert_close(F.cross_entropy(near, target), F.cross_entropy(far, target))
    assert compute_loss(far, target, "score") > compute_loss(near, target, "score")


def test_ordinal_loss_limits():
    target = torch.tensor([0])
    assert ranked_probability_score(torch.tensor([[100.0, -100, -100]]), target) == 0
    assert ranked_probability_score(torch.tensor([[-100.0, -100, 100]]), target) == 1
    assert ranked_probability_score(torch.tensor([[-100.0, 100, -100]]), target) == 0.5


def test_two_level_rps_equals_binary_brier():
    probability = torch.tensor([0.2, 0.8, 0.6])
    logits = torch.stack([1 - probability, probability], dim=-1).log()
    target = torch.tensor([0, 1, 0])
    torch.testing.assert_close(
        ranked_probability_score(logits, target), (probability - target).square().mean()
    )


def test_score_combines_ce_and_rps_with_correct_gradients():
    logits = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0, 1, 3])
    torch.testing.assert_close(
        compute_loss(logits, target, "score"),
        F.cross_entropy(logits, target) + ranked_probability_score(logits, target),
    )
    assert torch.autograd.gradcheck(lambda z: compute_loss(z, target, "score"), (logits,))


def test_choice_and_noul_losses_unchanged():
    target = torch.tensor([0, 1])
    choice = torch.randn(2, 3)
    binary = torch.randn(2)
    torch.testing.assert_close(
        compute_loss(choice, target, "choice"), F.cross_entropy(choice, target)
    )
    torch.testing.assert_close(
        compute_loss(binary, target, "noul"),
        F.binary_cross_entropy_with_logits(binary, target.float()),
    )


@pytest.mark.parametrize(
    "logits,target",
    [
        (torch.zeros(2), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(0, 3), torch.zeros(0, dtype=torch.long)),
        (torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(2, 3), torch.zeros(2, 1, dtype=torch.long)),
        (torch.zeros(2, 3), torch.zeros(2)),
    ],
)
def test_invalid_ordinal_shapes_or_target_type(logits, target):
    with pytest.raises(ValueError, match="Score"):
        ranked_probability_score(logits, target)


@pytest.mark.parametrize("target", [-1, 3])
def test_invalid_ordinal_level(target):
    with pytest.raises(RuntimeError):
        ranked_probability_score(torch.zeros(1, 3), torch.tensor([target]))
