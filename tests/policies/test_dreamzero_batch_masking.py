"""`has_real_action` masking must put the batch axis on dim 0.

The bug these guard: `keep[:, None]` is `[B, 1]`, and broadcasting right-aligns it onto the
loss's trailing dims, so the batch axis lands on the action horizon. At batch 1 that happens
to give the right answer, so every DreamZero run (all of which used batch 1) passed while
batch 2 raised `size of tensor a (2) must match tensor b (96)`.
"""

import pytest
import torch

from lerobot.policies.dreamzero.wan.action_head import mask_per_sample


@pytest.mark.parametrize("batch", [1, 2, 3, 8])
@pytest.mark.parametrize("trailing", [(96,), (96, 32), (9, 96, 32)])
def test_masking_keeps_every_sample_when_all_flags_are_set(batch, trailing):
    loss = torch.rand(batch, *trailing)
    keep = torch.ones(batch, dtype=torch.bool)
    torch.testing.assert_close(mask_per_sample(loss, keep), loss)


@pytest.mark.parametrize("batch", [2, 3, 8])
def test_masking_zeroes_only_the_samples_whose_flag_is_clear(batch):
    loss = torch.rand(batch, 96, 32) + 1.0  # strictly positive, so a zero is unambiguous
    keep = torch.zeros(batch, dtype=torch.bool)
    keep[0] = True
    out = mask_per_sample(loss, keep)
    torch.testing.assert_close(out[0], loss[0])
    assert out[1:].abs().sum() == 0


def test_masking_is_not_the_right_answer_by_accident_at_batch_one():
    """The regression itself: the old expression only agreed with this one at batch 1."""
    loss = torch.rand(1, 96, 32)
    keep = torch.ones(1, dtype=torch.bool)
    torch.testing.assert_close(keep[:, None].float() * loss, mask_per_sample(loss, keep))

    # ... and at batch 2 the old expression raises, which is what hid behind batch 1.
    loss = torch.rand(2, 96, 32)
    keep = torch.ones(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="must match"):
        _ = keep[:, None].float() * loss
    assert mask_per_sample(loss, keep).shape == loss.shape


def test_masking_rejects_a_flag_count_that_does_not_match_the_batch():
    loss = torch.rand(4, 96, 32)
    with pytest.raises(ValueError, match="2 flags for 4 samples"):
        mask_per_sample(loss, torch.ones(2, dtype=torch.bool))


def test_masking_rejects_a_flag_tensor_that_is_not_one_per_sample():
    loss = torch.rand(4, 96, 32)
    with pytest.raises(ValueError, match="one flag per sample"):
        mask_per_sample(loss, torch.ones(4, 1, dtype=torch.bool))


def test_masking_preserves_the_loss_dtype():
    loss = torch.rand(2, 96, 32, dtype=torch.float64)
    out = mask_per_sample(loss, torch.ones(2, dtype=torch.bool))
    assert out.dtype == torch.float64
