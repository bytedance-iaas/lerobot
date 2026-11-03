# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.gemma.modeling_gemma import eager_attention_forward

from lerobot.policies.pi05.attention import sdpa_attention_forward


@pytest.mark.parametrize("query_length", [13, 5])
def test_sdpa_preserves_masked_gqa_outputs_and_gradients(query_length):
    """Cover joint training and cached inference with different query/key lengths."""
    torch.manual_seed(42)
    query = torch.randn(2, 8, query_length, 16, requires_grad=True)
    key = torch.randn(2, 1, 13, 16, requires_grad=True)
    value = torch.randn(2, 1, 13, 16, requires_grad=True)
    mask = torch.zeros(2, 1, query_length, 13)
    mask[0, :, :, -3:] = -2.3819763e38
    mask[1, :, :2, 7:] = -2.3819763e38
    module = SimpleNamespace(num_key_value_groups=8, training=True)
    actual, _ = sdpa_attention_forward(module, query, key, value, mask, 0.25)
    expected, _ = eager_attention_forward(module, query, key, value, mask, 0.25)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (query, key, value), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (query, key, value), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-5)
