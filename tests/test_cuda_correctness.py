from __future__ import annotations

import pytest
import torch

from triton_flash_attention import scaled_dot_product_attention

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires an NVIDIA CUDA GPU"
    ),
]


def _compare_forward_and_backward(
    *,
    sequence: int,
    head_dim: int,
    causal: bool,
    query_heads: int = 2,
    kv_heads: int | None = None,
) -> None:
    kv_heads = kv_heads or query_heads
    torch.manual_seed(123)
    query = (
        torch.randn(
            1,
            query_heads,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.5
    ).requires_grad_()
    key = (
        torch.randn(
            1,
            kv_heads,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.5
    ).requires_grad_()
    value = (
        torch.randn(
            1,
            kv_heads,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.5
    ).requires_grad_()
    grad_output = torch.randn_like(query) * 0.1
    enable_gqa = query_heads != kv_heads

    actual = scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
        enable_gqa=enable_gqa,
        backend="triton",
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
        enable_gqa=enable_gqa,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)

    actual_grads = torch.autograd.grad(
        actual, (query, key, value), grad_output, retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected, (query, key, value), grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-2, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sequence", [17, 129])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_attention_matches_torch(sequence: int, head_dim: int, causal: bool) -> None:
    _compare_forward_and_backward(sequence=sequence, head_dim=head_dim, causal=causal)


def test_gqa_matches_torch() -> None:
    _compare_forward_and_backward(
        sequence=65,
        head_dim=64,
        causal=True,
        query_heads=4,
        kv_heads=2,
    )
