from __future__ import annotations

import pytest
import torch

from triton_flash_attention import (
    DispatchDecision,
    explain_dispatch,
    scaled_dot_product_attention,
)


@pytest.mark.parametrize("causal", [False, True])
def test_auto_dispatch_fallback_matches_reference(causal: bool) -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 4, 7, 32, requires_grad=True)
    key = torch.randn(2, 4, 7, 32, requires_grad=True)
    value = torch.randn(2, 4, 7, 32, requires_grad=True)
    grad = torch.randn_like(query)

    actual = scaled_dot_product_attention(query, key, value, is_causal=causal)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=causal
    )
    torch.testing.assert_close(actual, expected)

    actual_grads = torch.autograd.grad(actual, (query, key, value), grad)
    expected_grads = torch.autograd.grad(expected, (query, key, value), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gqa_fallback_matches_reference() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 4, 11, 32)
    key = torch.randn(1, 2, 11, 32)
    value = torch.randn(1, 2, 11, 32)
    actual = scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )
    torch.testing.assert_close(actual, expected)


def test_dispatch_explains_ineligible_device() -> None:
    tensor = torch.randn(1, 1, 3, 32)
    assert explain_dispatch(tensor, tensor, tensor) == DispatchDecision(
        backend="torch",
        reason="the Triton kernel requires a CUDA tensor",
    )


def test_strict_triton_backend_rejects_ineligible_device() -> None:
    tensor = torch.randn(1, 1, 3, 32)
    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        scaled_dot_product_attention(tensor, tensor, tensor, backend="triton")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda q, k, v: (q.squeeze(0), k, v),
            "query must have shape",
        ),
        (
            lambda q, k, v: (q[..., :-1], k, v),
            "query and key head dimensions",
        ),
        (
            lambda q, k, v: (q, k[:, :1], v[:, :1]),
            "pass enable_gqa=True",
        ),
    ],
)
def test_validation_errors(mutator, message: str) -> None:
    query = torch.randn(1, 2, 5, 32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    query, key, value = mutator(query, key, value)
    with pytest.raises(ValueError, match=message):
        scaled_dot_product_attention(query, key, value)


def test_explicit_torch_backend_honors_custom_scale() -> None:
    query = torch.randn(1, 2, 5, 32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    actual = scaled_dot_product_attention(
        query, key, value, scale=0.25, backend="torch"
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, scale=0.25
    )
    torch.testing.assert_close(actual, expected)


def test_cross_attention_fallback_matches_reference() -> None:
    query = torch.randn(1, 2, 3, 32)
    key = torch.randn(1, 2, 5, 32)
    value = torch.randn(1, 2, 5, 16)
    decision = explain_dispatch(query, key, value)
    assert decision.backend == "torch"
    assert "self-attention" in decision.reason
    actual = scaled_dot_product_attention(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(actual, expected)
