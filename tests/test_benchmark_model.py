from benchmarks.benchmark_attention import attention_flops, memory_model


def test_causal_forward_counts_only_visible_pairs() -> None:
    assert (
        attention_flops(
            batch=1,
            query_heads=1,
            sequence=4,
            head_dim=8,
            causal=True,
            mode="forward",
        )
        == 4 * 10 * 8
    )


def test_backward_is_two_and_a_half_forward_passes() -> None:
    forward = attention_flops(
        batch=2,
        query_heads=4,
        sequence=16,
        head_dim=32,
        causal=False,
        mode="forward",
    )
    backward = attention_flops(
        batch=2,
        query_heads=4,
        sequence=16,
        head_dim=32,
        causal=False,
        mode="backward",
    )
    assert backward == int(2.5 * forward)


def test_memory_model_is_linear_for_saved_lse() -> None:
    small = memory_model(batch=1, query_heads=1, sequence=128)
    large = memory_model(batch=1, query_heads=1, sequence=256)
    assert large["saved_lse_mib"] == 2 * small["saved_lse_mib"]
    assert large["naive_score_mib"] == 4 * small["naive_score_mib"]
