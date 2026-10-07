from __future__ import annotations

import pytest
import torch

from benchmarks.correctness import CORRECTNESS_CASES, AttentionCase, run_case

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires an NVIDIA CUDA GPU"
    ),
]
@pytest.mark.parametrize("case", CORRECTNESS_CASES, ids=lambda case: case.name)
def test_attention_matches_torch(case: AttentionCase) -> None:
    run_case(case, seed=1729 + CORRECTNESS_CASES.index(case))
