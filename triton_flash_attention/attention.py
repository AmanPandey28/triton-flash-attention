"""FlashAttention-2 kernels and a production-style PyTorch dispatcher.

The Triton path implements fused self-attention without materializing the
quadratic score matrix.  ``scaled_dot_product_attention`` is the public API:
it selects this kernel for supported CUDA inputs and safely falls back to
PyTorch SDPA otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch
import triton.language as tl

import triton

_INV_LN2 = tl.constexpr(1.4426950408889634)
_SUPPORTED_HEAD_DIMS = (32, 64, 128)
Backend = Literal["auto", "triton", "torch"]


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_block_ptr,
    v_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    """Stream K/V tiles while maintaining an online-softmax state."""
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    k_block_ptr = tl.advance(k_block_ptr, (0, lo))
    v_block_ptr = tl.advance(v_block_ptr, (lo, 0))

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_offsets = start_n + offs_n
        valid_keys = key_offsets < N_CTX

        k = tl.load(k_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero")
        qk = tl.dot(q, k) * qk_scale

        valid = valid_keys[None, :]
        if STAGE == 2:
            valid = valid & (offs_m[:, None] >= key_offsets[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc=acc)
        m_i = m_ij

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


_FWD_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=4, num_warps=8),
]


@triton.autotune(configs=_FWD_CONFIGS, key=["N_CTX", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    LSE,
    Out,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    stride_o_batch,
    stride_o_head,
    stride_o_seq,
    stride_o_dim,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """FlashAttention-2 forward pass with boundary-safe GQA indexing."""
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)
    q_batch_head = tl.program_id(1)
    batch = q_batch_head // NUM_Q_HEADS
    q_head = q_batch_head % NUM_Q_HEADS
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = q_head // group_size

    q_offset = batch.to(tl.int64) * stride_q_batch + q_head * stride_q_head
    k_offset = batch.to(tl.int64) * stride_k_batch + kv_head * stride_k_head
    v_offset = batch.to(tl.int64) * stride_v_batch + kv_head * stride_v_head
    o_offset = batch.to(tl.int64) * stride_o_batch + q_head * stride_o_head

    q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_q_seq, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_k_dim, stride_k_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_v_seq, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_o_seq, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    q = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")
    qk_scale = sm_scale * _INV_LN2

    if CAUSAL:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            start_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            1,
            offs_m,
            offs_n,
            N_CTX,
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            start_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            start_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            3,
            offs_m,
            offs_n,
            N_CTX,
        )

    lse = m_i + tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    valid_queries = offs_m < N_CTX
    tl.store(LSE + q_batch_head * N_CTX + offs_m, lse, mask=valid_queries)
    tl.store(o_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_preprocess(
    Out,
    dO,
    Delta,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute Delta_i = rowsum(O_i * dO_i) once per query row."""
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    batch_head = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)
    valid = row < N_CTX
    offsets = batch_head * N_CTX * HEAD_DIM + row[:, None] * HEAD_DIM + dim[None, :]
    o = tl.load(Out + offsets, mask=valid[:, None], other=0.0)
    do = tl.load(dO + offsets, mask=valid[:, None], other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(Delta + batch_head * N_CTX + row, delta, mask=valid)


@triton.jit
def _attn_bwd_dk_dv_inner(
    k_scaled,
    v,
    dk,
    dv,
    Q,
    dO,
    LSE,
    Delta,
    start_q,
    start_kv,
    num_steps,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    MASK_DIAGONAL: tl.constexpr,
):
    q_offsets = start_q + tl.arange(0, BLOCK_Q)
    kv_offsets = start_kv + tl.arange(0, BLOCK_KV)
    dim = tl.arange(0, HEAD_DIM)

    for _ in tl.range(0, num_steps):
        valid_q = q_offsets < N_CTX
        q_t = tl.load(
            Q + dim[:, None] + q_offsets[None, :] * HEAD_DIM,
            mask=valid_q[None, :],
            other=0.0,
        )
        do = tl.load(
            dO + q_offsets[:, None] * HEAD_DIM + dim[None, :],
            mask=valid_q[:, None],
            other=0.0,
        )
        lse = tl.load(LSE + q_offsets, mask=valid_q, other=0.0)
        delta = tl.load(Delta + q_offsets, mask=valid_q, other=0.0)

        scores_t = tl.dot(k_scaled, q_t)
        p_t = tl.math.exp2(scores_t - lse[None, :])
        p_t = tl.where(valid_q[None, :], p_t, 0.0)
        if MASK_DIAGONAL:
            p_t = tl.where(kv_offsets[:, None] <= q_offsets[None, :], p_t, 0.0)

        dv = tl.dot(p_t.to(tl.float16), do, acc=dv)
        dp_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (dp_t - delta[None, :]) * 0.6931471824645996
        dk = tl.dot(ds_t.to(tl.float16), tl.trans(q_t), acc=dk)
        q_offsets += BLOCK_Q

    return dk, dv


@triton.jit
def _attn_bwd_dq_inner(
    dq,
    q_scaled,
    do,
    lse,
    Delta,
    K,
    V,
    start_q,
    start_kv,
    num_steps,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    MASK_DIAGONAL: tl.constexpr,
):
    q_offsets = start_q + tl.arange(0, BLOCK_Q)
    kv_offsets = start_kv + tl.arange(0, BLOCK_KV)
    dim = tl.arange(0, HEAD_DIM)
    delta = tl.load(Delta + q_offsets, mask=q_offsets < N_CTX, other=0.0)

    for _ in tl.range(0, num_steps):
        valid_kv = kv_offsets < N_CTX
        kv_t_offsets = dim[:, None] + kv_offsets[None, :] * HEAD_DIM
        k_t = tl.load(K + kv_t_offsets, mask=valid_kv[None, :], other=0.0)
        v_t = tl.load(V + kv_t_offsets, mask=valid_kv[None, :], other=0.0)
        scores = tl.dot(q_scaled, k_t)
        p = tl.math.exp2(scores - lse)
        p = tl.where(valid_kv[None, :], p, 0.0)
        if MASK_DIAGONAL:
            p = tl.where(q_offsets[:, None] >= kv_offsets[None, :], p, 0.0)

        dp = tl.dot(do, v_t)
        ds = p * (dp - delta[:, None]) * 0.6931471824645996
        dq = tl.dot(ds.to(tl.float16), tl.trans(k_t), acc=dq)
        kv_offsets += BLOCK_KV

    return dq


_BWD_CONFIGS = [
    triton.Config(
        {"BLOCK_MACRO": 64, "BLOCK_MICRO": 16},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_MACRO": 64, "BLOCK_MICRO": 32},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_MACRO": 128, "BLOCK_MICRO": 16},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_MACRO": 128, "BLOCK_MICRO": 32},
        num_stages=3,
        num_warps=8,
    ),
]


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
    reset_to_zero=["dK_ptr", "dV_ptr"],
)
@triton.jit
def _attn_bwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    LSE_ptr,
    Delta_ptr,
    sm_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_MACRO: tl.constexpr,
    BLOCK_MICRO: tl.constexpr,
):
    """Backward pass that never visits the masked causal triangle."""
    tl.static_assert(BLOCK_MACRO % BLOCK_MICRO == 0)

    pid = tl.program_id(0)
    q_batch_head = tl.program_id(1)
    batch = q_batch_head // NUM_Q_HEADS
    q_head = q_batch_head % NUM_Q_HEADS
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = q_head // group_size

    q_head_offset = ((batch * NUM_Q_HEADS + q_head) * N_CTX * HEAD_DIM).to(tl.int64)
    kv_head_offset = ((batch * NUM_KV_HEADS + kv_head) * N_CTX * HEAD_DIM).to(tl.int64)
    aux_offset = (batch * NUM_Q_HEADS + q_head) * N_CTX

    Q = Q_ptr + q_head_offset
    K = K_ptr + kv_head_offset
    V = V_ptr + kv_head_offset
    dO = dO_ptr + q_head_offset
    dQ = dQ_ptr + q_head_offset
    dK = dK_ptr + kv_head_offset
    dV = dV_ptr + kv_head_offset
    LSE = LSE_ptr + aux_offset
    Delta = Delta_ptr + aux_offset

    dim = tl.arange(0, HEAD_DIM)

    # dK and dV: keep a macro K/V tile resident and sweep only contributing Q.
    start_kv = pid * BLOCK_MACRO
    kv_offsets = start_kv + tl.arange(0, BLOCK_MACRO)
    valid_kv = kv_offsets < N_CTX
    kv_matrix_offsets = kv_offsets[:, None] * HEAD_DIM + dim[None, :]
    k = tl.load(K + kv_matrix_offsets, mask=valid_kv[:, None], other=0.0)
    v = tl.load(V + kv_matrix_offsets, mask=valid_kv[:, None], other=0.0)
    k_scaled = (k * (sm_scale * _INV_LN2)).to(tl.float16)
    dk = tl.zeros([BLOCK_MACRO, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_MACRO, HEAD_DIM], tl.float32)

    if CAUSAL:
        dk, dv = _attn_bwd_dk_dv_inner(
            k_scaled,
            v,
            dk,
            dv,
            Q,
            dO,
            LSE,
            Delta,
            start_kv,
            start_kv,
            BLOCK_MACRO // BLOCK_MICRO,
            N_CTX,
            HEAD_DIM,
            BLOCK_MICRO,
            BLOCK_MACRO,
            True,
        )
        start_q_off_diagonal = start_kv + BLOCK_MACRO
        padded_n = tl.cdiv(N_CTX, BLOCK_MACRO) * BLOCK_MACRO
        remaining_steps = (padded_n - start_q_off_diagonal) // BLOCK_MICRO
        dk, dv = _attn_bwd_dk_dv_inner(
            k_scaled,
            v,
            dk,
            dv,
            Q,
            dO,
            LSE,
            Delta,
            start_q_off_diagonal,
            start_kv,
            remaining_steps,
            N_CTX,
            HEAD_DIM,
            BLOCK_MICRO,
            BLOCK_MACRO,
            False,
        )
    else:
        dk, dv = _attn_bwd_dk_dv_inner(
            k_scaled,
            v,
            dk,
            dv,
            Q,
            dO,
            LSE,
            Delta,
            0,
            start_kv,
            tl.cdiv(N_CTX, BLOCK_MICRO),
            N_CTX,
            HEAD_DIM,
            BLOCK_MICRO,
            BLOCK_MACRO,
            False,
        )

    dk *= sm_scale * _INV_LN2
    if NUM_Q_HEADS == NUM_KV_HEADS:
        tl.store(dK + kv_matrix_offsets, dk, mask=valid_kv[:, None])
        tl.store(dV + kv_matrix_offsets, dv, mask=valid_kv[:, None])
    else:
        tl.atomic_add(dK + kv_matrix_offsets, dk, mask=valid_kv[:, None])
        tl.atomic_add(dV + kv_matrix_offsets, dv, mask=valid_kv[:, None])

    # dQ: keep a macro Q tile resident and sweep only visible K/V tiles.
    start_q = pid * BLOCK_MACRO
    q_offsets = start_q + tl.arange(0, BLOCK_MACRO)
    valid_q = q_offsets < N_CTX
    q_matrix_offsets = q_offsets[:, None] * HEAD_DIM + dim[None, :]
    q = tl.load(Q + q_matrix_offsets, mask=valid_q[:, None], other=0.0)
    do = tl.load(dO + q_matrix_offsets, mask=valid_q[:, None], other=0.0)
    lse = tl.load(LSE + q_offsets, mask=valid_q, other=0.0)[:, None]
    q_scaled = (q * (sm_scale * _INV_LN2)).to(tl.float16)
    dq = tl.zeros([BLOCK_MACRO, HEAD_DIM], tl.float32)

    if CAUSAL:
        dq = _attn_bwd_dq_inner(
            dq,
            q_scaled,
            do,
            lse,
            Delta,
            K,
            V,
            start_q,
            start_q,
            BLOCK_MACRO // BLOCK_MICRO,
            N_CTX,
            HEAD_DIM,
            BLOCK_MACRO,
            BLOCK_MICRO,
            True,
        )
        dq = _attn_bwd_dq_inner(
            dq,
            q_scaled,
            do,
            lse,
            Delta,
            K,
            V,
            start_q,
            0,
            start_q // BLOCK_MICRO,
            N_CTX,
            HEAD_DIM,
            BLOCK_MACRO,
            BLOCK_MICRO,
            False,
        )
    else:
        dq = _attn_bwd_dq_inner(
            dq,
            q_scaled,
            do,
            lse,
            Delta,
            K,
            V,
            start_q,
            0,
            tl.cdiv(N_CTX, BLOCK_MICRO),
            N_CTX,
            HEAD_DIM,
            BLOCK_MACRO,
            BLOCK_MICRO,
            False,
        )

    dq *= sm_scale * _INV_LN2
    tl.store(dQ + q_matrix_offsets, dq, mask=valid_q[:, None])


@dataclass(frozen=True)
class DispatchDecision:
    """The selected backend and the concrete reason for the decision."""

    backend: Literal["triton", "torch"]
    reason: str


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    enable_gqa: bool,
) -> None:
    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape (batch, heads, sequence, head_dim); "
                f"got {tuple(tensor.shape)}"
            )
        if any(dimension == 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must all be greater than zero")

    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value batch dimensions must match")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError(
            "key and value batch, head, and sequence dimensions must match"
        )
    if query.shape[3] != key.shape[3]:
        raise ValueError("query and key head dimensions must match")

    q_heads, kv_heads = query.shape[1], key.shape[1]
    if q_heads != kv_heads and not enable_gqa:
        raise ValueError(
            "query has a different number of heads from key/value; pass "
            "enable_gqa=True to use grouped-query attention"
        )
    if q_heads % kv_heads != 0:
        raise ValueError(
            "the number of query heads must be divisible by the number of "
            "key/value heads"
        )


def explain_dispatch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    enable_gqa: bool = False,
) -> DispatchDecision:
    """Explain whether the custom Triton kernel can serve these inputs."""
    _validate_inputs(query, key, value, enable_gqa=enable_gqa)

    if query.shape[2] != key.shape[2]:
        return DispatchDecision(
            "torch", "the custom kernel currently specializes in self-attention"
        )
    if value.shape[3] != query.shape[3]:
        return DispatchDecision(
            "torch", "the custom kernel requires equal query/key and value head sizes"
        )
    if query.device.type != "cuda":
        return DispatchDecision("torch", "the Triton kernel requires a CUDA tensor")
    if torch.version.hip is not None:
        return DispatchDecision(
            "torch", "this project currently targets NVIDIA CUDA, not ROCm"
        )
    if query.dtype != torch.float16:
        return DispatchDecision(
            "torch", "the custom mixed-precision path currently supports float16"
        )
    if query.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        supported = ", ".join(str(dim) for dim in _SUPPORTED_HEAD_DIMS)
        return DispatchDecision(
            "torch", f"head_dim must be one of ({supported}) for the Triton path"
        )
    major, _ = torch.cuda.get_device_capability(query.device)
    if major < 8:
        return DispatchDecision(
            "torch", "the tuned kernel requires an Ampere-or-newer NVIDIA GPU"
        )
    return DispatchDecision(
        "triton",
        "CUDA float16 self-attention with a supported head dimension",
    )


class TritonAttention(torch.autograd.Function):
    """Low-level autograd bridge for the custom forward and backward kernels."""

    @staticmethod
    def forward(ctx, query, key, value, causal, softmax_scale):
        batch, q_heads, sequence, head_dim = query.shape
        kv_heads = key.shape[1]
        output = torch.empty_like(query)
        lse = torch.empty(
            (batch, q_heads, sequence),
            device=query.device,
            dtype=torch.float32,
        )

        def grid(meta):
            return triton.cdiv(sequence, meta["BLOCK_M"]), batch * q_heads

        _attn_fwd[grid](
            Q=query,
            K=key,
            V=value,
            sm_scale=softmax_scale,
            LSE=lse,
            Out=output,
            stride_q_batch=query.stride(0),
            stride_q_head=query.stride(1),
            stride_q_seq=query.stride(2),
            stride_q_dim=query.stride(3),
            stride_k_batch=key.stride(0),
            stride_k_head=key.stride(1),
            stride_k_seq=key.stride(2),
            stride_k_dim=key.stride(3),
            stride_v_batch=value.stride(0),
            stride_v_head=value.stride(1),
            stride_v_seq=value.stride(2),
            stride_v_dim=value.stride(3),
            stride_o_batch=output.stride(0),
            stride_o_head=output.stride(1),
            stride_o_seq=output.stride(2),
            stride_o_dim=output.stride(3),
            NUM_Q_HEADS=q_heads,
            NUM_KV_HEADS=kv_heads,
            N_CTX=sequence,
            HEAD_DIM=head_dim,
            CAUSAL=bool(causal),
        )
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = bool(causal)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        batch, q_heads, sequence, head_dim = query.shape
        kv_heads = key.shape[1]

        delta = torch.empty_like(lse)
        preprocess_grid = (
            triton.cdiv(sequence, 128),
            batch * q_heads,
        )
        _attn_bwd_preprocess[preprocess_grid](
            Out=output,
            dO=grad_output,
            Delta=delta,
            N_CTX=sequence,
            HEAD_DIM=head_dim,
            BLOCK_M=128,
            num_warps=4,
        )

        grad_query = torch.empty_like(query)
        # GQA shares K/V heads, so those gradients are accumulated atomically.
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)

        def grid(meta):
            return triton.cdiv(sequence, meta["BLOCK_MACRO"]), batch * q_heads

        _attn_bwd[grid](
            Q_ptr=query,
            K_ptr=key,
            V_ptr=value,
            dO_ptr=grad_output,
            dQ_ptr=grad_query,
            dK_ptr=grad_key,
            dV_ptr=grad_value,
            LSE_ptr=lse,
            Delta_ptr=delta,
            sm_scale=ctx.softmax_scale,
            NUM_Q_HEADS=q_heads,
            NUM_KV_HEADS=kv_heads,
            N_CTX=sequence,
            HEAD_DIM=head_dim,
            CAUSAL=ctx.causal,
        )
        return grad_query, grad_key, grad_value, None, None


def _torch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    kwargs = {"is_causal": is_causal, "scale": scale}
    if enable_gqa:
        kwargs["enable_gqa"] = True
    return torch.nn.functional.scaled_dot_product_attention(query, key, value, **kwargs)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Compute scaled dot-product attention with safe backend selection.

    The custom self-attention path supports forward/backward, causal and
    bidirectional attention, arbitrary positive sequence lengths, and GQA.
    Cross-attention and unsupported devices/dtypes fall back to PyTorch when
    ``backend="auto"``. Set ``backend="triton"`` to make fallback explicit.
    """
    if backend not in ("auto", "triton", "torch"):
        raise ValueError("backend must be one of: 'auto', 'triton', 'torch'")
    if not isinstance(is_causal, bool):
        raise TypeError("is_causal must be a bool")
    if not isinstance(enable_gqa, bool):
        raise TypeError("enable_gqa must be a bool")
    if scale is not None and (
        not isinstance(scale, Real) or not math.isfinite(float(scale))
    ):
        raise ValueError("scale must be a finite real number or None")
    _validate_inputs(query, key, value, enable_gqa=enable_gqa)

    if backend == "torch":
        return _torch_sdpa(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    decision = explain_dispatch(query, key, value, enable_gqa=enable_gqa)
    if decision.backend == "torch":
        if backend == "triton":
            raise RuntimeError(f"Triton backend is unavailable: {decision.reason}")
        return _torch_sdpa(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    query_c = query.contiguous()
    key_c = key.contiguous()
    value_c = value.contiguous()
    softmax_scale = float(scale) if scale is not None else query.shape[-1] ** -0.5
    return TritonAttention.apply(query_c, key_c, value_c, is_causal, softmax_scale)
