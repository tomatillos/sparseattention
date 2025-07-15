import math

import torch
import triton
import triton.language as tl

from sparseattention.mask_utils import compute_mask_kernel


@triton.jit
def _sparse_inner(
    q_block,
    k, stride_kb, stride_kh, stride_ks, stride_kd,
    v, stride_vb, stride_vh, stride_vs, stride_vd,
    mask, stride_mq, stride_mk,
    kv_ids, stride_kv_q, stride_kv_k,
    kv_len,
    q_idx, b, h,
    SEQLEN_K,
    m, l, acc,
    sm_scale: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MASK: tl.constexpr,
):

    k_offset = b * stride_kb + h * stride_kh 
    v_offset = b * stride_vb + h * stride_vh
    offset_d = tl.arange(0, BLOCK_DMODEL)

    for i in tl.range(kv_len):
        kv_idx = tl.load(kv_ids + q_idx * stride_kv_q + i * stride_kv_k)
        kv_offset = kv_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        kv_mask = (kv_offset < SEQLEN_K)[:, None]
        k_ptrs = k + k_offset + kv_offset[:, None] * stride_ks + offset_d[None, :] * stride_kd
        v_ptrs = v + v_offset + kv_offset[:, None] * stride_vs + offset_d[None, :] * stride_vd

        k_block = tl.load(k_ptrs, mask=kv_mask, other=0.)
        qkt = tl.dot(q_block, k_block.T)
        if MASK:
            mask_q_offset = (q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q))[:, None]
            mask_k_offset = kv_offset[None, :]
            mask_ptrs = mask + mask_q_offset * stride_mq + mask_k_offset * stride_mk
            mask_mask = mask_k_offset < SEQLEN_K[None, :]
            mask_block = tl.load(mask_ptrs, mask=mask_mask, other=False).to(tl.int1)
            qkt = qkt * sm_scale + tl.where(mask_block, 0, -float("inf"))
            mij = tl.maximum(m, tl.max(qkt, axis=-1))
        else:
            mij = tl.maximum(m, tl.max(qkt, axis=-1)*sm_scale)
            qkt *= sm_scale

        # todo: add a condition to convert mij to 0 if it's -inf
        p = tl.exp2(qkt - mij[:, None])
        alpha = tl.exp2(m - mij)
        m = mij
        l = l * alpha + tl.sum(p, axis=-1)
        acc = acc * alpha[:, None]

        v_block = tl.load(v_ptrs, mask=kv_mask, other=0.)
        acc = tl.dot(p.to(v_block.dtype), v_block, acc)

    return m, l, acc


autotune_configs = [
    triton.Config({}, num_stages=s, num_warps=w) \
    for s in ([1,2,3,4])\
    for w in [4,8]\
]

@triton.autotune(autotune_configs, key=[])
@triton.jit
def sparse_attention_kernel(
    q, stride_qb, stride_qh, stride_qs, stride_qd,
    k, stride_kb, stride_kh, stride_ks, stride_kd,
    v, stride_vb, stride_vh, stride_vs, stride_vd,
    o, stride_ob, stride_oh, stride_os, stride_od,
    mask, stride_mb, stride_mh, stride_mq, stride_mk,
    unmasked_kv_ids, stride_um_kv_b, stride_um_kv_h, stride_um_kv_q, stride_um_kv_k,
    masked_kv_ids, stride_m_kv_b, stride_m_kv_h, stride_m_kv_q, stride_m_kv_k,
    unmasked_lens, stride_n_un_b, stride_n_un_h, stride_n_un_q,
    masked_lens, stride_n_m_b, stride_n_m_h, stride_n_m_q,
    SEQLEN_Q,
    SEQLEN_K,
    sm_scale: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):

    q_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b = bh_idx // NUM_HEADS
    h = bh_idx % NUM_HEADS

    q_offs = (q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q))
    q_ptrs = q + b * stride_qb + h * stride_qh + q_offs[:, None] * stride_qs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd
    q_mask = (q_offs < SEQLEN_Q)[:, None]
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.)

    m = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_DMODEL], dtype=tl.float32)

    unmasked_len = tl.load(unmasked_lens + b * stride_n_un_b + h * stride_n_un_h + q_idx * stride_n_un_q)
    unmasked_kv_ids = unmasked_kv_ids + b * stride_um_kv_b + h * stride_um_kv_h

    mask = mask + b * stride_mb + h * stride_mh

    # unmasked blocks
    m, l, acc = _sparse_inner(q_block,
        k, stride_kb, stride_kh, stride_ks, stride_kd,
        v, stride_vb, stride_vh, stride_vs, stride_vd,
        mask, stride_mq, stride_mk,
        unmasked_kv_ids, stride_um_kv_q, stride_um_kv_k,
        unmasked_len,
        q_idx, b, h,
        SEQLEN_K,
        m, l, acc,
        sm_scale,
        BLOCK_Q, BLOCK_K, BLOCK_DMODEL,
        MASK=False,
    )

    masked_len = tl.load(masked_lens + b * stride_n_m_b + h * stride_n_m_h + q_idx * stride_n_m_q)
    masked_kv_ids = masked_kv_ids + b * stride_m_kv_b + h * stride_m_kv_h
    # masked blocks
    m, l, acc = _sparse_inner(q_block,
        k, stride_kb, stride_kh, stride_ks, stride_kd,
        v, stride_vb, stride_vh, stride_vs, stride_vd,
        mask, stride_mq, stride_mk,
        masked_kv_ids, stride_m_kv_q, stride_m_kv_k,
        masked_len,
        q_idx, b, h,
        SEQLEN_K,
        m, l, acc,
        sm_scale,
        BLOCK_Q, BLOCK_K, BLOCK_DMODEL,
        MASK=True,
    )

    acc /= l[:, None]
    o_ptrs = o + b * stride_ob + h * stride_oh + q_offs[:, None] * stride_os + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od
    tl.store(o_ptrs, acc, q_mask)


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, BLOCK_Q, BLOCK_K, mask_metadata=None):
    B, H, Sq, D = q.shape
    Sk = k.shape[-2]
    sm_scale = q.shape[-1]**(-0.5) * 1/math.log(2)

    mask = mask.expand(B, H, Sq, Sk)

    # Compute mask metadata
    if mask_metadata is None:
        mask_metadata = compute_mask_kernel(mask, BLOCK_Q, BLOCK_K)

    unmasked_kv_ids = mask_metadata.unmasked_kv_ids
    unmasked_lens = mask_metadata.unmasked_lens
    masked_kv_ids = mask_metadata.masked_kv_ids
    masked_lens = mask_metadata.masked_lens

    o = torch.empty_like(q)

    grid = (triton.cdiv(Sq, BLOCK_Q), B*H)
    sparse_attention_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        mask, mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        unmasked_kv_ids, unmasked_kv_ids.stride(0), unmasked_kv_ids.stride(1), unmasked_kv_ids.stride(2), unmasked_kv_ids.stride(3),
        masked_kv_ids, masked_kv_ids.stride(0), masked_kv_ids.stride(1), masked_kv_ids.stride(2), masked_kv_ids.stride(3),
        unmasked_lens, unmasked_lens.stride(0), unmasked_lens.stride(1), unmasked_lens.stride(2),
        masked_lens, masked_lens.stride(0), masked_lens.stride(1), masked_lens.stride(2),
        Sq, Sk,
        sm_scale=sm_scale,
        NUM_HEADS=H,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_DMODEL=D,
    )
    return o
