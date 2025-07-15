from dataclasses import dataclass
import torch
import triton
import triton.language as tl

@dataclass
class MaskMetadata:
    unmasked_kv_ids: torch.Tensor
    unmasked_lens: torch.Tensor
    masked_kv_ids: torch.Tensor
    masked_lens: torch.Tensor


def compute_mask_slow(mask: torch.Tensor, BLOCK_Q: int, BLOCK_K: int) -> MaskMetadata:
    # 1: completely unmasked blocks
    # 2: semi masked blocks
    B, H, Sq, Sk = mask.shape
    assert mask.dtype == torch.bool
    assert Sq % BLOCK_Q == 0
    assert Sk % BLOCK_K == 0

    unmasked_kv_ids = torch.empty(B, H, Sq // BLOCK_Q, Sk // BLOCK_K, dtype=torch.int, device=mask.device)
    num_unmasked = torch.empty(B, H, Sq // BLOCK_Q, dtype=torch.int, device=mask.device)
    masked_kv_ids = torch.empty(B, H, Sq // BLOCK_Q, Sk // BLOCK_K, dtype=torch.int, device=mask.device)
    num_masked = torch.empty(B, H, Sq // BLOCK_Q, dtype=torch.int, device=mask.device)

    for b in range(B):
        for h in range(H):
            for i in range(0, Sq // BLOCK_Q):
                row_num_unmasked = 0
                row_num_masked = 0
                for j in range(0, Sk // BLOCK_K):
                    block = mask[b, h, i*BLOCK_Q: i*BLOCK_Q+BLOCK_Q, j*BLOCK_K:j*BLOCK_K+BLOCK_K]
                    count = torch.sum(block)
                    if count == BLOCK_Q * BLOCK_K:
                        unmasked_kv_ids[b, h, i, row_num_unmasked] = j
                        row_num_unmasked += 1
                    elif count > 0:
                        masked_kv_ids[b, h, i, row_num_masked] = j
                        row_num_masked += 1
                num_unmasked[b, h, i] = row_num_unmasked
                num_masked[b, h, i] = row_num_masked

    return MaskMetadata(unmasked_kv_ids, num_unmasked, masked_kv_ids, num_masked)


@triton.jit
def _compute_mask_inner(
    mask, stride_mb, stride_mh, stride_mq, stride_mk,
    unmasked_kv_ids, stride_um_kv_b, stride_um_kv_h, stride_um_kv_q, stride_um_kv_k,
    masked_kv_ids, stride_m_kv_b, stride_m_kv_h, stride_m_kv_q, stride_m_kv_k,
    num_unmasked, stride_n_un_b, stride_n_un_h, stride_n_un_q,
    num_masked, stride_n_m_b, stride_n_m_h, stride_n_m_q,
    Sq, Sk,
    num_heads: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    q_idx = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // num_heads
    h = bh % num_heads
    mask_q_offset = q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    unmasked_count = 0
    masked_count = 0
    for k_idx in range(0, tl.cdiv(Sk, BLOCK_K)):
        mask_k_offset = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_ptrs = mask + b * stride_mb + h * stride_mh + mask_q_offset[:, None] * stride_mq + mask_k_offset[None, :] * stride_mk
        load_mask = (mask_q_offset[:, None] < Sq) & (mask_k_offset[None, :] < Sk)
        mask_block = tl.load(mask_ptrs, mask=load_mask, other=0.)
        num_unmasked_entries = tl.sum(mask_block)
        if num_unmasked_entries == BLOCK_Q * BLOCK_K:
            unmasked_ptrs = unmasked_kv_ids + b * stride_um_kv_b + h * stride_um_kv_h + q_idx * stride_um_kv_q + unmasked_count * stride_um_kv_k
            tl.store(unmasked_ptrs, k_idx)
            unmasked_count += 1
        elif num_unmasked_entries > 0:
            masked_ptrs = masked_kv_ids + b * stride_m_kv_b + h * stride_m_kv_h + q_idx * stride_m_kv_q + masked_count * stride_m_kv_k
            tl.store(masked_ptrs, k_idx)
            masked_count += 1

    tl.store(num_unmasked + q_idx * stride_n_un_q + b * stride_n_un_b + h * stride_n_un_h, unmasked_count)
    tl.store(num_masked + q_idx * stride_n_m_q + b * stride_n_m_b + h * stride_n_m_h, masked_count)


def compute_mask_kernel(mask: torch.Tensor, BLOCK_Q: int, BLOCK_K: int) -> MaskMetadata:
    # todo: make this split-k
    B, H, Sq, Sk = mask.shape
    # todo: change div to triton.cdiv + mask boundaries
    assert Sq % BLOCK_Q == 0
    assert Sk % BLOCK_K == 0
    unmasked_kv_ids = torch.empty(B, H, Sq // BLOCK_Q, Sk // BLOCK_K, dtype=torch.int, device=mask.device)
    masked_kv_ids = torch.empty(B, H, Sq // BLOCK_Q, Sk // BLOCK_K, dtype=torch.int, device=mask.device)
    num_unmasked = torch.empty(B, H, Sq//BLOCK_Q, dtype=torch.int, device=mask.device)
    num_masked = torch.empty(B, H, Sq//BLOCK_Q, dtype=torch.int, device=mask.device)
    grid = (triton.cdiv(Sq, BLOCK_Q), B*H)
    _compute_mask_inner[grid](
        mask, mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        unmasked_kv_ids, unmasked_kv_ids.stride(0), unmasked_kv_ids.stride(1), unmasked_kv_ids.stride(2), unmasked_kv_ids.stride(3),
        masked_kv_ids, masked_kv_ids.stride(0), masked_kv_ids.stride(1), masked_kv_ids.stride(2), masked_kv_ids.stride(3),
        num_unmasked, num_unmasked.stride(0), num_unmasked.stride(1), num_unmasked.stride(2),
        num_masked, num_masked.stride(0), num_masked.stride(1), num_masked.stride(2),
        Sq, Sk,
        num_heads=H,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        )
    return MaskMetadata(unmasked_kv_ids, num_unmasked, masked_kv_ids, num_masked)
