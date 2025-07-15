import datetime
import json

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import triton
import matplotlib.pyplot as plt
import numpy as np

from sparseattention.sparse_attn_kernel import sparse_attention
from sparseattention.mask_utils import compute_mask_kernel

flex_attention = torch.compile(flex_attention)


def random_blocked_mask(B, Sq, Sk, BLOCK_Q, BLOCK_K, p):
    N_blocks = B * (Sq // BLOCK_Q) * (Sk // BLOCK_K)
    rand_vals = torch.rand(N_blocks)
    masked_blocks = torch.zeros(N_blocks, dtype=torch.int)
    masked_blocks[rand_vals > (1-p)] = 1
    masked_blocks[rand_vals >= (1 - 0.1)] = 2
    masked_blocks = masked_blocks.reshape(B, Sq // BLOCK_Q, Sk // BLOCK_K)
    mask = masked_blocks.repeat_interleave(BLOCK_Q, 1).repeat_interleave(BLOCK_K, 2).reshape(B,1,Sq,Sk)
    semi_masked = mask == 2
    new_vals = torch.randint(0,2, (int(semi_masked.sum()),)).to(torch.int)
    mask[semi_masked] = new_vals
    mask = mask.to(torch.bool).to("cuda")
    return mask


def test_attn(B, H, Sq, Sk, D, mask, BLOCK_Q=32, BLOCK_K=128, dtype=torch.bfloat16):
    q = torch.randn(B, H, Sq, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, Sk, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, Sk, D, dtype=dtype, device="cuda")
    mask_expanded = mask.expand(B, H, Sq, Sk)

    mask_metadata = compute_mask_kernel(mask_expanded, BLOCK_Q, BLOCK_K)

    ref_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    sparse_attn = sparse_attention(q, k, v, mask_expanded, BLOCK_Q, BLOCK_K, mask_metadata=mask_metadata)

    # there's probably a better way to do this...
    # ignoring materialisation costs for now though so it's okay
    def mask_mod(b, h, q_idx, kv_idx):
        return mask_expanded[b, h, q_idx, kv_idx]
    block_mask = create_block_mask(mask_mod, B, H, Sq, Sk, device="cuda")
    flex_attn_result = flex_attention(q, k, v, block_mask=block_mask)

    try:
        torch.testing.assert_close(ref_attn, sparse_attn, atol=1e-2, rtol=1e-4)
        print("sparse and ref match!")
    except Exception as e:
        print(e)
        pass

    try:
        torch.testing.assert_close(ref_attn, flex_attn_result, atol=1e-2, rtol=1e-4)
        print("flex and ref match!")
    except Exception as e:
        # print(e)
        pass

    ref_time = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    sparse_time = triton.testing.do_bench(lambda: sparse_attention(q, k, v, mask_expanded, BLOCK_Q, BLOCK_K, mask_metadata=mask_metadata))
    flex_time = triton.testing.do_bench(lambda: flex_attention(q, k, v, block_mask=block_mask))

    mask_time = triton.testing.do_bench(lambda: compute_mask_kernel(mask_expanded, BLOCK_Q, BLOCK_K))
    print(f"MASK_TIME: {mask_time:.2f}ms")
    print(f"SPARSE_TIME: {sparse_time:.2f}ms")
    print(f"FLEX_TIME: {flex_time:.2f}ms")
    return {
        "ref_time": ref_time,
        "sparse_time": sparse_time,
        "flex_time": flex_time,
    }


if __name__ == "__main__":
    torch.manual_seed(1352)

    B, H, Sq, Sk, D = 8, 16, 4096, 4096, 128
    BLOCK_Q, BLOCK_K = 128, 32
    total_flops = 2 * 2 * B * H * Sq * Sk * D

    p_values = []
    ref_flops_per_sec = []
    sparse_flops_per_sec = []
    flex_flops_per_sec = []

    detailed_results = {
        "config": {
            "shape": (B, H, Sq, Sk),
            "BLOCK_Q": BLOCK_Q, "BLOCK_K": BLOCK_K,
            "total_flops": total_flops
        },
        "results": []
    }

    for i in range(1, 10):
        p = i/10
        # randomly block out 64x64 blocks with probability p
        mask = random_blocked_mask(B, Sq, Sk, BLOCK_Q=128, BLOCK_K=128, p=p)

        fraction_unmasked = float(mask.sum() / mask.numel())
        print(f"p: {p:.2f}, fraction_unmasked: {fraction_unmasked:.2f}")
        results = test_attn(B, H, Sq, Sk, D, mask, BLOCK_Q, BLOCK_K)
        useful_flops = total_flops * fraction_unmasked
        ref_flops_s = useful_flops / (results['ref_time'] / 1000)
        sparse_flops_s = useful_flops / (results['sparse_time'] / 1000)
        flex_flops_s = useful_flops / (results['flex_time'] / 1000)

        p_values.append(p)
        ref_flops_per_sec.append(ref_flops_s)
        sparse_flops_per_sec.append(sparse_flops_s)
        flex_flops_per_sec.append(flex_flops_s)

        detailed_results["results"].append({
            "p": p,
            "fraction_unmasked": fraction_unmasked,
            "useful_flops": useful_flops,
            "ref_time_ms": results['ref_time'],
            "sparse_time_ms": results['sparse_time'],
            "flex_time_ms": results['flex_time'],
            "ref_tflops_per_sec": ref_flops_s / 1e12,
            "sparse_tflops_per_sec": sparse_flops_s / 1e12,
            "flex_tflops_per_sec": flex_flops_s / 1e12
        })

        # print(f"p: {p:.2f}, ref: {results['ref_time']:.2f}ms, sparse: {results['sparse_time']:.2f}ms, flex: {results['flex_time']:.2f}ms")
        print(f"  FLOPS/s - ref: {ref_flops_s/1e12:.2f} TFLOPS/s, sparse: {sparse_flops_s/1e12:.2f} TFLOPS/s, flex: {flex_flops_s/1e12:.2f} TFLOPS/s")

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, np.array(ref_flops_per_sec)/1e12, 'b-o', label='Reference (SDPA)', linewidth=2, markersize=6)
    plt.plot(p_values, np.array(sparse_flops_per_sec)/1e12, 'r-s', label='Sparse Kernel', linewidth=2, markersize=6)
    plt.plot(p_values, np.array(flex_flops_per_sec)/1e12, 'g-^', label='Flex Attention', linewidth=2, markersize=6)

    plt.xlabel('Fraction of blocks unmasked', fontsize=12)
    plt.ylabel('TFLOPS/s', fontsize=12)
    plt.title('Attention Performance vs Sparsity', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'attention_performance_{timestamp}.png', dpi=300, bbox_inches='tight')

    with open('attention_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)


