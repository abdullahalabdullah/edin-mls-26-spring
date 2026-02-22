"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    denom = tl.sum(exp_s, axis=0)
    out = exp_s / denom
    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        attn_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    is_causal: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention-style fused attention kernel.

    Grid: (batch * num_heads, seq_q) — one program per query position.
    Iterates over K/V in BLOCK_K tiles with online softmax (single pass over K/V).
    No intermediate score tensor stored — O(seq_k * head_dim) SRAM, O(1) extra HBM.
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)

    # Load Q for this query position: (BLOCK_D,)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    # Running accumulators for online softmax
    m_i = tl.zeros([1], dtype=tl.float32) - 1e30   # running max
    l_i = tl.zeros([1], dtype=tl.float32)           # running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)     # running weighted-sum of V

    # Tile over K/V in BLOCK_K chunks — online softmax with rescaling
    for k_start in range(0, seq_k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < seq_k

        # Load K block: (BLOCK_K, BLOCK_D)
        k_block = tl.load(
            k_ptr
            + pid_bh * stride_k0
            + offs_k[:, None] * stride_k1
            + offs_d[None, :] * stride_k2,
            mask=mask_k[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        # Dot product scores: (BLOCK_K,) = sum(q * k, axis=1) * scale
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale

        # Causal mask: future keys → -inf
        if is_causal:
            scores = tl.where(offs_k > pid_q, float("-inf"), scores)

        # Mask out-of-bounds key positions
        scores = tl.where(mask_k, scores, float("-inf"))

        # Online softmax: update running max and rescale previous accumulator
        m_block = tl.max(scores, axis=0)         # max over this block (scalar)
        m_new = tl.maximum(m_i, m_block)         # new global max [1]
        alpha = tl.exp(m_i - m_new)              # rescaling factor for prev acc [1]
        beta = tl.exp(scores - m_new)            # exp of this block's scores (BLOCK_K,)

        # Load V block: (BLOCK_K, BLOCK_D)
        v_block = tl.load(
            v_ptr
            + pid_bh * stride_v0
            + offs_k[:, None] * stride_v1
            + offs_d[None, :] * stride_v2,
            mask=mask_k[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        # Update running stats
        l_i = l_i * alpha + tl.sum(beta, axis=0)               # [1]
        acc = acc * alpha + tl.sum(beta[:, None] * v_block, axis=0)  # (BLOCK_D,)
        m_i = m_new

    # Normalize output
    out = acc / l_i

    # Store result
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256
FLASH_BLOCK_K = 64  # tile size over seq_k in FlashAttention


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    head_dim_padded = next_power_of_two(head_dim)

    # FlashAttention path: fused single-pass kernel, no intermediate score tensor.
    # Falls back to torch when attention_mask is provided (mask is not None).
    use_flash = (
        q.is_cuda
        and head_dim_padded <= MAX_ATTENTION_DIM
        and attention_mask is None
    )

    if use_flash:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32).contiguous()
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()

        output = torch.zeros(
            (batch * num_heads, seq_q, head_dim),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        flash_attention_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            output,
            float(scale),
            seq_k,
            head_dim,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            is_causal,
            BLOCK_K=FLASH_BLOCK_K,
            BLOCK_D=head_dim_padded,
        )

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # --- Original 3-kernel path (commented out, replaced by FlashAttention above) ---
    # seq_k_padded = next_power_of_two(seq_k)
    #
    # use_triton = (
    #     q.is_cuda
    #     and seq_k_padded <= MAX_ATTENTION_DIM
    #     and head_dim_padded <= MAX_ATTENTION_DIM
    # )
    #
    # if use_triton:
    #     q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
    #     k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
    #     v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
    #
    #     if seq_k_padded != seq_k or head_dim_padded != head_dim:
    #         k_padded = torch.zeros((batch * num_heads, seq_k_padded, head_dim_padded), ...)
    #         ...
    #
    #     scores = torch.empty((batch * num_heads, seq_q, seq_k_padded), ...)
    #     output = torch.empty((batch * num_heads, seq_q, head_dim_padded), ...)
    #
    #     grid = (batch * num_heads, seq_q)
    #     attention_scores_kernel[grid](...)  # kernel 1: Q @ K^T
    #
    #     softmax_inplace_kernel[(...)](...)  # kernel 2: in-place softmax
    #
    #     attention_output_kernel[grid](...)  # kernel 3: attn_weights @ V
    #
    #     return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")
