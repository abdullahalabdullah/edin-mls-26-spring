"""
Example demonstrating a simplified tiled attention mechanism.
Out = exp(Q @ K.T / sqrt(d)) @ V
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def simple_attention(Q, K, V, Out, M,
                     stride_qm, stride_qd,
                     stride_km, stride_kd,
                     stride_vm, stride_vd,
                     stride_om, stride_od,
                     SCALE,
                     SEQ_LEN_K: tl.constexpr,
                     HEAD_DIM: tl.constexpr,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM), other=0.0)

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for k_start in range(0, SEQ_LEN_K, BLOCK_N):
        offs_n = k_start + tl.arange(0, BLOCK_N)

        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=(offs_n[:, None] < SEQ_LEN_K) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < SEQ_LEN_K) & (offs_d[None, :] < HEAD_DIM), other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        scores = scores * SCALE
        scores = tl.exp(scores)

        acc += tl.dot(scores, v, input_precision="ieee")

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    mask_out = (offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM)
    tl.store(out_ptrs, acc, mask=mask_out)


def test_attention():
    M = 128  # Number of Queries
    N = 128  # Number of Keys/Values
    D = 64   # Head Dimension

    BLOCK_M = 32
    BLOCK_N = 32

    print(f"Attention Problem: Q({M}x{D}) @ K({N}x{D}).T @ V({N}x{D})")

    q = torch.randn((M, D), device="cuda", dtype=torch.float32)
    k = torch.randn((N, D), device="cuda", dtype=torch.float32)
    v = torch.randn((N, D), device="cuda", dtype=torch.float32)
    out = torch.zeros((M, D), device="cuda", dtype=torch.float32)

    stride_qm, stride_qd = q.stride()
    stride_km, stride_kd = k.stride()
    stride_vm, stride_vd = v.stride()
    stride_om, stride_od = out.stride()

    scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(M, BLOCK_M),)

    simple_attention[grid](
        q, k, v, out, M,
        stride_qm, stride_qd,
        stride_km, stride_kd,
        stride_vm, stride_vd,
        stride_om, stride_od,
        scale,
        SEQ_LEN_K=N,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    # Reference on CPU
    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    expected = torch.exp((q_cpu @ k_cpu.T) * scale) @ v_cpu

    out_cpu = out.cpu()

    print("Checking accuracy...")
    torch.testing.assert_close(out_cpu, expected, rtol=1e-3, atol=1e-3)
    print("[PASS] Tiled Attention Passed!")


@triton.jit
def flash_attention(Q, K, V, Out,
                    stride_qm, stride_qd,
                    stride_km, stride_kd,
                    stride_vm, stride_vd,
                    stride_om, stride_od,
                    M, N,
                    SCALE,
                    HEAD_DIM: tl.constexpr,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q tile once — stays in registers for the entire loop
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM), other=0.0)

    # Online softmax state per query row
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                       # running sum (denominator)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)              # weighted V accumulator

    for k_start in range(0, N, BLOCK_N):
        offs_n = k_start + tl.arange(0, BLOCK_N)

        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < HEAD_DIM), other=0.0)

        # Scaled dot-product scores: (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE

        # Update running max across this tile's keys
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))  # (BLOCK_M,)

        # Numerically stable exp: subtract running max before exponentiating
        p = tl.exp(scores - m_new[:, None])               # (BLOCK_M, BLOCK_N)

        # Rescale old accumulator and sum to account for updated max
        alpha = tl.exp(m_i - m_new)                       # (BLOCK_M,)
        l_i = alpha * l_i + tl.sum(p, axis=1)             # (BLOCK_M,)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")

        m_i = m_new

    # Divide by the accumulated softmax denominator
    acc = acc / l_i[:, None]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    mask_out = (offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM)
    tl.store(out_ptrs, acc, mask=mask_out)


def test_flash_attention():
    M = 128  # Number of Queries
    N = 128  # Number of Keys/Values
    D = 64   # Head Dimension

    BLOCK_M = 32
    BLOCK_N = 32

    print(f"FlashAttention Problem: Q({M}x{D}) @ K({N}x{D}).T @ V({N}x{D})")

    q = torch.randn((M, D), device="cuda", dtype=torch.float32)
    k = torch.randn((N, D), device="cuda", dtype=torch.float32)
    v = torch.randn((N, D), device="cuda", dtype=torch.float32)
    out = torch.zeros((M, D), device="cuda", dtype=torch.float32)

    stride_qm, stride_qd = q.stride()
    stride_km, stride_kd = k.stride()
    stride_vm, stride_vd = v.stride()
    stride_om, stride_od = out.stride()

    scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(M, BLOCK_M),)

    flash_attention[grid](
        q, k, v, out,
        stride_qm, stride_qd,
        stride_km, stride_kd,
        stride_vm, stride_vd,
        stride_om, stride_od,
        M, N,
        scale,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    # Reference: proper softmax attention (not the unnormalized exp used in simple_attention)
    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    scores_ref = (q_cpu @ k_cpu.T) * scale          # (M, N)
    attn_ref = torch.softmax(scores_ref, dim=-1)     # (M, N) — normalized
    expected = attn_ref @ v_cpu                      # (M, D)

    out_cpu = out.cpu()

    print("Checking accuracy...")
    torch.testing.assert_close(out_cpu, expected, rtol=1e-3, atol=1e-3)
    print("[PASS] FlashAttention Passed!")


@triton.jit
def two_pass_attention(Q, K, V, Out,
                       stride_qm, stride_qd,
                       stride_km, stride_kd,
                       stride_vm, stride_vd,
                       stride_om, stride_od,
                       M, N,
                       SCALE,
                       HEAD_DIM: tl.constexpr,
                       BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    """Correct two-pass attention: pass 1 finds the row max, pass 2 normalizes.
    Reads K twice (two full passes over the sequence), so it uses 2x the memory
    bandwidth of flash_attention for the same correct result."""
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM), other=0.0)

    # Pass 1: scan all K tiles to find the global row max
    row_max = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    for k_start in range(0, N, BLOCK_N):
        offs_n = k_start + tl.arange(0, BLOCK_N)
        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        scores = tl.where(offs_n[None, :] < N, scores, float("-inf"))
        row_max = tl.maximum(row_max, tl.max(scores, axis=1))

    # Pass 2: now that we know the max, compute exp(scores - max), accumulate
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    for k_start in range(0, N, BLOCK_N):
        offs_n = k_start + tl.arange(0, BLOCK_N)
        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < N) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        scores = tl.where(offs_n[None, :] < N, scores, float("-inf"))
        p = tl.exp(scores - row_max[:, None])
        row_sum += tl.sum(p, axis=1)
        acc += tl.dot(p, v, input_precision="ieee")

    acc = acc / row_sum[:, None]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    mask_out = (offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM)
    tl.store(out_ptrs, acc, mask=mask_out)


def test_two_pass_attention():
    M = 128
    N = 128
    D = 64
    BLOCK_M = 32
    BLOCK_N = 32

    print(f"TwoPassAttention Problem: Q({M}x{D}) @ K({N}x{D}).T @ V({N}x{D})")

    q = torch.randn((M, D), device="cuda", dtype=torch.float32)
    k = torch.randn((N, D), device="cuda", dtype=torch.float32)
    v = torch.randn((N, D), device="cuda", dtype=torch.float32)
    out = torch.zeros((M, D), device="cuda", dtype=torch.float32)

    stride_qm, stride_qd = q.stride()
    stride_km, stride_kd = k.stride()
    stride_vm, stride_vd = v.stride()
    stride_om, stride_od = out.stride()

    scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(M, BLOCK_M),)

    two_pass_attention[grid](
        q, k, v, out,
        stride_qm, stride_qd,
        stride_km, stride_kd,
        stride_vm, stride_vd,
        stride_om, stride_od,
        M, N, scale,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )

    q_cpu, k_cpu, v_cpu = q.cpu(), k.cpu(), v.cpu()
    scores_ref = (q_cpu @ k_cpu.T) * scale
    expected = torch.softmax(scores_ref, dim=-1) @ v_cpu

    print("Checking accuracy...")
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)
    print("[PASS] TwoPass Attention Passed!")


def benchmark_attention(seq_len, n_warmup=20, n_iter=100):
    """Benchmark two_pass_attention (correct baseline) vs flash_attention."""
    D = 64
    BLOCK_M = 32
    BLOCK_N = 32
    scale = 1.0 / math.sqrt(D)

    q = torch.randn((seq_len, D), device="cuda", dtype=torch.float32)
    k = torch.randn((seq_len, D), device="cuda", dtype=torch.float32)
    v = torch.randn((seq_len, D), device="cuda", dtype=torch.float32)
    out = torch.zeros((seq_len, D), device="cuda", dtype=torch.float32)

    stride_qm, stride_qd = q.stride()
    stride_km, stride_kd = k.stride()
    stride_vm, stride_vd = v.stride()
    stride_om, stride_od = out.stride()

    grid = (triton.cdiv(seq_len, BLOCK_M),)

    def run_two_pass():
        two_pass_attention[grid](
            q, k, v, out,
            stride_qm, stride_qd,
            stride_km, stride_kd,
            stride_vm, stride_vd,
            stride_om, stride_od,
            seq_len, seq_len, scale,
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

    def run_flash():
        flash_attention[grid](
            q, k, v, out,
            stride_qm, stride_qd,
            stride_km, stride_kd,
            stride_vm, stride_vd,
            stride_om, stride_od,
            seq_len, seq_len, scale,
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

    def time_fn(fn):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / n_iter

    t_two_pass = time_fn(run_two_pass)
    t_flash    = time_fn(run_flash)
    return t_two_pass, t_flash


def test_performance():
    seq_lens = [128, 256, 512, 1024, 2048]

    print("\n== Attention Performance Benchmark (both correct) ==")
    print(f"{'Seq Len':<10} | {'TwoPass (ms)':<15} | {'Flash (ms)':<14} | {'Speedup':<10}")
    print("-" * 58)

    for seq_len in seq_lens:
        t_two_pass, t_flash = benchmark_attention(seq_len)
        speedup = t_two_pass / t_flash
        print(f"{seq_len:<10} | {t_two_pass:<15.4f} | {t_flash:<14.4f} | {speedup:<10.2f}x")


if __name__ == "__main__":
    test_attention()
    test_flash_attention()
    test_two_pass_attention()
    test_performance()


