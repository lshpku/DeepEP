import paddle
import paddle.distributed as dist
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)
paddle.enable_compat(scope={"deep_ep"})

import deep_ep
print("deep_ep:", deep_ep.__file__)

from utils import initialize_fleet, configure_buffer, get_buffer

E = 16
H = 4096
SEQLEN = 16384
TOPK = 8

NUM_SMS = 48
ALIGNMENT = 128


def prepare_case_inputs(group):
    num_experts = group.world_size * E

    x = paddle.randn([SEQLEN, H], "bfloat16")
    # 把 token 的全局唯一 id 编码进前三列, 用于后面做精确的乱序比对.
    # bf16 只能精确表示 [0, 256) 的整数, 所以拆成 (rank, i // 128, i % 128) 三位.
    ids = paddle.arange(SEQLEN)
    x[:, 0] = paddle.full([SEQLEN], dist.get_rank(), "bfloat16")
    x[:, 1] = (ids // 128).astype("bfloat16")
    x[:, 2] = (ids % 128).astype("bfloat16")

    scores = paddle.randn([SEQLEN, num_experts])
    scores += paddle.randn([num_experts]) * 0.1

    topk_weights, topk_idx = scores.topk(TOPK)
    topk_weights = F.sigmoid(topk_weights)
    topk_weights /= topk_weights.sum(axis=-1, keepdim=True)

    return x, topk_weights, topk_idx


def token_ids(tokens):
    """从 token 的前三列还原出全局唯一 id, 全整数运算, 不会有并列."""
    f = tokens.astype("float32")
    return (f[:, 0] * 16384 + f[:, 1] * 128 + f[:, 2]).astype("int64")


def run_dispatch(group, buffer, x, token_probs, token_indices, unzip_alignment):
    num_experts = group.world_size * E

    layout = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = layout

    return buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=False,
        allocate_on_comm_stream=False,
        unzip_alignment=unzip_alignment,
    )


def check_fused_unzip(recv_x, recv_token_indices, recv_token_probs, tokens_per_expert,
                      unzipped_tokens, unzipped_probs):
    """
    融合后的 unzip 输出在每个专家段内是 atomic 先到先放, 顺序不确定, 因此这里
    对每个专家段做集合比较: 用一个固定随机投影给每行算一个 key, 排序后比对.
    """
    ref_tokens, _, ref_probs, _ = paddle.nn.functional.moe_permute(
        recv_x,
        None,
        recv_token_indices,
        recv_token_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )
    assert ref_tokens.shape == unzipped_tokens.shape, (ref_tokens.shape, unzipped_tokens.shape)
    ok = True
    offset = 0
    for e, n in enumerate(tokens_per_expert):
        ref_seg = ref_tokens[offset:offset + n].astype("float32")
        new_seg = unzipped_tokens[offset:offset + n].astype("float32")

        ref_order = paddle.argsort(token_ids(ref_seg))
        new_order = paddle.argsort(token_ids(new_seg))

        num_expected = int((recv_token_indices == e).sum().item())
        id_diff = (token_ids(ref_seg)[ref_order] - token_ids(new_seg)[new_order]).abs().max().item() if n > 0 else 0

        token_diff = (ref_seg[ref_order] - new_seg[new_order]).abs().max().item() if n > 0 else 0.0
        prob_diff = (
            ref_probs[offset:offset + n][ref_order] - unzipped_probs[offset:offset + n][new_order]
        ).abs().max().item() if n > 0 else 0.0

        if token_diff != 0.0 or prob_diff != 0.0 or num_expected != n:
            ok = False
            print(f"  expert {e}: n={n} expected={num_expected} id_diff={id_diff} "
                  f"token_diff={token_diff} prob_diff={prob_diff}")

        offset += (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    print("fused unzip check:", "PASS" if ok else "FAIL")
    return ok


def main():
    group = initialize_fleet()
    configure_buffer(NUM_SMS)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices = prepare_case_inputs(group)

    # warmup
    run_dispatch(group, buffer, x, token_probs, token_indices, 0)

    recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, _, _, unzipped_tokens, unzipped_probs = \
        run_dispatch(group, buffer, x, token_probs, token_indices, ALIGNMENT)

    print("num_recv_tokens:", recv_x.shape[0], "tokens_per_expert:", tokens_per_expert)
    print("num_unzipped_tokens:", unzipped_tokens.shape[0])

    check_fused_unzip(
        recv_x,
        recv_token_indices.cast("int32"),
        recv_token_probs,
        tokens_per_expert,
        unzipped_tokens,
        unzipped_probs,
    )

    ############################### PERF COMPARE ###############################

    # 共享开发机上性能抖动大, 所以交替测量再取中位数
    def bench(fn, num_iters=20):
        for _ in range(3):
            fn()
        paddle.device.synchronize()

        start = paddle.device.cuda.Event(enable_timing=True)
        end = paddle.device.cuda.Event(enable_timing=True)
        start.record()
        paddle.base.core.nvprof_nvtx_push("bench")
        for _ in range(num_iters):
            paddle.base.core.nvprof_nvtx_push("iter")
            fn()
            paddle.base.core.nvprof_nvtx_pop()
        paddle.base.core.nvprof_nvtx_pop()
        end.record()
        paddle.device.synchronize()
        return start.elapsed_time(end) / num_iters

    def dispatch_only():
        run_dispatch(group, buffer, x, token_probs, token_indices, 0)

    def dispatch_then_permute():
        recv_x, idx, probs, tpe, _, _ = run_dispatch(group, buffer, x, token_probs, token_indices, 0)
        paddle.nn.functional.moe_permute(
            recv_x, None, idx.cast("int32"), probs,
            padding_alignment=ALIGNMENT, num_experts=E, tokens_per_expert=tpe,
        )

    def dispatch_fused():
        run_dispatch(group, buffer, x, token_probs, token_indices, ALIGNMENT)

    paddle.base.core.nvprof_start()
    cases = {"dispatch only": [], "dispatch + moe_permute": [], "dispatch w/ fused unzip": []}
    for _ in range(5):
        cases["dispatch only"].append(bench(dispatch_only))
        cases["dispatch + moe_permute"].append(bench(dispatch_then_permute))
        cases["dispatch w/ fused unzip"].append(bench(dispatch_fused))
    paddle.base.core.nvprof_stop()

    for tag, samples in cases.items():
        samples.sort()
        print(f"{tag:>24}: median {samples[2]:.3f} ms  min {samples[0]:.3f} ms")


if __name__ == "__main__":
    main()
