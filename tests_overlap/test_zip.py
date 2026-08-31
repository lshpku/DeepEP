import paddle
import deep_ep

paddle.set_printoptions(linewidth=200, edgeitems=5)


N = 4096       # num_recv_tokens
TOPK = 8
HIDDEN = 4096
E = 16         # 本地专家数
NUM_CTAS = 4


def make_case():
    """
    构造一份 zip 的输入:
      - 每个 token 随机命中 1..TOPK 个本地专家, 在行内的顺序是乱的 (和 DeepEP 一致)
      - zip_to_atomic 指向 o3 里互不相同的行, 且 o3 里的行也是乱的 (模拟 atomic 序)
      - recv_topk_idx 的无效槽位故意填垃圾值, 用来验证 zip 只信 zip_to_atomic 的有效性判断
    """
    paddle.seed(0)

    # 随机命中 1..TOPK 个本地专家, 其余位置写 -1
    num_valid_topk = paddle.randint(1, TOPK + 1, [N])
    topk_idx_perm = paddle.randn([N, E]).argsort()[..., :TOPK]
    topk_idx_mask = paddle.arange(TOPK) < num_valid_topk.unsqueeze(1)
    topk_idx = paddle.where(
        topk_idx_mask, topk_idx_perm, paddle.full([1], -1, dtype="int64")
    )

    # 再次打乱行内顺序, 让有效位和 -1 交替出现
    topk_idx_perm = paddle.randn([N, TOPK]).argsort()
    topk_idx = topk_idx.index_sample(topk_idx_perm)

    tokens_per_expert = paddle.sum(
        paddle.arange(E).unsqueeze(1) == topk_idx.flatten(), axis=1
    ).tolist()

    topk_probs = paddle.randn([N, TOPK])

    num_unzipped_tokens = sum((n + 127) // 128 * 128 for n in tokens_per_expert)

    # o3 使用 atomic 序, o3_ref 使用 paddle unzip 的递增序
    o3 = paddle.randn([num_unzipped_tokens, HIDDEN], "bfloat16")
    o3_ref = paddle.empty_like(o3)

    # 逐专家赋值其 zip_to_atomic, 打乱在 o3 中的顺序
    offset = 0
    zip_to_atomic = paddle.full([N, TOPK], -1, dtype="int32")
    for i, n in enumerate(tokens_per_expert):
        atomic_perm = paddle.randn(n).argsort().cast("int32")
        zip_to_atomic[topk_idx == i] = atomic_perm + offset
        o3_ref[offset : offset + n] = o3[offset : offset + n].gather(atomic_perm)
        offset += (n + 127) // 128 * 128

    # 模拟任务随机到达顺序的情况
    task_order = paddle.randn([N]).argsort().cast("int32")

    # 调用 unzip 获取 zipped_expertwise_rowmap, 这是 zip 依赖的输入, 其余返回值无用
    hidden_states = paddle.empty([N, HIDDEN], dtype="bfloat16")
    scale = None
    (
        unzipped_tokens,
        zipped_expertwise_rowmap,
        unzipped_probs,
        unzipped_scale,
    ) = paddle.nn.functional.moe_permute(
        hidden_states,
        scale,
        topk_idx.astype("int32"),
        topk_probs,
        padding_alignment=128,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    return o3, zip_to_atomic, topk_idx, task_order, o3_ref, zipped_expertwise_rowmap


def reference(o3, zipped_expertwise_rowmap, topk_idx):
    unzipped_probs = paddle.empty([o3.shape[0]], dtype="float32")
    zipped_out, zipped_probs_topk = paddle.nn.functional.moe_unpermute(
        o3,
        zipped_expertwise_rowmap,
        topk_idx,
        unzipped_probs,
        total_zipped_tokens=N,
        num_experts=E,
    )
    return zipped_out


def check(name, out, zip_done, out_ref):
    diff = paddle.abs(out.float() - out_ref.float())
    count = int((diff != 0).sum())
    unfinished = int((zip_done != 1).sum())
    print(name, "diff:", count, "avg:", float(diff.mean()), "max:", float(diff.max()),
          "unfinished:", unfinished)
    return count == 0 and unfinished == 0


def main():
    o3, zip_to_atomic, topk_idx, task_order, o3_ref, rowmap = make_case()

    topk_idx_i32 = topk_idx.cast("int32")
    out_ref = reference(o3_ref, rowmap, topk_idx_i32)

    # case 1: 队列预先填好, 纯功能校验
    zip_done = paddle.zeros([N], "int32")
    out = deep_ep.zip(o3, zip_to_atomic, topk_idx, task_order, zip_done, NUM_CTAS)
    ok = check("[zip ready]", out, zip_done, out_ref)
    paddle.base.core.nvprof_start()

    # case 2: 先 launch zip, 再从另一条流把队列填进去, 验证 persistent kernel 等待
    zip_done = paddle.zeros([N], "int32")
    task_queue = paddle.full([N], -1, "int32")
    stream = paddle.device.Stream()

    paddle.base.core.nvprof_nvtx_push("zip")
    out = deep_ep.zip(o3, zip_to_atomic, topk_idx, task_queue, zip_done, NUM_CTAS)
    paddle.base.core.nvprof_nvtx_pop()

    with paddle.device.stream_guard(stream):
        paddle.base.core.nvprof_nvtx_push("assign")
        task_queue[:] = task_order
        paddle.base.core.nvprof_nvtx_pop()

    ok = check("[zip delay]", out, zip_done, out_ref) and ok

    # 性能测试
    for i in range(5):
        paddle.base.core.nvprof_nvtx_push("deepep")
        deep_ep.zip(o3, zip_to_atomic, topk_idx, task_order, zip_done, NUM_CTAS)
        paddle.base.core.nvprof_nvtx_pop()
    for i in range(5):
        paddle.base.core.nvprof_nvtx_push("paddle")
        reference(o3_ref, rowmap, topk_idx_i32)
        paddle.base.core.nvprof_nvtx_pop()

    paddle.device.synchronize()
    paddle.base.core.nvprof_stop()

    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
