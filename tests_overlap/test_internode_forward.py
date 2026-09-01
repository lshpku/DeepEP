import os
import sys

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)
paddle.enable_compat(scope={"deep_ep"})

import deep_ep
print("deep_ep:", deep_ep.__file__)

import deep_gemm
print("deep_gemm:", deep_gemm.__file__)

from utils import initialize_fleet, configure_buffer, get_buffer

# 使用特别编译的注释掉所有第三方库的版本, 不然会和开发中的 deep_ep/deep_gemm 冲突
import paddlefleet_ops
assert not paddlefleet_ops._DEEP_EP_AVAILABLE
assert not paddlefleet_ops._DEEP_GEMM_AVAILABLE

E = 8
H = 4096
I = 2048
SEQLEN = 16384
TOPK = 8

COMM_NUM_SMS = 48
CALC_NUM_SMS = 94
ZIP_NUM_SMS = 6

ALIGNMENT = 128
CHUNK = 4096
SWIGLU_FUSION = True

zip_stream = paddle.device.Stream()


def prepare_case_inputs(group):
    # E 是本地专家数, num_experts 是全局专家数
    num_experts = group.world_size * E

    x = paddle.randn([SEQLEN, H], "bfloat16")

    scores = paddle.randn([SEQLEN, num_experts])
    # 模拟给专家选择增加一定的系统不均衡
    # scores += paddle.randn([num_experts]) * 0.1

    topk_weights, topk_idx = scores.topk(TOPK)
    topk_weights = F.sigmoid(topk_weights)
    topk_weights /= topk_weights.sum(axis=-1, keepdim=True)

    w_gateup = (paddle.randn([E, H, 2 * I]) * 0.02).cast("bfloat16")
    w_down = (paddle.randn([E, I, H]) * 0.02).cast("bfloat16")

    return x, topk_weights, topk_idx, w_gateup, w_down


def run_overlap_forward(group, buffer, x, token_probs, token_indices, w_gateup, w_down):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(CALC_NUM_SMS)
    paddle.zeros([1])

    ################################# DISPATCH #################################
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    (
        recv_x, recv_token_indices, recv_token_probs,
        num_recv_tokens_per_expert_list, handle, event,
        unzipped_tokens, unzipped_probs, atomic_to_zip, zip_to_atomic,
        num_valid_topk, task_queue
    ) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=True,
        allocate_on_comm_stream=False,
        unzip_alignment=ALIGNMENT,
        unzip_chunk_size=CHUNK,
    )

    tokens_per_expert = num_recv_tokens_per_expert_list
    num_unzipped_tokens = len(unzipped_tokens)
    num_tasks = len(task_queue)
    num_recv_tokens = len(recv_x)

    print("tokens_per_expert:", tokens_per_expert)
    print("num_tasks:", [(n + CHUNK - 1) // CHUNK for n in tokens_per_expert], "=", num_tasks)
    print("num_recv_tokens:", num_recv_tokens)

    ############################### GEMM OVERLAP ###############################

    # event.current_stream_wait()
    # dist.all_reduce(paddle.empty([1]))

    o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
    o2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
    o3 = paddle.empty_like(unzipped_tokens)

    token_done = paddle.zeros([num_recv_tokens], "int32")
    zip_task_queue = paddle.full([num_recv_tokens], -1, "int32")
    zip_queue_tail = paddle.zeros([1], "int32")
    zip_done = paddle.zeros([num_recv_tokens], "int32")

    full_compute = False
    combine_event = None

    for task_idx in range(num_tasks):
        paddle.base.core.nvprof_nvtx_push(f"task_{task_idx}")
        # gateup
        deep_gemm.bf16_chunk_gemm_nn(
            unzipped_tokens, w_gateup, o1, task_queue, task_idx,
            **(dict(o2=o2, probs=unzipped_probs) if SWIGLU_FUSION else {}),
        )
        # swiglu
        if not SWIGLU_FUSION:
            deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2, task_queue, task_idx, precise=True
            )
        # down
        deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx)
        # done
        deep_gemm.chunk_signal_token_done(
            atomic_to_zip, num_valid_topk, token_done, zip_task_queue, zip_queue_tail,
            task_queue, task_idx,
        )
        paddle.base.core.nvprof_nvtx_pop()

        if not full_compute and task_idx >= num_tasks * 0.2:
            print(f"switch to full compute at {task_idx}/{num_tasks}")
            full_compute = True
            deep_gemm.set_num_sms(CALC_NUM_SMS + COMM_NUM_SMS)

        if combine_event is None and task_idx >= num_tasks * 0.7:
            print(f"capture combine_event at {task_idx}/{num_tasks}")
            combine_event = deep_ep.Buffer.capture()
            deep_gemm.set_num_sms(CALC_NUM_SMS)

    ############################### ZIP OVERLAP ################################

    # zip_stream = paddle.device.current_stream()

    with paddle.device.stream_guard(zip_stream):
        zipped_out = deep_ep.zip(
            o3, zip_to_atomic, recv_token_indices, zip_task_queue, zip_done, ZIP_NUM_SMS
        )

    ############################# COMBINE OVERLAP ##############################

    out, _, event = buffer.combine(zipped_out, handle, async_finish=False,
                                   previous_event=combine_event, allocate_on_comm_stream=False,
                                   zip_done=zip_done)

    paddle.zeros([1])
    paddle.device.synchronize()
    zip_stream.synchronize()
    zipped_out._record_stream()
    dist.all_reduce(paddle.empty([1]))

    return o1, o2, o3, zipped_out, atomic_to_zip, tokens_per_expert


def run_serial_forward(group, buffer, x, token_probs, token_indices, w_gateup, w_down):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(0)
    paddle.zeros([1])

    ################################# DISPATCH #################################
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    (
        recv_x, recv_token_indices, recv_token_probs,
        num_recv_tokens_per_expert_list, handle, event,
    ) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    tokens_per_expert = num_recv_tokens_per_expert_list
    recv_token_indices = recv_token_indices.cast("int32")
    print("tokens_per_expert:", tokens_per_expert)

    ################################## UNZIP ###################################

    (
        unzipped_tokens,
        zipped_expertwise_rowmap,
        unzipped_probs,
        unzipped_scale,
    ) = paddle.nn.functional.moe_permute(
        recv_x,
        None,  # scale
        recv_token_indices,
        recv_token_probs,
        padding_alignment=128,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    ################################### GEMM ###################################

    o1 = paddle.empty([len(unzipped_tokens), 2 * I], dtype="bfloat16")
    m_indices = paddle.concat(
        [
            paddle.full([(n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT], i, "int32")
            for i, n in enumerate(tokens_per_expert)
        ]
    )

    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(unzipped_tokens, w_gateup, o1, m_indices)

    o2 = paddlefleet_ops.fused_swiglu_scale(o1, unzipped_probs)

    o3 = paddle.empty_like(unzipped_tokens)
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(o2, w_down, o3, m_indices)

    ################################### ZIP ####################################

    zipped_tokens, zipped_probs = paddle.nn.functional.moe_unpermute(
        o3,
        zipped_expertwise_rowmap,
        recv_token_indices,
        unzipped_probs,
        total_zipped_tokens=len(recv_x),
        num_experts=E,
    )

    ################################# COMBINE ##################################

    out, _, event = buffer.combine(zipped_tokens, handle, async_finish=False,
                                   allocate_on_comm_stream=False)

    dist.all_reduce(paddle.zeros([1]))
    return o1, o2, o3, zipped_tokens


def check(out, ref, perm, offset):
    # reorder out to deep_ep order
    out = paddle.gather(out, perm + offset)
    diff = paddle.abs(
        out.float() - ref[offset : offset + out.shape[0]].float()
    )
    count = int((diff != 0).sum())
    avg, max_ = float(diff.mean()), float(diff.max())
    return count == 0, f"{count} (avg: {avg}, max: {max_})"


def main():
    group = initialize_fleet()
    configure_buffer(48)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices, w_gateup, w_down = prepare_case_inputs(group)

    if SWIGLU_FUSION:
        w_gate, w_up = w_gateup.chunk(2, axis=-1)
        w_gateup_interleaved = paddle.concat(
            [w_gate.reshape([E, H, -1, 64]), w_up.reshape(E, H, -1, 64)], axis=-1
        ).reshape([E, H, 2 * I])
    else:
        w_gateup_interleaved = w_gateup

    configure_buffer(COMM_NUM_SMS)

    # warmup
    run_overlap_forward(
        group, buffer, x, token_probs, token_indices, w_gateup_interleaved, w_down
    )
    run_serial_forward(group, buffer, x, token_probs, token_indices, w_gateup, w_down)
    paddle.base.core.nvprof_start()
    dist.all_reduce(paddle.empty([1]))

    # profile
    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"overlap_{i}")
        o1, o2, o3, oz, atomic_to_zip, tokens_per_expert = run_overlap_forward(
            group, buffer, x, token_probs, token_indices, w_gateup_interleaved, w_down
        )
        paddle.base.core.nvprof_nvtx_pop()

    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"serial_{i}")
        o1_ref, o2_ref, o3_ref, oz_ref = run_serial_forward(
            group, buffer, x, token_probs, token_indices, w_gateup, w_down
        )
        paddle.base.core.nvprof_nvtx_pop()

    paddle.device.synchronize()
    paddle.base.core.nvprof_stop()

    # validate
    offset = 0
    for i, n in enumerate(tokens_per_expert):
        # sort tokens by their positions in zipped tensor
        perm = atomic_to_zip[offset : offset + n].argsort()

        ok1, msg1 = check(o1, o1_ref, perm, offset)
        ok2, msg2 = check(o2, o2_ref, perm, offset)
        ok3, msg3 = check(o3, o3_ref, perm, offset)

        print(f"[expert {i}] o1: {msg1} o2: {msg2} o3: {msg3}",
              *([] if (ok1 and ok2 and ok3) else ["X" * 80]))

        offset += (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    print(f"[zipped_out] diff: {int((oz != oz_ref).sum())}")


if __name__ == "__main__":
    main()
