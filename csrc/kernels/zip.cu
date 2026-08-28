#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace zip {

// The largest `topk` a single warp can rank without spilling into multiple rounds
constexpr int kNumMaxTopk = 32;

// A token's slots have to be accumulated in ascending local-expert order, so that the sum is
// reproducible no matter which order the experts happened to finish in
template <int kNumThreads>
__global__ __launch_bounds__(kNumThreads, 1) void zip(int4* combine_input,
                                                      const int4* o3,
                                                      const int* zip_to_atomic,
                                                      const topk_idx_t* recv_topk_idx,
                                                      const int* zip_task_queue,
                                                      int* zip_done,
                                                      int num_recv_tokens,
                                                      int num_topk,
                                                      int hidden_int4,
                                                      int num_ctas) {
    const auto cta_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32;
    const auto lane_id = get_lane_id();

    EP_DEVICE_ASSERT(num_topk <= kNumMaxTopk);

    // Each CTA owns a whole token, and the CTAs take turns over the queue by index parity. With a
    // realistic `hidden` one CTA already has enough work to fill its threads, so this beats slicing
    // the hidden dimension: one release per token instead of one per CTA per token
    // NOTES: no claiming is needed, as every CTA knows up front which tasks are its own

    // The accumulation order, rebuilt by warp 0 for every token
    __shared__ int smem_slots[kNumMaxTopk];
    __shared__ int smem_num_slots;
    __shared__ int smem_token_idx;

    // Persistent: the read pointer is just this loop variable, strided by the number of CTAs
    for (int task_idx = cta_id; task_idx < num_recv_tokens; task_idx += num_ctas) {
        if (warp_id == 0) {
            // The queue is initialized to -1, so a non-negative entry is the ready signal itself.
            // This is the acquire that publishes everything the compute side wrote for this token
            int token_idx = -1;
            if (lane_id == 0) {
                auto start_time = clock64();
                while ((token_idx = ld_acquire_global(zip_task_queue + task_idx)) < 0) {
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP zip timeout waiting for a task, CTA: %d, task: %d, num tasks: %d\n",
                               cta_id,
                               task_idx,
                               num_recv_tokens);
                        trap();
                    }
                }
            }
            token_idx = __shfl_sync(0xffffffff, token_idx, 0);

            // NOTES: validity must come from `zip_to_atomic`, which is pre-filled with -1, and not from
            // counting `recv_topk_idx`'s row: the row's unused slots are never written by the receiver,
            // and its written slots are only published by their own expert's chunk
            int slot = -1, expert_idx = INT_MAX;
            if (lane_id < num_topk) {
                slot = zip_to_atomic[token_idx * num_topk + lane_id];
                if (slot >= 0)
                    expert_idx = static_cast<int>(recv_topk_idx[token_idx * num_topk + lane_id]);
            }

            // Rank each valid slot by its expert index. A token never selects the same expert twice,
            // so the valid keys are distinct and this is a total order; the invalid lanes sit at
            // `INT_MAX` and are simply never counted nor written
            int rank = 0;
            #pragma unroll
            for (int i = 0; i < 32; ++i)
                rank += (__shfl_sync(0xffffffff, expert_idx, i) < expert_idx) ? 1 : 0;
            auto num_slots = __popc(__ballot_sync(0xffffffff, slot >= 0));
            if (slot >= 0)
                smem_slots[rank] = slot;

            if (lane_id == 0) {
                smem_num_slots = num_slots;
                smem_token_idx = token_idx;
            }
        }
        __syncthreads();

        const auto num_slots = smem_num_slots;
        const auto token_idx = smem_token_idx;
        for (int i = thread_id; i < hidden_int4; i += kNumThreads) {
            // Accumulate in fp32, one `int4` (8 bf16 channels) per thread per step
            float acc[8] = {0};
            for (int s = 0; s < num_slots; ++s) {
                auto value = o3[static_cast<int64_t>(smem_slots[s]) * hidden_int4 + i];
                auto channels = reinterpret_cast<nv_bfloat16*>(&value);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[j] += __bfloat162float(channels[j]);
            }

            int4 out;
            auto channels = reinterpret_cast<nv_bfloat16*>(&out);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                channels[j] = __float2bfloat16(acc[j]);
            st_na_global(combine_input + static_cast<int64_t>(token_idx) * hidden_int4 + i, out);
        }

        // Barrier first, so that thread 0 inherits the whole CTA's stores: a release publishes
        // everything that happens-before it, not just the writes of the thread executing it.
        // A single CTA owns the token, so `zip_done` is a plain 0/1 flag
        __syncthreads();
        if (thread_id == 0)
            st_na_release(zip_done + token_idx, 1);
    }
}

void zip(int4* combine_input,
         const int4* o3,
         const int* zip_to_atomic,
         const topk_idx_t* recv_topk_idx,
         const int* zip_task_queue,
         int* zip_done,
         int num_recv_tokens,
         int num_topk,
         int hidden_int4,
         int num_ctas,
         cudaStream_t stream) {
    constexpr int kNumThreads = 512;
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    SETUP_LAUNCH_CONFIG(num_ctas, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg,
                  (zip<kNumThreads>),
                  combine_input,
                  o3,
                  zip_to_atomic,
                  recv_topk_idx,
                  zip_task_queue,
                  zip_done,
                  num_recv_tokens,
                  num_topk,
                  hidden_int4,
                  num_ctas);
}

}  // namespace zip

}  // namespace deep_ep
