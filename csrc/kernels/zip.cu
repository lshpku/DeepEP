#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace zip {

// The largest `topk` a single warp can rank without spilling into multiple rounds
constexpr int kNumMaxTopk = 32;

// How many `int4` each lane accumulates per step, and how many `bfloat162` an `int4` holds
constexpr int kNumVecsPerLane = 4;
constexpr int kNumPairsPerInt4 = 4;
constexpr int kNumInt4PerStep = 32 * kNumVecsPerLane;

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

    constexpr int kNumWarpsPerCta = kNumThreads / 32;
    const auto num_warps = num_ctas * kNumWarpsPerCta;
    const auto global_warp_id = cta_id * kNumWarpsPerCta + warp_id;

    EP_DEVICE_ASSERT(num_topk <= kNumMaxTopk);

    // Ablation: each *warp* owns a whole token, so the accumulation order is private to the warp and
    // no CTA-wide barrier is needed at all. The task assignment stays static, only the granularity
    // changes: warp `w` takes the queue entries with `task_idx % num_warps == w`
    // NOTES: no claiming is needed, as every warp knows up front which tasks are its own

    // The accumulation order, one private row per warp
    __shared__ int smem_slots[kNumWarpsPerCta][kNumMaxTopk];
    auto warp_slots = smem_slots[warp_id];

    // Persistent: the read pointer is just this loop variable, strided by the number of warps
    for (int task_idx = global_warp_id; task_idx < num_recv_tokens; task_idx += num_warps) {
        // The queue is initialized to -1, so a non-negative entry is the ready signal itself.
        // This is the acquire that publishes everything the compute side wrote for this token
        int token_idx = -1;
        if (lane_id == 0) {
            auto start_time = clock64();
            while ((token_idx = ld_acquire_global(zip_task_queue + task_idx)) < 0) {
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP zip timeout waiting for a task, CTA: %d, warp: %d, task: %d, num tasks: %d\n",
                           cta_id,
                           warp_id,
                           task_idx,
                           num_recv_tokens);
                    trap();
                }
                __nanosleep(64);
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
            warp_slots[rank] = slot;
        __syncwarp();

        // Four `int4` per lane per step, laid out so that each of the four loads is still one fully
        // coalesced 32-lane transaction: lane `l` takes `base + l`, `base + 32 + l`, ...
        // NOTES: the point is memory-level parallelism, i.e. four independent 128-bit loads in flight
        // per lane instead of one, so the `num_slots` loop is no longer serialized on a single miss.
        // The step loop must stay free of any per-`k` predicate, or the unroll breaks down and the
        // accumulators get indexed dynamically, i.e. spilled to local memory; the remainder is
        // handled by a scalar tail instead
        const auto num_vectorized = hidden_int4 / kNumInt4PerStep * kNumInt4PerStep;
        for (int base = lane_id; base < num_vectorized; base += kNumInt4PerStep) {
            // Accumulate in fp32, kept as `float2` so a whole `bfloat162` is converted at a time
            float2 acc[kNumVecsPerLane][kNumPairsPerInt4];
            #pragma unroll
            for (int k = 0; k < kNumVecsPerLane; ++k)
                #pragma unroll
                for (int j = 0; j < kNumPairsPerInt4; ++j)
                    acc[k][j] = make_float2(0.0f, 0.0f);

            for (int s = 0; s < num_slots; ++s) {
                auto row = o3 + static_cast<int64_t>(warp_slots[s]) * hidden_int4 + base;
                #pragma unroll
                for (int k = 0; k < kNumVecsPerLane; ++k) {
                    auto value = row[k * 32];
                    auto pairs = reinterpret_cast<nv_bfloat162*>(&value);
                    #pragma unroll
                    for (int j = 0; j < kNumPairsPerInt4; ++j) {
                        auto pair = __bfloat1622float2(pairs[j]);
                        acc[k][j].x += pair.x;
                        acc[k][j].y += pair.y;
                    }
                }
            }

            auto dst = combine_input + static_cast<int64_t>(token_idx) * hidden_int4 + base;
            #pragma unroll
            for (int k = 0; k < kNumVecsPerLane; ++k) {
                int4 out;
                auto pairs = reinterpret_cast<nv_bfloat162*>(&out);
                #pragma unroll
                for (int j = 0; j < kNumPairsPerInt4; ++j)
                    pairs[j] = __float22bfloat162_rn(acc[k][j]);
                st_na_global(dst + k * 32, out);
            }
        }

        // Scalar tail, one `int4` per lane
        for (int i = num_vectorized + lane_id; i < hidden_int4; i += 32) {
            float2 acc[kNumPairsPerInt4];
            #pragma unroll
            for (int j = 0; j < kNumPairsPerInt4; ++j)
                acc[j] = make_float2(0.0f, 0.0f);

            for (int s = 0; s < num_slots; ++s) {
                auto value = o3[static_cast<int64_t>(warp_slots[s]) * hidden_int4 + i];
                auto pairs = reinterpret_cast<nv_bfloat162*>(&value);
                #pragma unroll
                for (int j = 0; j < kNumPairsPerInt4; ++j) {
                    auto pair = __bfloat1622float2(pairs[j]);
                    acc[j].x += pair.x;
                    acc[j].y += pair.y;
                }
            }

            int4 out;
            auto pairs = reinterpret_cast<nv_bfloat162*>(&out);
            #pragma unroll
            for (int j = 0; j < kNumPairsPerInt4; ++j)
                pairs[j] = __float22bfloat162_rn(acc[j]);
            st_na_global(combine_input + static_cast<int64_t>(token_idx) * hidden_int4 + i, out);
        }

        // Barrier first, so that lane 0 inherits the whole warp's stores: a release publishes
        // everything that happens-before it, not just the writes of the thread executing it.
        // A single warp owns the token, so `zip_done` is a plain 0/1 flag
        __syncwarp();
        if (lane_id == 0)
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
    constexpr int kNumThreads = 1024;
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
