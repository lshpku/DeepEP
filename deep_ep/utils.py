import os
import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple

# noinspection PyUnresolvedReferences
import deep_ep_cpp
# noinspection PyUnresolvedReferences
from deep_ep_cpp import EventHandle

import paddle


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(self, event: Optional[EventHandle] = None, extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

    def calc_stream_wait(self, group_idx) -> None:
        self.event.calc_stream_wait(group_idx)

    def comm_stream_wait(self, group_idx) -> None:
        self.event.comm_stream_wait(group_idx)

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap():
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()


def check_nvlink_connections(group):
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if 'PCIE' in torch.cuda.get_device_name():
        assert group.size() <= 2, 'PCIe GPUs only have pairwise NVLink connections'

        # noinspection PyUnresolvedReferences
        import pynvml
        pynvml.nvmlInit()

        # noinspection PyTypeChecker
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [
            0,
        ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                assert status == pynvml.NVML_P2P_STATUS_OK,\
                    f'GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink'

        # Close NVML
        pynvml.nvmlShutdown()


def get_event_from_comm_stream(group_id: int) -> EventOverlap:
    return EventOverlap(
        event=paddle.base.core.get_event_handle_from_comm_stream(group_id)
    )


def zip(o3: torch.Tensor, zip_to_atomic: torch.Tensor, recv_topk_idx: torch.Tensor,
        zip_task_queue: torch.Tensor, zip_done: torch.Tensor, num_ctas: int = 4) -> torch.Tensor:
    """
    Accumulate each token's expert outputs back into the DeepEP order, producing combine's input.

    This is a persistent kernel launched on the current (compute) stream, so it is meant to be called
    right after `dispatch(async_finish=True)` and to run while dispatch and the experts are still going.
    Each CTA owns a whole token and picks up the queue entries where `task_idx % num_ctas == cta_id`,
    and a token's slots are summed in ascending local-expert order so the result is reproducible.

    Arguments:
        o3: `[num_unzipped_tokens, hidden]` bf16, the experts' outputs in atomic order.
        zip_to_atomic: `[num_recv_tokens, topk]` int32 from dispatch, -1 for the invalid slots.
        recv_topk_idx: `[num_recv_tokens, topk]` from dispatch, the local expert indices.
        zip_task_queue: `[num_recv_tokens]` int32 filled by the compute side, initialized to -1.
        zip_done: `[num_recv_tokens]` int32, zeroed by the caller, set to 1 once a token is zipped.
        num_ctas: how many SMs to use.

    Returns:
        combine_input: `[num_recv_tokens, hidden]` bf16, in DeepEP order.
    """
    return deep_ep_cpp.zip(o3, zip_to_atomic, recv_topk_idx, zip_task_queue, zip_done, num_ctas)
