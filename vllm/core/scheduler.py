import enum
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union
import bisect

from vllm.config import CacheConfig, SchedulerConfig, CPUSchedulerConfig, ParallelConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.alloc_status import AllocStatus
from vllm.core.multi_block_manager import MultiBlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup, SequenceGroupMetadata, SequenceStatus)
                           

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.
    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()

class SchedulerOutputs:
    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        num_real_prompt_tokens: int,
        num_recompute_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.num_real_prompt_tokens = num_real_prompt_tokens
        self.num_recompute_tokens = num_recompute_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in and not self.blocks_to_swap_out and not self.blocks_to_copy)
                
    def __repr__(self) -> str:
        return (f"scheduled_seq_groups (request_id={self.scheduled_seq_groups}, "
                f"prompt_run={self.prompt_run}, ")

class Scheduler:
    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig, cpuscheduler_config: CPUSchedulerConfig, parallel_config: ParallelConfig) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config

        self.prompt_limit = min(self.scheduler_config.max_model_len, self.scheduler_config.max_num_batched_tokens)
        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []


        self.swapped_num = 0

    def print_all_waiting(self):
        temp = []
        for group in self.waiting:
            temp.append(group.get_prompt_length())
        print (temp)

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        # index = bisect.bisect_left([group.get_prompt_length() for group in self.waiting], seq_group.get_prompt_length())
        # self.waiting.insert(index, seq_group)    
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs() for seq_group in self.running)
            seq_lens: List[int] = []
            seq_recompute_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]
                waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, ("Waiting sequence group should have only one prompt sequence.")
                num_prompt_tokens = waiting_seqs[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(f"Input prompt ({num_prompt_tokens} tokens) is too long and exceeds limit of {self.prompt_limit}")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(f"Input prompt ({num_prompt_tokens} tokens) is too long and exceeds the capacity of block_manager")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                new_seq_recompute_lens = seq_recompute_lens
                if (seq_group.is_recompute):
                    new_seq_recompute_lens = new_seq_recompute_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens > self.scheduler_config.max_num_batched_tokens):
                    print ("promt select exceed tokens total_tokens={}, new_tokens={}".format(num_batched_tokens, num_prompt_tokens))
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs):
                    print ("promt select exit seqs total_seqs={}, num_curr_seqs={}".format(num_curr_seqs + num_new_seqs, num_curr_seqs))
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    print ("promt select exit padding exceed_paddings={}, cur_paddings={}, exceeed_seq={}".format(num_paddings, len(seq_lens) * max(seq_lens) - sum(seq_lens), num_prompt_tokens))
                    break
                seq_lens = new_seq_lens
                seq_recompute_lens = new_seq_recompute_lens
                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled or ignored_seq_groups:
                print ("==================== send to prefill ===============")
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) * max(seq_lens) if seq_lens else 0,
                    num_real_prompt_tokens = sum(seq_lens),
                    num_recompute_tokens = sum(seq_recompute_lens),
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs() for seq_group in self.running)
            while self.swapped:
                seq_group = self.swapped[0]
                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(seq_group.num_seqs(status=SequenceStatus.RUNNING) for seq_group in self.running)
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            num_real_prompt_tokens = 0,
            num_recompute_tokens = 0,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [seq_group for seq_group in self.running if not seq_group.is_finished()]

    def free_finished_seq_groups_cpu(self) -> None:
        self.running_cpu = [seq_group for seq_group in self.running_cpu if not seq_group.is_finished()]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slot(self, seq_group: SequenceGroup, blocks_to_copy: Dict[int, List[int]],) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int],  preemption_mode: Optional[PreemptionMode] = None,) -> None:
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(self, seq_group: SequenceGroup,) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        seq_group.set_recompute(True)
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int],) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(self, seq_group: SequenceGroup, blocks_to_swap_in: Dict[int, int],) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int],) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError("Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

class MultiScheduler:
    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig, cpuscheduler_config: CPUSchedulerConfig, parallel_config: ParallelConfig) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config

        self.prompt_limit = min(self.scheduler_config.max_model_len, self.scheduler_config.max_num_batched_tokens)
        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = MultiBlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_main_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            num_aux_gpu_blocks=self.cache_config.num_gpu_blocks,
            sliding_window=self.cache_config.sliding_window)


        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

        # Seqence groups swapped to CPU for computing
        self.waiting_cpu: List[SequenceGroup] = []
        self.running_cpu: List[SequenceGroup] = []

        self.running_aux: List[SequenceGroup] = []

        self.swapped_num = 0

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        # index = bisect.bisect_left([group.get_prompt_length() for group in self.waiting], seq_group.get_prompt_length())
        # self.waiting.insert(index, seq_group)    
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped, self.running_aux]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.running_aux

    def get_num_finished_seq_groups(self) -> int:
        return (self.scheduler_config.num_prompts - self.get_num_unfinished_seq_groups())


    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + len(self.running_aux)

    def _schedule_cpu(self)-> SchedulerOutputs:
        now = time.monotonic()

        num_curr_seqs = sum(seq_group.get_max_num_running_seqs() for seq_group in self.running_cpu)
        while self.waiting_cpu:
            seq_group = self.waiting_cpu[0]
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING_CPU)

    def _schedule_aux(self)->SchedulerOutputs:
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        now = time.monotonic()

        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs() for seq_group in self.running_aux)
        while self.swapped:
            seq_group = self.swapped[0]
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.check_in_main(seq_group):
                self.swapped.pop(0)
                continue

            if not self.block_manager.can_swap_in(seq_group):
                print ("can not swap in")
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs):
                print ("execeed max_num_seqs = {}".format(self.scheduler_config.max_num_seqs))
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_aux_slot(seq_group, blocks_to_copy)
            num_curr_seqs += num_new_seqs
            self.running_aux.append(seq_group)

        num_batched_tokens = sum(seq_group.num_seqs(status=SequenceStatus.RUNNING) for seq_group in self.running_aux)
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running_aux,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            num_real_prompt_tokens = 0,
            num_recompute_tokens = 0,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs


    def _schedule_main(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()
        print ("waiting = {}, running = {}, swapped = {}, running_aux = {}".format(len(self.waiting), len(self.running), len(self.swapped), len(self.running_aux)))
        finished_seqs = self.get_num_finished_seq_groups()
        unfinished_seqs = self.get_num_unfinished_seq_groups()
        print ("finished = {}, unfinished = {}".format(finished_seqs, unfinished_seqs))
        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs() for seq_group in self.running)
            seq_lens: List[int] = []
            seq_recompute_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]
                waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, ("Waiting sequence group should have only one prompt sequence.")
                num_prompt_tokens = waiting_seqs[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(f"Input prompt ({num_prompt_tokens} tokens) is too long and exceeds limit of {self.prompt_limit}")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(f"Input prompt ({num_prompt_tokens} tokens) is too long and exceeds the capacity of block_manager")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                new_seq_recompute_lens = seq_recompute_lens
                if (seq_group.is_recompute):
                    new_seq_recompute_lens = new_seq_recompute_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens > self.scheduler_config.max_num_batched_tokens):
                    print ("prompt select exceed tokens total_tokens={}, new_tokens={}".format(num_batched_tokens, num_prompt_tokens))
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs):
                    print ("prompt select exit seqs total_seqs={}, num_curr_seqs={}".format(num_curr_seqs + num_new_seqs, num_curr_seqs))
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > 0.01*self.scheduler_config.max_paddings:
                    print ("prompt select exit padding exceed_paddings={}, cur_paddings={}, exceeed_seq={}".format(num_paddings, len(seq_lens) * max(seq_lens) - sum(seq_lens), num_prompt_tokens))
                    break
                seq_lens = new_seq_lens
                seq_recompute_lens = new_seq_recompute_lens
                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled or ignored_seq_groups:
                print ("==================== send to prefill ===============")
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=len(seq_lens) * max(seq_lens) if seq_lens else 0,
                    num_real_prompt_tokens = sum(seq_lens),
                    num_recompute_tokens = sum(seq_recompute_lens),
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        self.running = self.policy.sort_by_priority(now, self.running)
        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            if not self.block_manager.check_in_main(seq_group):
                continue
            while not self.block_manager.can_append_main_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    if (len(self.running) > 0):
                        victim_seq_group = self.running.pop(-1)
                        self._preempt(victim_seq_group, blocks_to_swap_out)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    break
            else:
                if (len(self.running) > 1.5*len(self.running_aux)):
                    self._preempt(seq_group, blocks_to_swap_out)
                else:
                    self._append_main_slot(seq_group, blocks_to_copy)
                    running.append(seq_group)
        self.running = running

        num_batched_tokens = sum(seq_group.num_seqs(status=SequenceStatus.RUNNING) for seq_group in self.running)
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            num_real_prompt_tokens = 0,
            num_recompute_tokens = 0,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule_cpu(self)-> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        scheduler_outputs = self._schedule_cpu()

    def schedule_aux(self)-> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        scheduler_outputs = self._schedule_aux()
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_aux_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            # print (" sechedule_aux block_tables = {}".format(block_tables))
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def schedule_main(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        scheduler_outputs = self._schedule_main()
        seq_group_metadata_list: List[SequenceGroupMetadata] = []

        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_main_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            # print (" sechedule_main block_tables = {}".format(block_tables))
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_main_seq_groups(self) -> None:
        self.running = [seq_group for seq_group in self.running if not seq_group.is_finished()]

    def free_finished_aux_seq_groups(self) -> None:
        self.running_aux = [seq_group for seq_group in self.running_aux if not seq_group.is_finished()]

    def free_finished_seq_groups_cpu(self) -> None:
        self.running_cpu = [seq_group for seq_group in self.running_cpu if not seq_group.is_finished()]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_main_slot(self, seq_group: SequenceGroup, blocks_to_copy: Dict[int, List[int]],) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_main_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _append_aux_slot(self, seq_group: SequenceGroup, blocks_to_copy: Dict[int, List[int]],) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_aux_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int],  preemption_mode: Optional[PreemptionMode] = None,) -> None:
        self._preempt_by_swap(seq_group, blocks_to_swap_out)

    def _preempt_by_swap(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int],) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped_num += 1
        if self.swapped_num % 100000 == 0:
            self.waiting_cpu.append(seq_group)
        else:
            self.swapped.append(seq_group)

    def _swap_in(self, seq_group: SequenceGroup, blocks_to_swap_in: Dict[int, int],) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(self, seq_group: SequenceGroup, blocks_to_swap_out: Dict[int, int],) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            raise RuntimeError("Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED    