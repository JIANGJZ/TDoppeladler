import copy
import os
import time
import asyncio
from functools import partial
import threading
from multiprocessing import Process, Pool, Manager
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig)        
from vllm.core.scheduler import Scheduler, MultiScheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import record_metrics, multi_worker_record_metrics
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup, SequenceGroupMetadata, SequenceGroupOutput, SequenceOutput, SequenceStatus)                     
from vllm.transformers_utils.tokenizer import (detokenize_incrementally, get_tokenizer)                           
from vllm.utils import Counter
from vllm.worker.cpu_cache_engine import CPUCacheEngine
from vllm.core.asy_submmitter import RayTaskManager, AsySubmmitterConfig


from ray.air.util.torch_dist import init_torch_dist_process_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class LLMEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(model_config.tokenizer, tokenizer_mode=model_config.tokenizer_mode, trust_remote_code=model_config.trust_remote_code, tokenizer_revision=model_config.tokenizer_revision, revision=model_config.revision)
            
        self.seq_counter = Counter()
        self.submit_counter = Counter()
        self.aux_submit_counter = Counter()

        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            # Disable Ray usage stats collection.
            ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
            if ray_usage != "1":
                os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            self._init_workers_ray(placement_group)
            self._init_cache()
        elif self.parallel_config.multi_worker:
            ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
            if ray_usage != "1":
                os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            self.task_manager = RayTaskManager()
            self._init_multiworker()
            # self._init_cpu_workers()
            self._init_multiworker_cache()
        else:
            self._init_workers(distributed_init_method)
            self._init_cache()
        
        # Create the scheduler.
        if self.parallel_config.multi_worker:
            self.scheduler = MultiScheduler(scheduler_config, cache_config, parallel_config)
            self.asy_submmitter = AsySubmmitterConfig()
        else:
            self.scheduler = Scheduler(scheduler_config, cache_config, parallel_config)
        
        # Logging.
        self.last_logging_time = time.monotonic()
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_real_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []
        

        self.total_prompt_tokens: List[Tuple[float, int]] = []
        self.total_real_prompt_tokens: List[Tuple[float, int]] = []
        self.total_generation_tokens: List[Tuple[float, int]] = []
        self.total_recompute_tokens: List[Tuple[float, int]] = []

        self.output = []


    def _init_workers(self, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker
        assert self.parallel_config.world_size == 1, ("Ray is required if parallel_config.world_size > 1.")
        self.workers: List[Worker] = []
        worker = Worker(self.model_config, self.parallel_config, self.scheduler_config, 0, distributed_init_method,)
        self.workers.append(worker)
        self._run_workers("init_model", get_all_outputs=True,)
        self._run_workers("load_model", get_all_outputs=True, max_concurrent_workers=self.parallel_config.max_parallel_loading_workers,)            

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):   
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker
        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            if self.parallel_config.tensor_parallel_size == 1:
                num_gpus = self.cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            worker = ray.remote(num_cpus=0, num_gpus=num_gpus, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_capture_child_tasks=True), **ray_remote_kwargs, )(RayWorkerVllm).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        self._run_workers("init_worker", get_all_outputs=True, worker_init_fn=lambda: Worker(model_config, parallel_config, scheduler_config, None, None,))
        self._run_workers("init_model", get_all_outputs=True,)
        self._run_workers("load_model", get_all_outputs=True, max_concurrent_workers=self.parallel_config.max_parallel_loading_workers,)

    def _init_cpu_workers(self, ):
        from vllm.worker.cpu_worker import CPUWorker
        self.cpu_workers: List[CPUWorker] = []
        worker = CPUWorker(self.model_config, self.parallel_config, self.scheduler_config,)
        self.cpu_workers.append(worker)
        self._run_cpu_workers("init_model")
        self._run_cpu_workers("load_model")

    def _init_multiworker(self, **ray_remote_kwargs):
        from vllm.worker.multi_worker import MainWorker, AuxWorker
        self.workers: List[Worker] = []

        self.main_worker = ray.remote(num_cpus=0, num_gpus=1, **ray_remote_kwargs,)(RayWorkerVllm).remote(self.model_config.trust_remote_code)
        self.workers.append(self.main_worker)   
        self.aux_worker = ray.remote(num_cpus=0, num_gpus=1, **ray_remote_kwargs,)(RayWorkerVllm).remote(self.model_config.trust_remote_code)
        self.workers.append(self.aux_worker)   

        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)

        print ("init main worker")
        self._run_main_worker("init_worker", worker_init_fn=lambda: MainWorker(model_config, parallel_config, scheduler_config, None, None,))
        self._run_main_worker("init_model")
        self._run_main_worker("load_model")

        print ("init aux worker")
        self._run_aux_worker("init_worker", worker_init_fn=lambda: AuxWorker(model_config, parallel_config, scheduler_config, None, None,))
        self._run_aux_worker("init_model")
        self._run_aux_worker("load_model")
           
    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, # CPU blocks: {num_cpu_blocks}")
                    
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.")          
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self._run_workers("warm_up_model")

    def _init_multiworker_cache(self)-> None:
        num_main_gpu_blocks, num_cpu_blocks = self._run_main_worker(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        num_aux_gpu_blocks, num_cpu_blocks = self._run_aux_worker(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )
    
        logger.info(f"# Main GPU blocks: {num_main_gpu_blocks}, # CPU blocks: {num_cpu_blocks} # Aux GPU blocks: {num_aux_gpu_blocks}")
                    
        if num_main_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.")          
        max_seq_len = self.cache_config.block_size * num_main_gpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_main_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

 
        # Initialize the cache.
        print ("init main cache engine")
        self._run_main_worker("init_cache_engine", cache_config=self.cache_config)
        self._run_main_worker("warm_up_model")  


        self.cache_config.num_gpu_blocks = num_aux_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks   

        print ("init aux cache engine")
        self._run_aux_worker("init_cache_engine", cache_config=self.cache_config)
        self._run_aux_worker("warm_up_model")

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs, distributed_init_method, placement_group, log_stats=not engine_args.disable_log_stats)    
        return engine

    def add_request(self, request_id: str, prompt: Optional[str], sampling_params: SamplingParams, prompt_token_ids: Optional[List[int]] = None, arrival_time: Optional[float] = None, ) -> None:
    
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params, arrival_time)
        # Add the sequence group to the scheduler.
        if self.scheduler_config.sorted_request:
            self.scheduler.add_sorted_seq_group(seq_group)
        else:
            self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.
        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[RequestOutput]]: 
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return seq_group_metadata_list, scheduler_outputs, [RequestOutput.from_seq_group(seq_group) for seq_group in scheduler_outputs.ignored_seq_groups]
    
    def _schedule_main(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[RequestOutput]]: 
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule_main()
        return seq_group_metadata_list, scheduler_outputs, [RequestOutput.from_seq_group(seq_group) for seq_group in scheduler_outputs.ignored_seq_groups]

    def _schedule_aux(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[RequestOutput]]: 
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule_aux()
        return seq_group_metadata_list, scheduler_outputs, [RequestOutput.from_seq_group(seq_group) for seq_group in scheduler_outputs.ignored_seq_groups]


    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset, read_offset) = detokenize_incrementally(
             self.tokenizer,
             all_input_ids=seq.get_token_ids(),
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
         )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        print (new_tokens, prefix_offset, read_offset)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence, sampling_params: SamplingParams) -> None:    
        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            print ("seq {} end with max_tokens".format(seq.seq_id))
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

    def _process_sequence_group_outputs_multi(self, seq_group: SequenceGroup, outputs: SequenceGroupOutput) -> None:         
       # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        if (len(parent_seqs) == 0 or len(samples) == 0):
            return
        parent_seq = parent_seqs[0]
        sample = samples[0]

        parent_seq.append_token_id(sample.output_token, sample.logprobs)    
        self._decode_sequence(parent_seq, seq_group.sampling_params)
        self._check_stop(parent_seq, seq_group.sampling_params)
        if parent_seq.is_finished():
            self.scheduler.free_seq(parent_seq)

    def _process_model_outputs_multi(self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs_multi(seq_group, outputs)
        
        self.scheduler.free_finished_aux_seq_groups()
        self.scheduler.free_finished_main_seq_groups()

        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups + scheduler_outputs.ignored_seq_groups):  
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            self._log_multi_worker_system_stats(scheduler_outputs.prompt_run, scheduler_outputs.num_batched_tokens)
        return request_outputs


    def _process_sequence_group_outputs(self, seq_group: SequenceGroup, outputs: SequenceGroupOutput) -> None:         
       # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        if (len(parent_seqs) == 0 or len(samples) == 0):
            return
        
        parent_seq = parent_seqs[0]
        sample = samples[0]

        parent_seq.append_token_id(sample.output_token, sample.logprobs)    
        self._decode_sequence(parent_seq, seq_group.sampling_params)
        self._check_stop(parent_seq, seq_group.sampling_params)
        if parent_seq.is_finished():
            self.scheduler.free_seq(parent_seq)

    def _process_model_outputs(self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_sequence_group_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()
        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in  (scheduled_seq_groups + scheduler_outputs.ignored_seq_groups):  
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            self._log_system_stats(scheduler_outputs.prompt_run, scheduler_outputs.num_batched_tokens, scheduler_outputs.num_real_prompt_tokens, scheduler_outputs.num_recompute_tokens)    
        return request_outputs

    def handle_main_result(self, result, callback_arg):
        # print ("handling main result")
        main_output = result
        scheduler_outputs_main = callback_arg
        main_processoutput = self._process_model_outputs_multi(main_output, scheduler_outputs_main)
        self.output.extend(main_processoutput)
        self.scheduler.set_finished_swap_out_seq_groups(scheduler_outputs_main.current_swap)

        for output in main_processoutput:
            if output.finished:
                print ("main device: {}".format(output.outputs[0].text))

    def handle_aux_result(self, result, callback_arg):
        # print ("handling aux result")
        aux_output = result
        scheduler_outputs_aux = callback_arg
        aux_processoutput = self._process_model_outputs_multi(aux_output, scheduler_outputs_aux)    
        self.output.extend(aux_processoutput)

        for output in aux_processoutput:
            if output.finished:
                print ("aux device: {}".format(output.outputs[0].text))
        

    def step(self) -> List[RequestOutput]:
        # Execute the model.
        if self.parallel_config.multi_worker:
                # print ("main list len = {}".format(len(seq_group_metadata_list_main)))
            if (self.task_manager.get_main_pending_len() < self.asy_submmitter.get_pending_length()):
                submit_id = next(self.submit_counter) 
                seq_group_metadata_list_main, scheduler_outputs_main, ignored_main = self._schedule_main()
                if (len(seq_group_metadata_list_main) > 0):
                    seq_group_metadata_list_main[0].submit_id = submit_id
                if scheduler_outputs_main.is_empty() :
                    # print ("sechduling main empty")
                    self.output.extend(ignored_main)
                else:
                    self._run_main_worker(
                        "execute_model",
                        resulthandler=self.handle_main_result,
                        callback_arg = scheduler_outputs_main,
                        seq_group_metadata_list=seq_group_metadata_list_main,
                        blocks_to_swap_out=scheduler_outputs_main.blocks_to_swap_out,
                        blocks_to_copy=scheduler_outputs_main.blocks_to_copy,
                    )
                
            if (self.task_manager.get_aux_pending_len() < self.asy_submmitter.get_pending_length()):
                submit_id = next(self.aux_submit_counter) 
                seq_group_metadata_list_aux, scheduler_outputs_aux, ignored_aux = self._schedule_aux()
                if (len(seq_group_metadata_list_aux) > 0):
                    seq_group_metadata_list_aux[0].submit_id = submit_id
                if scheduler_outputs_aux.is_empty():
                    # print ("sechduling aux empty")
                    self.output.extend(ignored_aux)
                else:
                    self._run_aux_worker(
                        "execute_model",
                        resulthandler=self.handle_aux_result,
                        callback_arg = scheduler_outputs_aux,
                        seq_group_metadata_list=seq_group_metadata_list_aux,
                        blocks_to_swap_in=scheduler_outputs_aux.blocks_to_swap_in,
                        blocks_to_copy=scheduler_outputs_aux.blocks_to_copy,
                    )
            return self.output
        else:
            seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
            if scheduler_outputs.is_empty():
                return ignored
            temp_output = self._run_workers(
                "execute_model",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
            )
            self.output = self._process_model_outputs(temp_output, scheduler_outputs)

            return self.output


    def clear_step_output(self):
        # print ("clear output")
        self.output = []

    def exit_clear(self):
        self.task_manager.stop_waiting()    

    def _log_system_stats(self, prompt_run: bool, num_batched_tokens: int, num_real_prompt_tokens: int, num_recompute_tokens:int,) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
            self.num_real_prompt_tokens.append((now, num_real_prompt_tokens))
            self.total_prompt_tokens.append((now, num_batched_tokens))
            self.total_real_prompt_tokens.append((now, num_real_prompt_tokens))
            self.total_recompute_tokens.append((now, num_recompute_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))
            self.total_generation_tokens.append((now, num_batched_tokens))

        should_log = now - self.last_logging_time >= _LOGGING_INTERVAL_SEC
        if not should_log:
            return


        total_prompt_tokens = sum(n for _, n in self.total_prompt_tokens[:-1])    
        total_real_prompt_tokens = sum(n for _, n in self.total_real_prompt_tokens[:-1])    
        total_generation_tokens = sum(n for _, n in self.total_generation_tokens[:-1])  
        total_recompute_tokens =  sum(n for _, n in self.total_recompute_tokens[:-1]) 
        total_final_prompt_tokens =  total_real_prompt_tokens - total_recompute_tokens

        print ((now, self.last_logging_time, self.total_prompt_tokens[0][0]))
        avg_total_prompt_throughput = total_prompt_tokens / (now - self.total_prompt_tokens[0][0])
        avg_total_real_prompt_throughput = total_real_prompt_tokens / (now - self.total_real_prompt_tokens[0][0])
        avg_total_final_prompt_throughput = total_final_prompt_tokens / (now - self.total_real_prompt_tokens[0][0])
        avg_total_generation_throughput = total_generation_tokens / (now - self.total_generation_tokens[0][0])

        # Discard the old stats.
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens if now - t < _LOGGING_INTERVAL_SEC]         
        self.num_real_prompt_tokens = [(t, n) for t, n in self.num_real_prompt_tokens if now - t < _LOGGING_INTERVAL_SEC]             
        self.num_generation_tokens = [(t, n) for t, n in self.num_generation_tokens if now - t < _LOGGING_INTERVAL_SEC]


        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window

            total_num_tokens = sum(n for _, n in self.num_real_prompt_tokens[:-1])
            window = now - self.num_real_prompt_tokens[0][0]
            avg_real_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
            avg_real_prompt_throughput = 0.0

        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0


        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        print ("total_num_gpu_blocks = {}, num_free_gpu_blocks={} num_used_gpu_blocks={} ".format(total_num_gpu_blocks, num_free_gpu_blocks, num_used_gpu_blocks ))
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        record_metrics(
            avg_prompt_throughput=avg_prompt_throughput,
            avg_real_prompt_throughput=avg_real_prompt_throughput,
            avg_generation_throughput=avg_generation_throughput,
            scheduler_running=len(self.scheduler.running),
            scheduler_swapped=len(self.scheduler.swapped),
            scheduler_waiting=len(self.scheduler.waiting),
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
        )

        logger.info("Avg past 5 seconds prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg past 5 seconds real prompt throughput: "
                    f"{avg_real_prompt_throughput:.1f} tokens/s, "
                    "Avg past 5 seconds generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    "total prompt throughput: "
                    f"{total_prompt_tokens:.1f} tokens/s, "
                    "total real prompt throughput: "
                    f"{total_real_prompt_tokens:.1f} tokens/s, "
                    "total final prompt throughput: "
                    f"{total_final_prompt_tokens:.1f} tokens/s, "
                    "total generation throughput: "
                    f"{total_generation_tokens:.1f} tokens/s, "
                    "avg total prompt throughput: "
                    f"{avg_total_prompt_throughput:.1f} tokens/s, "
                    "avg total real prompt throughput: "
                    f"{avg_total_real_prompt_throughput:.1f} tokens/s, "
                    "avg total final prompt throughput: "
                    f"{avg_total_final_prompt_throughput:.1f} tokens/s, "
                    "avg total generation throughput: "
                    f"{avg_total_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Waiting: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now


    def _log_multi_worker_system_stats(self, prompt_run: bool, num_batched_tokens: int) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        should_log = now - self.last_logging_time >= _LOGGING_INTERVAL_SEC
        if not should_log:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens if now - t < _LOGGING_INTERVAL_SEC]           
        self.num_generation_tokens = [(t, n) for t, n in self.num_generation_tokens if now - t < _LOGGING_INTERVAL_SEC]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        multi_worker_record_metrics(
            avg_prompt_throughput=avg_prompt_throughput,
            avg_generation_throughput=avg_generation_throughput,
            scheduler_running=len(self.scheduler.running),
            scheduler_swapped=len(self.scheduler.swapped),
            scheduler_waiting=len(self.scheduler.waiting),
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
        )

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now


    def _run_workers_in_batch(self, workers, method: str, *args, **kwargs, ):
        all_outputs = []
        for worker in workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        return all_outputs

    def _run_workers(self, method: str, *args, get_all_outputs: bool = False, max_concurrent_workers: Optional[int] = None, **kwargs, ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [self.workers[i:i + max_concurrent_workers] for i in range(0, len(self.workers), max_concurrent_workers)]
        else:
            work_groups = [self.workers]

        for workers in work_groups:
            all_outputs.extend(self._run_workers_in_batch(workers, method, *args, **kwargs))

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

    def _run_main_worker(self, method: str, resulthandler=None, callback_arg=None, *args,**kwargs) -> Any:
        executor = partial(self.main_worker.execute_method.remote, method)
        if resulthandler == None:
            output = self.task_manager.apply(func=executor, args=args, kwargs=kwargs)
            return output
        else:
            self.task_manager.apply_async_main(func=executor, args=args, kwargs=kwargs, callback=resulthandler, callback_arg=callback_arg)

    def _run_aux_worker(self, method: str, resulthandler=None, callback_arg=None, *args,**kwargs) -> Any:
        executor = partial(self.aux_worker.execute_method.remote, method)
        if resulthandler == None:
            output = self.task_manager.apply(func=executor, args=args, kwargs=kwargs)
            return output
        else:
            self.task_manager.apply_async_aux(func=executor, args=args, kwargs=kwargs, callback=resulthandler, callback_arg=callback_arg)

    def _run_cpu_workers(self, method: str, *args, **kwargs, ) -> Any:
        executor = getattr(self.aux_worker, method)
        output = executor(*args, **kwargs)
        return output    
    
# vicuna-7b
# num_main_gpu_blocks = 954
# num_cpu_blocks = 1024
# num_aux_gpu_blocks = 954

# baichuang-7b
# num_main_gpu_blocks = 750
# num_cpu_blocks = 8192
# num_aux_gpu_blocks = 750

#aquila-7b
# num_main_gpu_blocks = 670
# num_cpu_blocks = 10000
# num_aux_gpu_blocks = 670