import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig)    
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (initialize_model_parallel)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.gpu_cache_engine import GPUCacheEngine
from vllm.worker.model_runner import ModelRunner, CPUModelRunner
from vllm.worker.cpu_cache_engine import CPUCacheEngine


class CPUWorker:
    def __init__(self, model_config: ModelConfig, parallel_config: ParallelConfig, scheduler_config: SchedulerConfig, rank: Optional[int] = None, distributed_init_method: Optional[str] = None,) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config

        self.model_runner = CPUModelRunner(model_config, parallel_config, scheduler_config)        
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.cpu_cache = None    

    def init_model(self) -> None:
        self.device = torch.device(f"cpu")
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        print ("loading CPU model")
        self.model_runner.load_model()

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = GPUCacheEngine(self.cache_config, self.model_config, self.parallel_config)            
        self.cpu_cache = self.cache_engine.cpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    @torch.inference_mode()
    def execute_model(self, seq_group_metadata_list: List[SequenceGroupMetadata]) -> SamplerOutput:
        if not seq_group_metadata_list:
            return {}
        output = self.model_runner.execute_model(seq_group_metadata_list, self.cpu_cache)        