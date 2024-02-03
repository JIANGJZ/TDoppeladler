from typing import Dict, List, Tuple
import torch
from vllm._C import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
logger = init_logger(__name__)
KVCache = Tuple[torch.Tensor, torch.Tensor]

class CPUCacheEngine:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig, parallel_config: ParallelConfig,) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.cpu_cache = self.allocate_cpu_cache()

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (self.num_heads, self.head_size // x, self.block_size, x,)


    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (self.num_heads, self.head_size, self.block_size,)


    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = True        
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            key_blocks.share_memory_()
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks.share_memory_()
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache
