import torch
import ray
import numpy as np
import mmap
import os
from typing import Dict, List, Tuple
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
        self.block_size = cache_config.block_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        self.dtype = model_config.dtype

        self.key_filename = "/tmp/key_tensor_shared.mmap"
        self.key_tensor_shape = self.get_key_block_shape()
        # key_shape = (32, 4096, 32, 16, 16, 8) 
        self.key_shape = (self.num_layers, self.num_cpu_blocks, *(self.key_tensor_shape))
        self.key_mmap_file =  self.__init_key_shared_memory()
        self.key_shared_tensor = self.create_key_shared_tensor()

        self.value_filename = "/tmp/value_tensor_shared.mmap"
        self.value_tensor_shape = self.get_value_block_shape()
        # value_tensor_size = (32, 4096, 32, 128, 16) 
        self.value_shape = (self.num_layers, self.num_cpu_blocks, *(self.value_tensor_shape))   
        self.value_mmap_file =  self.__init_value_shared_memory()     
        self.value_shared_tensor = self.create_value_shared_tensor()


    def __init_key_shared_memory(self):
        dtype = np.float16      
        itemsize = np.dtype(dtype).itemsize  
        key_buffer_size = np.prod(self.key_shape) * itemsize  
        if not os.path.exists(self.key_filename):
            with open(self.key_filename, 'wb') as f:
                 f.write(b'\x00' * key_buffer_size)
        key_mmap_file = mmap.mmap(os.open(self.key_filename, os.O_RDWR), key_buffer_size)
        return key_mmap_file

    def create_key_shared_tensor(self):
        dtype = np.float16
        key_shared_array = np.frombuffer(self.key_mmap_file, dtype=dtype).reshape(self.key_shape)
        key_shared_tensor = torch.from_numpy(key_shared_array)
        return key_shared_tensor

    def __init_value_shared_memory(self):
        dtype = np.float16      
        itemsize = np.dtype(dtype).itemsize  
        value_buffer_size = np.prod(self.value_shape) * itemsize  
        if not os.path.exists(self.value_filename):
            with open(self.value_filename, 'wb') as f:
                 f.write(b'\x00' * value_buffer_size)
        value_mmap_file = mmap.mmap(os.open(self.value_filename, os.O_RDWR), value_buffer_size)
        return value_mmap_file       

    def create_value_shared_tensor(self):
        dtype = np.float16
        value_shared_array = np.frombuffer(self.value_mmap_file, dtype=dtype).reshape(self.value_shape)
        value_shared_tensor = torch.from_numpy(value_shared_array)
        return value_shared_tensor


    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (self.num_heads, self.head_size // x, self.block_size, x,)


    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (self.num_heads, self.head_size, self.block_size,)

