from typing import Dict, List, Tuple
import torch
from vllm._C import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
logger = init_logger(__name__)
KVCache = Tuple[torch.Tensor, torch.Tensor]
import ray
import numpy as np
import mmap
import os


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
        # self.key_tensor_shape = self.get_key_block_shape()
        # print ("key_tensor_shape = {}".format(self.key_tensor_shape))
        # self.key_block_shape = (self.num_cpu_blocks, *(self.key_tensor_shape))
        # print ("key_block_shape = {}".format(self.key_block_shape))
        # self.value_tensor_shape = self.get_value_block_shape()
        # print ("value_tensor_shape = {}".format(self.value_tensor_shape))
        # self.value_block_shape = (self.num_cpu_blocks, *(self.value_tensor_shape))
        # print ("value_block_shape = {}".format(self.value_block_shape))
        # self.item_size = np.dtype(np.float16).itemsize
        # print ("item_size = {}".format(self.item_size))
        # self.key_block_size = (np.prod(self.key_block_shape)+1) * self.item_size
        # print ("key_block_size = {}".format(self.key_block_size))        
        # self.value_block_size = (np.prod(self.value_block_shape)+1) * self.item_size
        # print ("value_block_size = {}".format(self.value_block_size))         
        # self.num_elements = self.num_layers
        # print ("num_elements = {}".format(self.num_elements))
        # self.kv_pair_buffer_size = self.key_block_size + self.value_block_size
        # print ("kv_pair_buffer_size = {}".format(self.kv_pair_buffer_size))
        # self.total_buffer_size = self.num_elements * self.kv_pair_buffer_size
        # print ("total_buffer_size = {}".format(self.total_buffer_size))        
        # self.filename = "/tmp/kv_tensor_shared.mmap"

        # self.kv_mmap_file = self.__init_shared_kvcache()
        # self.cpu_cache = self.create_kv_shared_tensor()

        # self.cpu_cache = self.allocate_cpu_cache()

        self.key_mmap_file =  self.__init_key_shared_memory()
        self.key_shared_tensor = self.create_key_shared_tensor()
        self.value_mmap_file =  self.__init_value_shared_memory()
        self.value_shared_tensor = self.create_value_shared_tensor()


    def __init_key_shared_memory(self):
        key_tensor_size = (32, 4096, 32, 16, 16, 8) 
        dtype = np.float16      
        itemsize = np.dtype(dtype).itemsize  
        key_buffer_size = np.prod(key_tensor_size) * itemsize  
        key_filename = "/tmp/key_tensor_shared.mmap"
        if not os.path.exists(key_filename):
            with open(key_filename, 'wb') as f:
                 f.write(b'\x00' * key_buffer_size)
        key_mmap_file = mmap.mmap(os.open(key_filename, os.O_RDWR), key_buffer_size)
        return key_mmap_file

    def create_key_shared_tensor(self):
        key_tensor_size = (32, 4096, 32, 16, 16, 8)
        dtype = np.float16
        key_shared_array = np.frombuffer(self.key_mmap_file, dtype=dtype).reshape(key_tensor_size)
        key_shared_tensor = torch.from_numpy(key_shared_array)
        key_shared_tensor[0, 0, 0, 0, 0, 0] = 4
        return key_shared_tensor

    def __init_value_shared_memory(self):
        value_tensor_size = (32, 4096, 32, 128, 16) 
        dtype = np.float16      
        itemsize = np.dtype(dtype).itemsize  
        value_buffer_size = np.prod(value_tensor_size) * itemsize  
        value_filename = "/tmp/value_tensor_shared.mmap"
        if not os.path.exists(value_filename):
            with open(value_filename, 'wb') as f:
                 f.write(b'\x00' * value_buffer_size)
        value_mmap_file = mmap.mmap(os.open(value_filename, os.O_RDWR), value_buffer_size)
        return value_mmap_file       

    def create_value_shared_tensor(self):
        value_tensor_size = (32, 4096, 32, 128, 16) 
        dtype = np.float16
        value_shared_array = np.frombuffer(self.value_mmap_file, dtype=dtype).reshape(value_tensor_size)
        value_shared_tensor = torch.from_numpy(value_shared_array)
        value_shared_tensor[0, 0, 0, 0, 0] = 4
        return value_shared_tensor

    def set_shared_tensor(self):
        self.key_shared_tensor[0, 0, 0, 0, 0, 0] = 5 
        self.value_shared_tensor[0, 0, 0, 0, 0] = 5 

    def get_shared_tensor(self):
        return self.key_shared_tensor, self.value_shared_tensor


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


    def __init_shared_kvcache(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'wb') as f:
                 f.write(b'\x00' * self.total_buffer_size)

        with open(self.filename, 'r+b') as f:
            kv_mmap_file = mmap.mmap(f.fileno(), self.total_buffer_size, access=mmap.ACCESS_WRITE)
        return kv_mmap_file


    def __retrieve_kv_pair(self, index):
        start_pos = index * self.kv_pair_buffer_size
        key_pos = start_pos
        value_pos = start_pos + self.key_block_size
        print ("kv pair key_start_pos = {}, end_pos={}".format(key_pos, key_pos + self.key_block_size-2))
        np_key = np.frombuffer(self.kv_mmap_file[key_pos:key_pos + self.key_block_size-2], dtype=np.float16).reshape(self.key_block_shape)
        print ("kv pair value_start_pos = {}, end_pos={}".format(value_pos, value_pos + self.value_block_size-2))
        np_value = np.frombuffer(self.kv_mmap_file[value_pos:value_pos + self.value_block_size-2], dtype=np.float16).reshape(self.value_block_shape)
        key_block = torch.from_numpy(np_key)
        value_block = torch.from_numpy(np_value)
        
        return key_block, value_block

    def create_kv_shared_tensor(self):
        cpu_cache: List[KVCache] = []
        for i in range(self.num_elements):
            key_blocks, value_blocks = self.__retrieve_kv_pair(i)
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache
