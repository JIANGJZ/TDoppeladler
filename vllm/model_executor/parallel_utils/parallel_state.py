"""Tensor and pipeline parallel groups."""

import torch
from vllm.config import ParallelConfig

# Tensor model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Pipeline model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None

# A list of global ranks for each pipeline group to ease calculation of the
# source rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

_MULTI_WORKER = None


def initialize_model_parallel(tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, multi_worker: bool=False) -> None:
    # Get world size and rank. Ensure some consistencies.
    global _MULTI_WORKER
    if multi_worker:
        _MULTI_WORKER = True
    else:
        _MULTI_WORKER = False

        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()

        if (world_size != tensor_model_parallel_size * pipeline_model_parallel_size):
            raise RuntimeError(
                f"world_size ({world_size}) is not equal to "
                f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
                f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

        num_tensor_model_parallel_groups: int = (world_size // tensor_model_parallel_size)                            
        num_pipeline_model_parallel_groups: int = (world_size // pipeline_model_parallel_size)                 
        rank = torch.distributed.get_rank()

        # Build the tensor model-parallel groups.
        global _TENSOR_MODEL_PARALLEL_GROUP
        assert _TENSOR_MODEL_PARALLEL_GROUP is None, (
            "tensor model parallel group is already initialized")
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)            
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group

        # Build the pipeline model-parallel groups.
        global _PIPELINE_MODEL_PARALLEL_GROUP
        global _PIPELINE_GLOBAL_RANKS
        assert _PIPELINE_MODEL_PARALLEL_GROUP is None, (
            "pipeline model parallel group is already initialized")
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks



def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (_TENSOR_MODEL_PARALLEL_GROUP is not None and _PIPELINE_MODEL_PARALLEL_GROUP is not None)
            


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, ("tenosr model parallel group is not initialized")
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, ("pipeline model parallel group is not initialized")
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MULTI_WORKER
    if _MULTI_WORKER:
        return 1
    else:
        return torch.distributed.get_world_size( group=get_tensor_model_parallel_group())

       


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())
        


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MULTI_WORKER
    if _MULTI_WORKER:
        return 0
    else:
        return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())
        


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, ("Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, ("Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, ("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, ("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None
    global _MULTI_WORKER
    _MULTI_WORKER = None
