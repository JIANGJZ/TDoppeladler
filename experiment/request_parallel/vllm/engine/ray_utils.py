from typing import Optional, Tuple, TYPE_CHECKING

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import get_open_port, is_hip

logger = init_logger(__name__)

try:
    import ray
    from ray.air.util.torch_dist import TorchDistributedWorker

    class RayWorkerVllm(TorchDistributedWorker):
        """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, init_cached_hf_modules=False) -> None:
            if init_cached_hf_modules:
                from transformers.dynamic_module_utils import init_hf_modules
                init_hf_modules()
            self.worker = None

        def init_worker(self, worker_init_fn):
            self.worker = worker_init_fn()

        def __getattr__(self, name):
            return getattr(self.worker, name)

        def execute_method(self, method, *args, **kwargs):
            executor = getattr(self, method)
            return executor(*args, **kwargs)

except ImportError as e:
    logger.warning(f"Failed to import Ray with {e!r}. "
                   "For distributed inference, please install Ray with "
                   "`pip install ray pandas pyarrow`.")
    ray = None
    TorchDistributedWorker = None
    RayWorkerVllm = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
):
    """Initialize the distributed cluster probably with Ray.

    Args:
        parallel_config: The configurations for parallel execution.
        engine_use_ray: Whether to use Ray for async engine.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    Returns:
        A tuple of (`distributed_init_method`, `placement_group`). The
        `distributed_init_method` is the address for initializing the
        distributed backend. `placement_group` includes the specification
        of the resources for each distributed worker.
    """
    if parallel_config.worker_use_ray or engine_use_ray:
        if ray is None:
            raise ImportError(
                "Ray is not installed. Please install Ray to use distributed "
                "serving.")
        # Connect to a ray cluster.
        if is_hip():
            ray.init(address=ray_address,
                     ignore_reinit_error=True,
                     num_gpus=parallel_config.world_size)
        else:
            ray.init(address=ray_address, ignore_reinit_error=True)