import importlib
from typing import List, Optional, Type
import torch.nn as nn
from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)

# Architecture -> (module, class).
_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
}


class CPUModelRegistry:
    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None

        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"vllm.model_executor.models_cpu.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())


__all__ = [
    "ModelRegistry",
]
