from typing import List, Optional, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq" and "squeezellm". If None, we first check
            the `quantization_config` attribute in the model config file. If
            that is None, we assume the model weights are not quantized and use
            `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        request_parallel_size: int=4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            request_parallel_size=request_parallel_size,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            **kwargs,
        )
        self.engines: List[LLMEngine] = []
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        initialize_cluster(parallel_config)
        distributed_init_method = None
        print(parallel_config.world_size, distributed_init_method)
        
        for i in range(request_parallel_size):
            engine = LLMEngine.remote(*engine_configs,
                     distributed_init_method,
                     None,
                     log_stats=not engine_args.disable_log_stats)
            self.engines.append(engine)
        
        # self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()
        self.request_parallel_size = request_parallel_size
        self.count = 0
    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(prompt, sampling_params, token_ids)
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.engines[self.count % self.request_parallel_size].add_request.remote(request_id, prompt, sampling_params,
                                    prompt_token_ids)
        self.count = self.count + 1

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = sum(ray.get(self.engines[i].get_num_unfinished_requests.remote()) for i in range(self.request_parallel_size))
            pbar = tqdm(total=num_requests, desc="Processed prompts")

        outputs = []
        active_tasks = {self.engines[i].step.remote(): i for i in range(self.request_parallel_size)}
        while active_tasks:
            done_ids, _ = ray.wait(list(active_tasks.keys()), num_returns=1)
            done_id = done_ids[0]
            results = ray.get(done_id)

            engine_index = active_tasks.pop(done_id)
            for result in results:
                if result.finished:
                    outputs.append(result)
                    if use_tqdm:
                        pbar.update(1)

            # Check if the engine has more work to do and if so, start a new task
            if ray.get(self.engines[engine_index].has_unfinished_requests.remote()):
                new_task = self.engines[engine_index].step.remote()
                active_tasks[new_task] = engine_index

            # Recheck the unfinished requests status after processing current batch
            if not active_tasks:  # Only re-check if there are no active tasks
                has_unfinished_requests = any(ray.get(self.engines[i].has_unfinished_requests.remote()) for i in range(self.request_parallel_size))
                if has_unfinished_requests:
                    active_tasks = {self.engines[i].step.remote(): i for i in range(self.request_parallel_size) if ray.get(self.engines[i].has_unfinished_requests.remote())}

        if use_tqdm:
            pbar.close()

        # Sort the outputs by request ID.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
