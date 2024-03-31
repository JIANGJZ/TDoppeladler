from typing import List, Optional, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
import asyncio


class LLM:
    def __init__(self, model: str, tokenizer: Optional[str] = None, tokenizer_mode: str = "auto", trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,  dtype: str = "auto", quantization: Optional[str] = None, revision: Optional[str] = None, 
        tokenizer_revision: Optional[str] = None, seed: int = 0, gpu_memory_utilization: float = 0.9, swap_space: int = 8,
        enforce_eager: bool = False, max_context_len_to_capture: int = 8192, multi_worker: bool=False, 
        worker_use_ray: bool=False, num_prompts:int = 0, load_format:str="dummy", sorted_request:bool=False, 
        disable_log_stats: bool=True, **kwargs,) -> None: 
    
        engine_args = EngineArgs(model=model, tokenizer=tokenizer, tokenizer_mode=tokenizer_mode, trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size, dtype=dtype, quantization=quantization, revision=revision, tokenizer_revision=tokenizer_revision, 
            seed=seed, gpu_memory_utilization=gpu_memory_utilization, swap_space=swap_space, enforce_eager=enforce_eager, 
            max_context_len_to_capture=max_context_len_to_capture, multi_worker=multi_worker, worker_use_ray=worker_use_ray, 
            num_prompts=num_prompts, load_format=load_format, sorted_request=sorted_request, disable_log_stats=disable_log_stats, 
            **kwargs, )
  
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(self, prompts: Optional[Union[str, List[str]]] = None, sampling_params: Optional[SamplingParams] = None, prompt_token_ids: Optional[List[List[int]]] = None, use_tqdm: bool = True,) -> List[RequestOutput]:
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids must be the same.")
                             
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[ i]
            self._add_request(prompt, sampling_params, token_ids)
        return self._run_engine(use_tqdm)

    def _add_request(self, prompt: Optional[str], sampling_params: SamplingParams, prompt_token_ids: Optional[List[int]], ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params, prompt_token_ids)
                                    
    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")

        outputs: List[RequestOutput] = []
        outputs_requestid: List[str] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs =  self.llm_engine.step()
            if len(step_outputs) > 0:
                self.llm_engine.clear_step_output()
                for output in step_outputs:
                    if output.finished and (output.request_id not in outputs_requestid):
                        print (output.outputs[0].text)
                        outputs.append(output)
                        outputs_requestid.append(output.request_id)
                        if use_tqdm:
                            pbar.update(1)

        if self.llm_engine.parallel_config.multi_worker:
            self.llm_engine.exit_clear()
        if use_tqdm:
            pbar.close()
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
