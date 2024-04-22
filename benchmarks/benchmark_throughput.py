"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple
from vllm import LLM, SamplingParams 

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase)         
from tqdm import tqdm


def sample_requests(dataset_path: str, num_requests: int, tokenizer: PreTrainedTokenizerBase, fixed_output_len: Optional[int],) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        # if prompt_len < 4 or output_len < 50 or output_len > 60:
        #     continue
        # if prompt_len < 4 or output_len < 80:
        #     continue
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 1024 or prompt_len + output_len > 1512:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def run_vllm(requests: List[Tuple[str, int, int]],  model: str, tokenizer: str, quantization: Optional[str], 
            tensor_parallel_size: int, seed: int, n: int, use_beam_search: bool, trust_remote_code: bool, dtype: str,
            max_model_len: Optional[int], enforce_eager: bool, multi_worker:bool, worker_use_ray:bool,
            gpu_memory_utilization:float, swap_space:int, num_prompts:int, load_format: str, sorted_request:bool, 
            disable_log_stats:bool) -> float:

    llm = LLM(model=model, tokenizer=tokenizer, quantization=quantization, tensor_parallel_size=tensor_parallel_size, 
            seed=seed, trust_remote_code=trust_remote_code, dtype=dtype, max_model_len=max_model_len, 
            enforce_eager=enforce_eager, multi_worker=multi_worker, worker_use_ray=worker_use_ray,
            gpu_memory_utilization=gpu_memory_utilization, swap_space=swap_space, num_prompts=num_prompts, 
            load_format=load_format, sorted_request=sorted_request, disable_log_stats=disable_log_stats)
        
    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(n=n, temperature=0.0 if use_beam_search else 1.0, top_p=1.0, use_beam_search=use_beam_search, ignore_eos=True, max_tokens=output_len, )
        llm._add_request(prompt=prompt, prompt_token_ids=None, sampling_params=sampling_params,)
            
    start = time.perf_counter()
    llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len) for _ in range(args.num_prompts)]      
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.output_len)
                                   
    elapsed_time = run_vllm(requests, args.model, args.tokenizer, args.quantization, args.tensor_parallel_size, 
                    args.seed, args.n, args.use_beam_search, args.trust_remote_code, args.dtype, args.max_model_len, 
                    args.enforce_eager, args.multi_worker, args.worker_use_ray, args.gpu_memory_utilization,
                    args.swap_space, args.num_prompts, args.load_format, args.sorted_request, args.disable_log_stats)
     
                
    total_num_tokens = sum(prompt_len + output_len for _, prompt_len, output_len in requests)                           
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, {total_num_tokens / elapsed_time:.2f} tokens/s")
    total_prompt_tokens = sum(prompt_len for _, prompt_len, output_len in requests)
    print(f"PromptThroughput: {total_prompt_tokens / elapsed_time:.2f} tokens/s")
    total_output_tokens = sum(output_len for _, prompt_len, output_len in requests)
    print(f"OutputThroughput: {total_output_tokens / elapsed_time:.2f} tokens/s")         
    print(f"Total Time : {elapsed_time:.2f} s")         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")    
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")     
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")              
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request. Overrides the output length from the dataset.")  
    parser.add_argument('--quantization', '-q', choices=['awq', 'gptq', 'squeezellm', None], default=None)
    parser.add_argument("--n", type=int, default=1, help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument('--trust-remote-code', action='store_false', help='trust remote code from huggingface')
    parser.add_argument('--max-model-len', type=int, default=None, help='Maximum length of a sequence (including prompt and output). If None, will be derived from the model.')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], help='data type for model weights and activations. \
        The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.')
    parser.add_argument("--enforce-eager", action="store_true", help="enforce eager execution")

    #如果要用sepration去处理则把multi-worker 的action="store_false"， worker-use-ray为true
    #把worker-use-ray设置为store_false， 然后multi-worker设置为store true， tensor-parallel-size可测试单卡用ray的性能
    #用ray做request并行在experiment文件夹
    #用ray的时候profile出来的可用内存空间比不用ray大概少了10%，做对比实验需要把profile出来的可用的内存空间对齐
    #--swap-space 不能设置太大，超过可使用用内存的70%
    parser.add_argument("--multi-worker", action="store_false", help="is use multiworker, store_false is true")
    parser.add_argument("--worker-use-ray", action="store_true", help="is use ray, store_true is False")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=1500, help="Number of prompts to process.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help='the fraction of GPU memory')
    parser.add_argument('--swap-space', type=int, default=32, help='CPU swap space size (GiB) per GPU')   
    parser.add_argument("--model", type=str, default="/home/users/jiangjz/llm/TDoppeladler/model/vicuna-7b")
    parser.add_argument("--tokenizer", type=str, default="/home/users/jiangjz/llm/TDoppeladler/model/vicuna-7b")
    parser.add_argument("--load-format", type=str, default="auto")
    parser.add_argument("--disable_log_stats", action="store_false", help="is disable stats, store_false is true")
    parser.add_argument("--response_aware", action="store_true", help="is enable response_aware kv cache swap")
    parser.add_argument("--async_submit", type=int, help="async submit queue, synchronous submission when queue length is 1 ")
    parser.add_argument("--sorted_request", action="store_true", help="is sort request, store_false is true")
               
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
                             
    main(args)
