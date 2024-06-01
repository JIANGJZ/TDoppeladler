# FuseSpill
This is the computational artifact of paper `FuseSpill: Efficient KV Cache Spillover Management on Memory-Constrained GPU for LLM Inference`.
FuseSpill an runtime system developped based on vllm for large model inference that showcases efficient KV Cache spillover handling. By leveraging auxiliary devices for managing the decoding phase of spillover sequences, FuseSpill incorporates adaptive offloading, asynchronous submission, and length perception techniques to enhance device utilization and boost the throughput of LLM inference.

## Environment
### Hardware
We perform experiments on two multi-GPU platforms.
One equipped with two Nvidia A10 GPUs, another equipped with one Nvidia GTX 4090 GPU, and one Nvidia GTX 3090 GPU. CPU of both platforms is Intel Xeon Platinum 8358P. The host memory capacity is 500GB. There is no nvlink between GPUs. CPUs and GPUs are connected via PCIe 4.0 with 16GB bandwidth.

### Software
- python v3.8
- pytorch v2.1.2
- CUDA 12.1
- Ray v2.5.1
- vLLM v0.2.7

## FuseSpill install documents

```bash
# Install dependencies.
pip install -r requirements.txt

# Install FuseSpill.
pip install -e .
```

or use with docker
```bash
# Build with docker
cp Dockerfile ..
cd ..
docker build -t test .

# Run the docker
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/models:/models/ test
```

## Usage
To execution command of the demo is to run the benchmark.sh script in the project directory.

The default dataset can be find at:
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json

The hyper-parameters of FuseSpill can be set in benchmarks/benchmark_throughput.py:
```bash
parser.add_argument("--sorted_request", action="store_true", help="is sort request, store_false is true")
parser.add_argument("--multi-worker", action="store_false", help="is use multiworker, store_false is true")
parser.add_argument("--worker-use-ray", action="store_true", help="is use ray, store_true is False")
parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="is enable tensor parallism of vllm")
parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to process.")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help='the fraction of GPU memory')
parser.add_argument('--swap-space', type=int, default=32, help='CPU swap space size (GiB) per GPU')   
parser.add_argument("--model", type=str, default="/home/TDoppeladler/model/vicuna-7b")
parser.add_argument("--tokenizer", type=str, default="/home/TDoppeladler/model/vicuna-7b")
parser.add_argument("--load-format", type=str, default="auto")
parser.add_argument("--disable_log_stats", action="store_false", help="is disable stats, store_false is true")
parser.add_argument("--response_aware", action="store_true", help="is enable response_aware kv cache swap")
parser.add_argument("--async_submit", type=int, help="async submit queue, synchronous submission when queue length is 1 ")
```
