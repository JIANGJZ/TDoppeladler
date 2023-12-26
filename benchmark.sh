#!/bin/bash

main () {

#   cd /home/jiangjz/llm/TDoppeladler

  positional_args=()

  model_path="/home/jiangjz/llm/TDoppeladler/model/vicuna-7b-v1.5"
  dataset_path="/home/jiangjz/llm/TDoppeladler/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
  dataset_path_modified=0

  while [[ $# -gt 0 ]]; do
    case $1 in
      "-h"|"--help")
        python3 /home/jiangjz/llm/TDoppeladler/benchmarks/benchmark_throughput.py --help
        return 0
        ;;
      "--dataset")
        dataset_path="$2"
        dataset_path_modified=1
        shift
        shift
        ;;
      "--model")
        model_path="$2"
        shift
        shift
        ;;
      *)
        positional_args+=("$1")
        shift
        ;;
    esac
  done

#   if [ ! -f "$dataset_path" ]; then
#     if [[ $dataset_path_modified -lt 1 ]]; then
#       cd /home/jiangjz/vllm-rocm/dataset
#       wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
#       cd /home/jiangjz/vllm-rocm
#     fi
#   fi

  python3 /home/jiangjz/llm/TDoppeladler/benchmarks/benchmark_throughput.py --dataset "$dataset_path" --model "$model_path" "${positional_args[@]}"
  return $?

}

main "$@"
exit $?
