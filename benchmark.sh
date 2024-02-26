#!/bin/bash

main () {

#   cd /home/jiangjz/llm/TDoppeladler

  positional_args=()

  path_prefix="/home/users/jiangjz/llm/TDoppeladler"

  model_path=$path_prefix"/model/vicuna-7b-v1.5"
  dataset_path=$path_prefix"/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
  dataset_path_modified=0

  benchmark_path=$path_prefix"/benchmarks/benchmark_throughput.py" 

  while [[ $# -gt 0 ]]; do
    case $1 in
      "-h"|"--help")
        python3 $benchmark_path --help
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

  # python3 $benchmark_path --dataset "$dataset_path" --model "$model_path" "${positional_args[@]}"
  python3 $benchmark_path --dataset "$dataset_path"  "${positional_args[@]}"
  return $?

}

main "$@"
exit $?
