#!/bin/bash

main () {

#   cd /home/jiangjz/llm/TDoppeladler

  positional_args=()

  path_prefix="/workspace/fuse"

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

  python3 $benchmark_path --dataset "$dataset_path"  "${positional_args[@]}"
  return $?

}

main "$@"
exit $?
