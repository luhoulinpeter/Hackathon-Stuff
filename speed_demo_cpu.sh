#!/bin/bash

weights_and_biases="weights_and_biases.txt"
input_tensor_dir="tensors"
repeats=1

if [ "$#" -ge 1 ]; then
    weights_and_biases=$1
fi
if [ "$#" -ge 2 ]; then
   input_tensor_dir=$2
fi
if [ "$#" -ge 3 ]; then
   repeats=$3
fi

binary="bin/speed_cpu"

if [ ! -f "$binary" ]; then
    echo "Binary $binary not found!"
    exit 1
fi

start_time=$(date +%s)
./$binary "$weights_and_biases" "$input_tensor_dir" "$repeats"

end_time=$(date +%s)
execution_time=$((end_time - start_time))

if [ ! -f "results.csv" ]; then
    echo "Error: results.csv not found!"
    exit 1
else
    echo "results.csv found!"
fi

echo "Execution time: $execution_time seconds"
