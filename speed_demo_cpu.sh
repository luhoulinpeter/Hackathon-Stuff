#!/bin/bash

weights_and_biases=$1
input_tensor_dir=$2

binary="bin/speed_cpu"

if [ ! -f "$binary" ]; then
    echo "Binary $binary not found!"
    exit 1
fi

start_time=$(date +%s)
./$binary "$weights_and_biases" "$input_tensor_dir"

end_time=$(date +%s)
execution_time=$((end_time - start_time))

if [ ! -f "results.csv" ]; then
    echo "Error: results.csv not found!"
    exit 1
else
    echo "results.csv found!"
fi

echo "Execution time: $execution_time seconds"
