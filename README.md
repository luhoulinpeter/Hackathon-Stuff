StartHack Hackathon HPC Neural Network on Digit Recognition
===========================================================

**High Performance Computing Track Sponsered by QDX**

Authors: Carlvince Tan, Lucas Yu, Peter Lu, Volodymyr Kazmirchuk

## Overview
This code includes two implementations of a Neural Network, based on pre-trained weights and biases obtained from https://github.com/talo/hackathon-go-brrr: 
1. Using just the cpu
2. Using the gpu on top of the cpu to maximise time efficiency for matrix multiplication operations

## Dependencies 
- CUDA version 12.5
- OpenMP version 5.2
- Eigen version 3.4.0

## Optimisations
- Matrices are stored/parsed as vectors
- Use of multi-threading to improve time efficiency

## Platform
This code was written to be executed on:
- Ubuntu 22.04
- 8x A100 GPUs each with 80GB memory
- CPU
    - Model name: AMD EPYC 7J13 64-Core Processor
    - Family: 25
    - Model: 1
    - Thread(s) per core: 1
    - Core(s) per socket: 1
    - Socket(s): 240
- Caches (sum of all):
    - L1d: 15 MiB (240 instances)
    - L1i: 15 MiB (240 instances)
    - L2: 120 MiB (240 instances)
    - L3: 3.8 GiB (240 instances)
- NUMA:
    - Node(s): 2
    - node0 CPU(s): 0-119
    - node1 CPU(s): 120-239
