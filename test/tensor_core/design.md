## High Performance Operator Design

### Memory Access Order
- Global mem -> L2 Cache -> Register -> Global mem
- Global mem -> L2 Cache -> L1 Cache -> Register -> Global mem
- Global mem -> L2 Cache -> Register -> Share mem -> Register -> Global mem
- Global mem -> L2 Cache -> L1 Cache -> Register -> Share mem -> Register -> Global mem
- Global mem -> L2 Cache -> Share mem -> Register -> Global mem (Ampere)
### Memory Access Trick of Bag
- float4/uint4
- ThreadBlock Swizzle
- WrapBlock Swizzle
- load-matrix
- async_copy
### Computing Pipeline
- Load Data -> Compute Data -> Store Data

### Computing Trick of Bag
- Double buffer

### Benchmark

| --Op Name-- | --benchmark---                          |
| ----------- | --------------------------------------- |
| GEMM        | [benchmark](./md_doc/gemm_benchmark.md) |
|             |                                         |


### Reference
- https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master
- https://github.com/nicolaswilde/cuda-tensorcore-hgemm/tree/master