# GEMM Benchmark

### Memory Access Trick of Bag

- [x] float4/Uint4
- [x] ThreadBlock Swizzle
- [x] WrapBlock Swizzle
- [x] Load Matrix
- [x] Async Copy
- [x] Share Mem Bank Free

### Computing Trick of Bag

- [x] MultiStage Buffer



### TB128 & TB256

**design**: ![TB128&256](../asset/TB256_M16N8K16_FP16FP32.drawio.png)

**benchmark**: 
- Device: GTX3080TI
- CUDA Version: 11.8

![TB128K256](../asset/GEMM_TB128_K256_benchmark.jpg)

![TB256MNK](../asset/GEMM_TB128_MNK_B_benchmark.jpg)
