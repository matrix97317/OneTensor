# GEMM Benchmark

### Memory Access Trick of Bag

- [x] float4/uint4
- [x] ThreadBlock Swizzle
- [ ] WrapBlock Swizzle
- [ ] load-matrix
- [ ] async_copy

### Computing Trick of Bag

- [x] Double buffer

For M,N >>> INF, K = 16, cublas TB=128.

### V1

design: ![TB32](../asset/TB32_M16N8K16_FP16FP32.drawio.png)

benchmark: 